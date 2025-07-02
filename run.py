#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Iterable, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_lite.utilities.rank_zero import _get_rank

# third-party L₀ layers --------------------------------------------------
from acphubert.module.l0_layers import L0Conv1d, L0Dense
from acphubert.model import _wrap_with_l0

# local project imports --------------------------------------------------
from lightning import DistillLoss, DistillModule
from acphubert.wav2vec2.model import wav2vec2_model

_LG = logging.getLogger(f"{__name__}:{_get_rank()}")

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _init_layer_transform(module: nn.Linear):
    """Initialises a linear mapping as near-identity (student→teacher)."""
    module.weight.data.copy_(torch.eye(len(module.weight)))  # type: ignore[arg-type]
    module.bias.data.zero_()

def _collect_l0_modules(module: nn.Module) -> list[L0Dense | L0Conv1d]:
    """Return a flat ``list`` of *all* ``L0Dense`` / ``L0Conv1d`` instances
    within *module* (recursive).  Convenient for summing regularisation terms
    or logging the number of prunable layers.
    """
    return [m for m in module.modules() if isinstance(m, (L0Dense, L0Conv1d))]

# ----------------------------------------------------------------------
# LightningModule extension to add L₀ penalty
# ----------------------------------------------------------------------

class DistillLitModule(DistillModule):
    """Extends project’s *DistillModule* to add L₀ penalties automatically."""

    def __init__(self, 
        *args, 
        l0_lambda: float,
        ac_weight: float = 1.0,
        ac_max_tau: int = 100, 
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("l0_lambda", "ac_weight", "ac_max_tau")
        self.l0_modules: List[nn.Module] = _collect_l0_modules(self.student_model)

# ----------------------------------------------------------------------
# training routine
# ----------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:  # noqa: C901 – top-level script
    pl.seed_everything(2022, workers=True)

    # lightning callbacks --------------------------------------------------
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=args.exp_dir / "ckpts",
            save_last=True,
            save_top_k=0,
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        callbacks=callbacks,
        max_steps=args.max_updates,
        strategy="ddp",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accumulate_grad_batches=args.accum_grad,
        replace_sampler_ddp=False,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=args.clip_norm,
        log_every_n_steps=args.log_interval,
        precision=args.precision,
    )

    # teacher --------------------------------------------------------------
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher_model = wav2vec2_model(**teacher_ckpt["config"])
    _LG.info("Teacher model: %s", teacher_model.__class__.__name__)
    teacher_model.load_state_dict(teacher_ckpt["state_dict"], strict=False)
    teacher_model.eval().requires_grad_(False)

    # student --------------------------------------------------------------
    student_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    student_model = wav2vec2_model(**student_ckpt["config"])
    _LG.info("Student model: %s", student_model.__class__.__name__)
    student_model.load_state_dict(student_ckpt["state_dict"], strict=False)

    # apply L₀ wrappers BEFORE distillation layers so dimensions stay valid
    _wrap_with_l0(
        student_model,
        droprate_init=args.droprate_init,
        l0_temperature=args.l0_temperature,
        l0_weight_decay=args.l0_weight_decay,
    )
    _LG.info("Number of modules: %d",
             len([m for m in teacher_model.modules() if isinstance(m, (nn.Linear, nn.Conv1d))]))
    _LG.info("Wrapped student with L₀ gates; total prunable modules: %d",
             len(_collect_l0_modules(student_model)))

    s_dim = student_model.encoder.feature_projection.projection.out_features
    t_dim = teacher_model.encoder.feature_projection.projection.out_features

    # if args.distill_mode == "layer2layer":
    #     proj_layers = nn.ModuleList()
    #     for group in distill_layer_groups:
    #         shared = nn.Linear(s_dim, t_dim)
    #         _init_layer_transform(shared)
    #         proj_layers.extend(shared for _ in group)
    # elif args.distill_mode == "predlayer":
    #     proj_layers = nn.ModuleList(
    #         nn.Sequential(nn.Linear(s_dim, t_dim), nn.GELU()) for _ in distill_layers
    #     )
    # else:
    #     raise ValueError(f"Invalid distill_mode: {args.distill_mode}")

    # criterion ------------------------------------------------------------
    l2_weight=1             # weight for L2 loss
    l1_weight=1             # weight for L1 loss
    cos_weight=0            # weight for cosine similarity
    cos_type="raw"            # "raw", "log_sig"
    distill_criterion = DistillLoss(
        l2_weight=l2_weight,
        l1_weight=l1_weight,
        cos_weight=cos_weight,
        cos_type=cos_type,
    )

    module = DistillLitModule(
        teacher_model=teacher_model,
        student_model=student_model,
        distill_mode=args.distill_mode,
        distill_layers=None,
        distill_linear_projs=None,
        distill_loss=distill_criterion,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
        use_reg=True,
        tsv_dir=args.tsv_dir,
        train_subset=args.train_subset,
        seconds_per_batch=args.seconds_per_batch,
        num_workers=args.num_workers,
        reg_learning_rate=0.02,
        target_sparsity=0.75,
        sparsity_warmup_updates=30000,
        l0_lambda=args.l0_lambda,
    )

    trainer.fit(module, ckpt_path=args.resume_checkpoint)

    # ------------------------------------------------------------------
    # optional export: remove gates and save a dense pruned checkpoint
    # ------------------------------------------------------------------
    if args.export_path is not None:
        _LG.info("Exporting dense-sparse checkpoint → %s", args.export_path)
        module.student_model.eval()
        for m in _collect_l0_modules(module.student_model):
            m.prune()  # permanently apply mask & drop auxiliary params
        torch.save(
            {
                "config": student_ckpt["config"],
                "state_dict": module.student_model.state_dict(),
            },
            args.export_path,
        )

# ----------------------------------------------------------------------
# CLI arguments
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("HuBERT distillation with L₀ unstructured pruning")
    # paths ---------------------------------------------------------------
    p.add_argument("--exp-dir", type=pathlib.Path, default="exps/train100_exp")
    p.add_argument("--teacher-ckpt", type=pathlib.Path, default="pretrained/hubert-base-ls960.fairseq.pth")
    p.add_argument("--student-ckpt", type=pathlib.Path, default="pretrained/hubert-base-ls960.fairseq.pth")
    p.add_argument("--resume-checkpoint", type=pathlib.Path, default=None)
    p.add_argument("--export-path", type=pathlib.Path, default=None,
                   help="Save a dense checkpoint with gates removed at the end.")

    # training ------------------------------------------------------------
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--accum-grad", type=int, default=1)
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--warmup-updates", type=int, default=30000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=2e-2)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--log-interval", type=int, default=100)

    # dataset -------------------------------------------------------------
    p.add_argument("--tsv-dir", type=pathlib.Path, default="data/librispeech")
    p.add_argument("--train-subset", type=str, default="train100")
    p.add_argument("--seconds-per-batch", type=float, default=160.0)
    p.add_argument("--num-workers", type=int, default=8)

    # distillation --------------------------------------------------------
    p.add_argument("--distill-mode", choices=["layer2layer", "predlayer"], default="layer2layer")
    p.add_argument(
        "--distill-layers",
        type=str,
        default="0.2.4.6.8.10",  # groups separated by '.'
        help="Transformer layers (0-11) to align. Groups separated by '.' share a projection.",
    )
    p.add_argument("--l2-weight", type=float, default=1.0)
    p.add_argument("--l1-weight", type=float, default=0.0)
    p.add_argument("--cos-weight", type=float, default=0.0)
    p.add_argument("--cos-type", choices=["sample", "mean"], default="sample")

    # L₀ regularisation ---------------------------------------------------
    p.add_argument("--l0-lambda", type=float, default=0.1,
                   help="Strength of the L₀ penalty (λ in the paper).")
    p.add_argument("--droprate-init", type=float, default=0.2,
                   help="Initial drop probability (π₀).")
    p.add_argument("--l0-temperature", type=float, default=2.0 / 3.0)
    p.add_argument("--l0-weight-decay", type=float, default=1e-5,
                   help="Weight-decay term used *inside* the L₀ layer; has no effect on non-gated params.")

    return p.parse_args()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.exp_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    run_train(args)
