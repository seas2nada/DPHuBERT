"""Perform distillation for the pruned model."""

import logging
import pathlib
from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite.utilities.rank_zero import _get_rank

from lightning import (
    CTCModule,
    DistillModule,
    DistillLoss,
    CtcLoss,
)
from wav2vec2.model import (
    wav2vec2_model,
)

_LG = logging.getLogger(f"{__name__}:{_get_rank()}")


def run_train(args):
    pl.seed_everything(2022)

    # Callbacks
    lr_monitor = LearningRateMonitor()  # log learning rates for all param groups
    model_checkpoint = ModelCheckpoint(dirpath=args.exp_dir / "ckpts", verbose=True)   # only save the latest epoch
    callbacks = [lr_monitor, model_checkpoint]

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        callbacks=callbacks,
        max_steps=args.max_updates,
        strategy="ddp",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accumulate_grad_batches=args.accum_grad,
        replace_sampler_ddp=False,  # we use the custom distributed sampler for ddp
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=args.clip_norm,
        log_every_n_steps=args.log_interval,
        precision=args.precision,
    )
    
    # Create student model
    student_ckpt = torch.load(args.student_ckpt, map_location='cpu')
    student_model = wav2vec2_model(**student_ckpt["config"])
    _LG.info(f"Student model:\n{student_model}")
    student_result = student_model.load_state_dict(student_ckpt["state_dict"], strict=False)
    _LG.info(f"Load pretrained ckpt to student: missing {student_result.missing_keys}, unexpected {student_result.unexpected_keys}")

    # Create CtcLoss module
    ctc_loss_criterion = CtcLoss(
        args.tsv_dir / "dict.ltr.txt",
    )
    _LG.info(f"CTC loss module:\n{ctc_loss_criterion}")

    ctc_module = CTCModule(
        student_model=student_model,
        ctc_loss=ctc_loss_criterion,
        ctc_weight=args.ctc_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
        tsv_dir=args.tsv_dir,
        label_dir=args.label_dir,
        train_subset=args.train_subset,
        seconds_per_batch=args.seconds_per_batch,
        num_workers=args.num_workers,
    )

    trainer.fit(
        ctc_module, 
        ckpt_path=args.resume_checkpoint,
    )


def _parse_args():
    parser = ArgumentParser(
        description="Distill the pruned model.",
    )

    # dataset and dataloader related
    parser.add_argument(
        "--tsv_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the directory containing tsv files.",
    )
    parser.add_argument(
        "--label_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the directory containing label files.",
    )
    parser.add_argument(
        "--train_subset",
        default="train100",
        type=str,
        help="The subset name for training. (Default: 'train100')",
    )
    parser.add_argument(
        "--seconds_per_batch",
        default=87.5,
        type=float,
        help="Number of seconds of audio in a mini-batch. (Default: 87.5)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers in DataLoader."
    )

    # general training related
    parser.add_argument(
        "--resume_checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to the feature and label directories. (Default: None)",
    )
    parser.add_argument(
        "--exp_dir",
        type=pathlib.Path,
        help="Suffix of the exp directory name."
    )
    parser.add_argument(
        "--log_interval",
        default=50,
        type=int,
        help="Log interval in steps."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="The peak learning rate. (Default: 0.0001)",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay (L2 penalty) (Default: 0.0)",
    )
    parser.add_argument(
        "--warmup_updates",
        default=5000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 5000)",
    )
    parser.add_argument(
        "--max_updates",
        default=25000,
        type=int,
        help="Total number of training steps. (Default: 25000)",
    )
    parser.add_argument(
        "--clip_norm",
        default=10.0,
        type=float,
        help="The gradient norm value to clip. (Default: 10.0)",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--accum_grad",
        default=1,
        type=int,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Precision for training."
    )

    # distillation related
    parser.add_argument(
        "--teacher_ckpt",
        default=pathlib.Path("pretrained_ckpts/hubert-base-ls960.pth"),
        type=pathlib.Path,
        help="Path to the teacher model checkpoint."
    )
    parser.add_argument(
        "--student_ckpt",
        type=pathlib.Path,
        help="Path to the student model checkpoint (for initialization)."
    )
    parser.add_argument(
        "--distill_layers",
        default="0.4,8,12",
        type=str,
        help="Distill layer indices (use period to separate groups and comma to separate layers within a group)."
    )
    parser.add_argument(
        "--distill_mode",
        type=str,
        default="layer2layer",
        choices=["layer2layer", "predlayer"],
        help="Distill mode, either layer2layer or predlayer."
    )
    parser.add_argument(
        "--l2_weight",
        default=0.0,
        type=float,
        help="Weight of MSE loss."
    )
    parser.add_argument(
        "--l1_weight",
        default=1.0,
        type=float,
        help="Weight of L1 loss."
    )
    parser.add_argument(
        "--cos_weight",
        default=1.0,
        type=float,
        help="Weight of cosine similarity loss."
    )
    parser.add_argument(
        "--cos_type",
        default="raw",
        type=str,
        choices=["raw", "log_sig"],
        help="Type of the cosine similarity loss."
    )

    parser.add_argument(
        "--ctc_weight",
        default=0.001,
        type=float,
        help="Weight for ctc loss."
    )
    parser.add_argument(
        "--distill_weight",
        default=1,
        type=float,
        help="Weight for ctc loss."
    )

    parser.add_argument(
        "--mask_prob",
        default=0.75,
        type=float,
        help="Weight for ctc loss."
    )
    parser.add_argument(
        "--mask_channel_prob",
        default=0.65,
        type=float,
        help="Weight for ctc loss."
    )
    
    return parser.parse_args()


def _init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    if _get_rank() == 0:
        _LG.setLevel(logging.INFO)
    else:
        _LG.setLevel(logging.WARN)


def cli_main():
    _init_logger()
    args = _parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
