#!/usr/bin/env python3
"""
test.py – inspect an L₀-pruned checkpoint.

Usage
-----
$ python test.py --ckpt /path/to/last.ckpt [--device cuda]

What it reports
---------------
1. #zero gates  : how many input units / output channels are switched *off*
2. #zero weights: how many weight *elements* are 0 after masking
3. sparsity     : (zero-weights / total-weights) in every L0 layer
4. global sparsity over the whole network

python test.py \
    --pruned_ckpt     $PWD/exps/train100_exp/ckpts/last.ckpt \
    --pretrained_ckpt $PWD/pretrained/hubert-base-ls960.fairseq.pth \
    --device cuda
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

# ────────────────────────────────────────────────────────────────────────────────
# import your layers & optionally your LightningModule
# (change "l0_layers" and "my_lightning" to match your repo)
from L0_regularization.l0_layers import L0Dense, L0Conv1d, L0Conv2d     # ← the code you supplied
try:
    from my_lightning import LitModel                  # ← your lightning class
except ModuleNotFoundError:
    LitModel = None                                    # we may fall back to torch.load

# constants copied from the layer definitions
limit_a, limit_b = -0.1, 1.1

# ────────────────────────────────────────────────────────────────────────────────
def gate_from_qz_loga(qz_loga: torch.Tensor) -> torch.Tensor:
    """Deterministic Hard-Concrete gate used at test time (no randomness)."""
    pi  = torch.sigmoid(qz_loga)                  # probability of “on”
    gate = torch.clamp(pi * (limit_b - limit_a) + limit_a, 0.0, 1.0)
    return gate                                   # same shape as `qz_loga`

def layer_stats(layer) -> tuple[int, int, int, int]:
    """
    Return (#zero-gates, #gates, #zero-weights-after-mask, #weights) for one L0 layer.
    """
    # ---- 1. derive the deterministic mask ------------------------------------
    g = gate_from_qz_loga(layer.qz_loga.detach())           # (dim_z,)
    mask_vec = (g > 1e-3).to(torch.bool)                       # True = kept
    num_gate_zero = mask_vec.numel() - mask_vec.sum().item()

    # ---- 2. broadcast to weight tensor & count zeros -------------------------
    if isinstance(layer, L0Dense):
        mask = g.view(-1, 1)                                # (in,1)
    elif isinstance(layer, L0Conv1d):
        mask = g.view(-1, 1, 1)                             # (out,1,1)
    elif isinstance(layer, L0Conv2d):
        mask = g.view(-1, 1, 1, 1)                          # (out,1,1,1)
    else:                                                   # should not happen
        raise TypeError(f"Unexpected layer type {type(layer)}")

    w_masked = layer.weights.detach() * mask
    num_weight_zero = w_masked.numel() - w_masked.nonzero().size(0)

    return (num_gate_zero, mask_vec.numel(), num_weight_zero, w_masked.numel())

# ────────────────────────────────────────────────────────────────────────────────
def inspect(model: torch.nn.Module) -> None:
    header = "{:<50} {:>10} {:>10} {:>10}".format(
        "layer (name/type)", "zero-g", "zero-w", "sparsity"
    )
    print(header)
    print("-" * len(header))

    total_zeros, total_elems = 0, 0

    for name, module in model.named_modules():
        if isinstance(module, (L0Dense, L0Conv1d, L0Conv2d)):
            z_g, n_g, z_w, n_w = layer_stats(module)
            sparsity = z_w / n_w
            total_zeros += z_w
            total_elems += n_w

            print(f"{name:<50} {z_g:10d} {z_w:10d} {sparsity:9.3%}")

    print("-" * len(header))
    print(f"{'GLOBAL':<50} {'-':>10} {total_zeros:10d} {(total_zeros/total_elems):9.3%}")

def _extract_student_state_dict(raw):
    # ── unwrap obvious nested structures ────────────────────────────────────
    if isinstance(raw, dict):
        # ‘student’ or ‘student_model’ nest
        for k in ("student", "student_model"):
            if k in raw and isinstance(raw[k], (dict, torch.nn.Module)):
                raw = raw[k]

        # Hugging-Face / Lightning style
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]

    # ── flat dict with prefixes (‘student_model.’ or ‘student.’) ─────────────
    if isinstance(raw, dict):
        prefixed = {
            k.split(".", 1)[1]: v
            for k, v in raw.items()
            if k.startswith(("student_model.", "student."))
        }
        if prefixed:                 # found at least one student weight
            return prefixed

    # ── already clean? ──────────────────────────────────────────────────────
    if isinstance(raw, dict):
        return raw                    # hope they already match the target net

    raise RuntimeError("Could not locate student weights in the pruned checkpoint.")

# ────────────────────────────────────────────────────────────────────────────────
def load_model(
    pruned_ckpt: Path,
    pretrained_ckpt: Path,
    device: str = "cpu",
    *,
    droprate_init: float = 0.5,
    l0_temperature: float = 2.0 / 3.0,
    l0_weight_decay: float = 1.0,
):

    # ------------------------------------------------------------------ import
    from wav2vec2.model import wav2vec2_model          # your model factory
    from run import _wrap_with_l0                 # injects L₀ layers

    # ------------------------------------------------------------------ 1) load *pre-trained* base model
    base_ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    if not {"config", "state_dict"} <= set(base_ckpt):
        raise RuntimeError(
            f"{pretrained_ckpt} must contain both 'config' and 'state_dict' keys "
            f"(got {list(base_ckpt.keys())})"
        )

    model = wav2vec2_model(**base_ckpt["config"])
    model.load_state_dict(base_ckpt["state_dict"], strict=False)

    # ------------------------------------------------------------------ 2) wrap with L₀
    _wrap_with_l0(
        model,
        droprate_init=droprate_init,
        l0_temperature=l0_temperature,
        l0_weight_decay=l0_weight_decay,
    )

    # ------------------------------------------------------------------ 3) load *pruned* weights (already contains qz_loga, etc.)
    raw = torch.load(pruned_ckpt, map_location="cpu")
    student_sd = _extract_student_state_dict(raw)

    missing, unexpected = model.load_state_dict(student_sd, strict=False)
    if missing or unexpected:
        print(
            f"[load_model] state-dict mismatch  –  missing {len(missing)} keys, "
            f"unexpected {len(unexpected)} keys"
        )

    return model.to(device).eval()


# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inspect L₀-pruned HuBERT/W2V2 checkpoints")
    parser.add_argument(
        "--pruned_ckpt",
        required=True,
        type=Path,
        help="checkpoint produced *after* pruning / L₀ training (contains qz_loga, etc.)",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        required=True,
        type=Path,
        help="baseline HuBERT / wav2vec2 checkpoint (no L₀ wrappers)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help='use "cuda" to run on the current default GPU',
    )
    # keep these defaults identical to training
    parser.add_argument("--droprate_init", type=float, default=0.5)
    parser.add_argument("--l0_temperature", type=float, default=2.0 / 3.0)
    parser.add_argument("--l0_weight_decay", type=float, default=1.0)

    args = parser.parse_args()

    model = load_model(
        pruned_ckpt=args.pruned_ckpt,
        pretrained_ckpt=args.pretrained_ckpt,
        device=args.device,
        droprate_init=args.droprate_init,
        l0_temperature=args.l0_temperature,
        l0_weight_decay=args.l0_weight_decay,
    )
    inspect(model)

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
