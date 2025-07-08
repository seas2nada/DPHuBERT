# model.py
"""Utilities for wrapping a *pre‑trained* wav2vec/hubert model with Hard‑Concrete
L₀ gates and loading a *pruned* checkpoint produced by our KD+pruning trainer.

This file is designed to be imported by s3prl `UpstreamExpert` classes but can
also be used standalone:

```bash
python - <<'PY'
from pathlib import Path
from model import load_model

m = load_model(
    pruned_ckpt = Path('student_pruned.ckpt'),
    pretrained_ckpt = Path('wav2vec_base.ckpt'),
    device = 'cuda',
)
print(sum(p.numel() for p in m.parameters())/1e6, 'M params')
PY
```
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Import the gating layers from your l0module implementation
# -----------------------------------------------------------------------------
from .module import L0Dense, L0Conv1d  # pylint: disable=import-error

__all__ = [
    "_wrap_with_l0",
    "load_model",
]

# -----------------------------------------------------------------------------
# 1. Recursive wrapper ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _wrap_with_l0(
    module: nn.Module,
    *,
    droprate_init: float = 0.5,
    l0_temperature: float = 2.0 / 3.0,
    l0_weight_decay: float = 1.0,
    collected: List[L0Dense | L0Conv1d] | None = None,
) -> None:
    """Recursively replace *Linear* / *Conv1d* layers with L₀‑gated variants.

    The original weights/biases are copied into the new layers so you can wrap a
    *pre‑trained* model and continue training / fine‑tuning immediately.
    All created L₀ layers are appended to *collected* for later bookkeeping.
    """

    if collected is None:
        collected = []

    for name, child in list(module.named_children()):
        # ── Fully‑connected ────────────────────────────────────────────────
        if isinstance(child, nn.Linear):
            wrapped = L0Dense(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                droprate_init=droprate_init,
                weight_decay=l0_weight_decay,
                temperature=l0_temperature,
            )
            with torch.no_grad():
                # (out,in) → (in,out)
                wrapped.weights.copy_(child.weight.data.t())
                if child.bias is not None:
                    wrapped.bias.copy_(child.bias.data)
            setattr(module, name, wrapped)
            collected.append(wrapped)

        # ── 1‑D convolution ────────────────────────────────────────────────
        elif isinstance(child, nn.Conv1d):
            wrapped = L0Conv1d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                dilation=child.dilation[0],
                groups=child.groups,
                bias=child.bias is not None,
                droprate_init=droprate_init,
                weight_decay=l0_weight_decay,
                temperature=l0_temperature,
            )
            with torch.no_grad():
                wrapped.weights.copy_(child.weight.data)
                if child.bias is not None:
                    wrapped.bias.copy_(child.bias.data)
            setattr(module, name, wrapped)
            collected.append(wrapped)

        # ── recurse ────────────────────────────────────────────────────────
        else:
            _wrap_with_l0(child, droprate_init=droprate_init, l0_temperature=l0_temperature,
                          l0_weight_decay=l0_weight_decay, collected=collected)

    # store for quick access
    module.__dict__.setdefault("_l0_modules", collected)


# -----------------------------------------------------------------------------
# 2. Utility to extract the *student* state‑dict from our Lightning checkpoint
# -----------------------------------------------------------------------------

def _extract_student_state_dict(raw: dict) -> dict:
    """Remove Lightning / trainer prefixes so that `model.load_state_dict` works."""
    if "state_dict" in raw:  # lightning checkpoint
        raw = raw["state_dict"]

    # Student weights are usually stored under "student_model." prefix.
    cleaned = {
        k.replace("student_model.", ""): v for k, v in raw.items() if k.startswith("student_model.")
    }
    if not cleaned:
        # Fall back to raw if no prefix matched
        cleaned = raw
    return cleaned


# -----------------------------------------------------------------------------
# 3. Public loader – returns *eval* model ready for s3prl
# -----------------------------------------------------------------------------

def load_model(
    *,
    pruned_ckpt: Path,
    pretrained_ckpt: Path,
    device: str | torch.device = "cpu",
    droprate_init: float = 0.5,
    l0_temperature: float = 2.0 / 3.0,
    l0_weight_decay: float = 1.0,
):
    """Load pre‑trained wav2vec/HuBERT, wrap with L₀ gates, then load pruned weights."""

    # --- import the model factory lazily to avoid abducting user PYTHONPATH ----
    from .wav2vec2.model import wav2vec2_model  # type: ignore

    # 1) base model ------------------------------------------------------------
    base_ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    if not {"config", "state_dict"} <= set(base_ckpt):
        raise RuntimeError(
            f"{pretrained_ckpt} must contain 'config' and 'state_dict' keys "
            f"(got {list(base_ckpt.keys())})"
        )
    model = wav2vec2_model(**base_ckpt["config"])
    model.load_state_dict(base_ckpt["state_dict"], strict=False)

    # 2) wrap ------------------------------------------------------------------
    _wrap_with_l0(
        model,
        droprate_init=droprate_init,
        l0_temperature=l0_temperature,
        l0_weight_decay=l0_weight_decay,
    )

    # 3) load pruned weights ---------------------------------------------------
    raw = torch.load(pruned_ckpt, map_location="cpu")
    student_sd = _extract_student_state_dict(raw)
    model.load_state_dict(student_sd, strict=False)

    model.eval()
    return model.to(device)