# expert.py
"""s3prl **UpstreamExpert** that serves a Hard-Concrete-pruned student HuBERT.

Drop this file under an `upstream/` folder of your s3prl checkout and register
it in `upstream/__init__.py` so the tag (e.g. "l0_hubert_student") becomes
available to `run_downstream.py`.

The expert wraps the *inference-time* student network only – no KD loss here.
Weights come from the lightning checkpoint produced by the distillation script
and are mapped via :pyfunc:`model.load_model`.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# ─────────────────────────────────────────────────────────────────────────────
# local imports
# ─────────────────────────────────────────────────────────────────────────────
from .model import load_model


# --------------------------------------------------------------------------- #
# UpstreamExpert
# --------------------------------------------------------------------------- #
class UpstreamExpert(nn.Module):
    """s3prl upstream wrapper for the L₀-pruned student HuBERT."""

    def __init__(
        self,
        ckpt: str | Path,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        ckpt
            Lightning checkpoint produced by the KD+pruning trainer
            (argument ``-k`` in `run_downstream.py`).

        model_config
            YAML config used by the trainer (argument ``-g``).

        kwargs
            Ignored – kept for s3prl compatibility.
        """
        super().__init__()

        model_dict = torch.load(ckpt)

        # ------------------------------------------------------------------ 1) load model
        self.model = load_model(
            pruned_ckpt=ckpt,
            pretrained_ckpt="/home/asml02/Workspace/DPHuBERT/pretrained/hubert-base-ls960.fairseq.pth",
            device="cpu",  # s3prl moves to the right device later
        )

    # --------------------------------------------------------------------- s3prl helpers
    def get_downsample_rates(self, key: str) -> int:
        # HuBERT / wav2vec-style extractors downsample raw audio by 320
        return 320

    # --------------------------------------------------------------------- forward
    @torch.inference_mode()
    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        wavs
            List of 1-D tensors (float32) – raw waveform at 16 kHz

        Returns
        -------
        dict
            * ``"last_hidden_state"`` – B × T' × D tensor
            * ``"hidden_states"``      – list[ B × T' × D ] for each Transformer layer
        """
        # Pad to the longest waveform in batch
        wav_lens = torch.LongTensor([len(w) for w in wavs])
        waveforms = pad_sequence(wavs, batch_first=True)

        # ── Call student ‑› returns (x, lengths) and optionally hidden states
        out, out_lens = self.model(waveforms, None)

        # If the model exposes internal layer reps via attribute, fetch them;
        # otherwise return only the last hidden state.
        hidden_states = None
        if hasattr(self.model, "layer_results"):
            hidden_states = [lr.transpose(0, 1) for lr in self.model.layer_results]

        result = {
            "last_hidden_state": out,  # B × T' × D
        }
        if hidden_states is not None:
            result["hidden_states"] = hidden_states
        return result
