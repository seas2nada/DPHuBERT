import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import sosfilt

from .audio import change_gender, change_gender_f0, params2sos

logger = logging.getLogger("dataset")

Qmin, Qmax = 2, 5

class Nansy():
    def __init__(
        self,
        spk2info: str = None
        ):
        with open(spk2info, "rb") as fp:
            self.spk2info = pickle.load(fp)
            self.spk2info = self.spk2info["train"]

        self.spk2info_keys = list(self.spk2info.keys())
        self.rng = np.random.default_rng()
        self.Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
        self.skip_count = 0

    def get_spk_info(self, spk: str):
        _, (lo, hi, _) = self.spk2info[spk]
        if lo == 50:
            lo = 75
        if spk == "1447":
            lo, hi = 60, 400
        return lo, hi

    def random_formant_f0(self, wav, sr, spk):
        lo, hi = self.get_spk_info(spk)

        ratio_fs = self.rng.uniform(1, 1.4)
        coin = self.rng.random() > 0.5
        ratio_fs = coin * ratio_fs + (1 - coin) * (1 / ratio_fs)

        ratio_ps = self.rng.uniform(1, 2)
        coin = self.rng.random() > 0.5
        ratio_ps = coin * ratio_ps + (1 - coin) * (1 / ratio_ps)

        ratio_pr = self.rng.uniform(1, 1.5)
        coin = self.rng.random() > 0.5
        ratio_pr = coin * ratio_pr + (1 - coin) * (1 / ratio_pr)

        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)

        return ss

    def random_eq(self, wav, sr):
        z = self.rng.uniform(0, 1, size=(10,))
        Q = Qmin * (Qmax / Qmin) ** z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav

    def perturb_speaker(self, wav, sr, spk):
        # Speaker perturbation
        try:
            wav_p = self.random_formant_f0(wav, sr, spk)
        except UserWarning:
            self.skip_count += 1
            wav_p = np.copy(wav)
            logger.info(f"Praat warning - Skipping {self.skip_count}")
        except RuntimeError:
            self.skip_count += 1
            wav_p = np.copy(wav)
            logger.info(f"Praat warning - Skipping {self.skip_count}")
        
        wav_p = self.random_eq(wav_p, sr)
        wav_p = np.clip(wav_p, -1.0, 1.0)
        return wav_p