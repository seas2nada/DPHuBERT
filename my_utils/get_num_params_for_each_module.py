import sys
sys.path.append(".")
import os

import torch

from whisper_test.model import (
    whisper_model,
)

model = torch.load("pretrained/whisper-large-v3.hf.pth")

config = model['config']

model = whisper_model(**config)

num_dict = {}
for n, p in model.named_parameters():
    num_params = p.numel()
    num_dict[n] = num_params

with open("whisper_num_params.txt", "w") as f:
    for k, v in num_dict.items():
        f.write(str(k) + ": " + str(v) + '\n')

    