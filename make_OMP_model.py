import torch

from wav2vec2.model import (
    wav2vec2_model,
)

out_name = "trained_models/omp_pruned_w2v2_base_ls100.pth"

model = torch.load("pretrained/wav2vec2_asr-base-ls100.hf.pth")

config = model['config']
model = wav2vec2_model(**config)

not_pruned = []
for n, p in model.named_parameters():
    if n == "feature_extractor.dummy_weight":
        not_pruned.append(n)
        continue

    org_p = model_org_dict[n]
    
    