import sys
sys.path.append(".")
import os

import torch

from wav2vec2.model import (
    wav2vec2_model,
)

sp = sys.argv[1]

out_name = f"exp/OMP/wav2vec2_asr-large-ls960-OMP-sp{str(sp)}/wav2vec2_asr-large-ls960-OMP-sp{str(sp)}.hf.pth"
if not os.path.exists("/".join(out_name.split("/")[:-1])):
    os.makedirs("/".join(out_name.split("/")[:-1]), exist_ok=True)

sp = float(sp)

model = torch.load("pretrained/wav2vec2_asr-large-ls960.hf.pth")

state_dict = model['state_dict']
# Flatten all parameters and their absolute values
all_params = torch.cat([param.abs().flatten() for param in state_dict.values()])

# Find the threshold for the lowest 10% of absolute values
threshold = torch.topk(all_params, int(sp * len(all_params)), largest=False).values[-1]

# Iterate through the model parameters and prune the values below the threshold
for key, param in state_dict.items():
    mask = param.abs() >= threshold
    state_dict[key] = torch.where(mask, param, torch.tensor(0.0))
model['state_dict'] = state_dict

# Check model is properly set
models = wav2vec2_model(**model['config'])
with torch.no_grad():
    for name, p in models.named_parameters():
        if "dummy_weight" in name:
            continue
        if name == "encoder.transformer.pos_conv_embed.conv.weight_g" and name not in model['state_dict'].keys():
            name = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0"
        if name == "encoder.transformer.pos_conv_embed.conv.weight_v" and name not in model['state_dict'].keys():
            name = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1"
        p.copy_(model['state_dict'][name])
print("load model okay")

# Save the pruned model
torch.save(model, out_name)

# Count the total number of parameters and the number of zero parameters
all_params_flat = torch.cat([param.flatten() for param in state_dict.values()])
total_params = all_params_flat.numel()
zero_params = torch.sum(all_params_flat == 0)

# Calculate the sparsity value
sparsity = zero_params.item() / total_params

# Print the sparsity value
print(f"Sparsity: {sparsity:.4f}")