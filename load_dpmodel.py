import torch
from wav2vec2.model import wav2vec2_model

# ckpt_path = "exp/hubert-base_train960_sp0.75_spup5000_lr0.0002_up15000_max50000_layer2layer0.4,8,12_reglr0.02_conv,head,interm/ckpts/pruned_hubert_base.pth"
ckpt_path = "pretrained/hubert-base-ls960.hf.pth"
ckpt = torch.load(ckpt_path)
model = wav2vec2_model(**ckpt["config"])
result = model.load_state_dict(ckpt["state_dict"], strict=False)
print(f"missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
print(f"{sum(p.numel() for p in model.parameters())} params")