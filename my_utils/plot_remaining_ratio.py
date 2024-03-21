import torch
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'model' is your dictionary containing the state_dict
exp_dir = "/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/whisper_trnone_8gpu_max10000_test"
model = torch.load(exp_dir + '/ckpts/pruned_hubert_base.pth')

# Calculate remaining ratio of attention, self-attention, and FFN dimensions for each layer
attention_ratios = [model['state_dict'][f'model.decoder.layers.{i}.encoder_attn.k_proj.weight'].size()[0] * 100 / 1280 for i in range(32)]
self_attention_ratios = [model['state_dict'][f'model.decoder.layers.{i}.self_attn.k_proj.weight'].size()[0] * 100 / 1280 for i in range(32)]
ffn_ratios = [model['state_dict'][f'model.decoder.layers.{i}.fc1.weight'].size()[0] * 100 / 5120 for i in range(32)]

# Create x-axis values (layer indices)
layers = np.arange(32)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Attention dimension ratios with hatch
plt.bar(layers - 0.2, attention_ratios, width=0.2, label='Attention', color='lightsalmon', hatch='/', edgecolor='black')

# Plot Self-Attention dimension ratios with hatch
plt.bar(layers, self_attention_ratios, width=0.2, label='Self-Attention', color='lightsteelblue', edgecolor='black')

# Plot FFN dimension ratios with hatch
plt.bar(layers + 0.2, ffn_ratios, width=0.2, label='FFN', color='lightcoral', hatch='-', edgecolor='black')

# Set labels and title
plt.xlabel('Layer Index')
plt.ylabel('Remaining Ratio (%)')

# Add legend with hatch patterns
legend_labels = ['Attention', 'Self-Attention', 'FFN']
plt.legend(labels=legend_labels, loc='upper right', frameon=True, edgecolor='black')

# Show the plot
plt.savefig(exp_dir + '/decoder_remaining_ratio.png')
plt.close()


# Calculate remaining ratio of attention and FFN dimensions for each layer
attention_ratios = [model['state_dict'][f'model.encoder.layers.{i}.self_attn.k_proj.weight'].size()[0] * 100 / 1280 for i in range(32)]
ffn_ratios = [model['state_dict'][f'model.encoder.layers.{i}.fc1.weight'].size()[0] * 100 / 5120 for i in range(32)]

# Create x-axis values (layer indices)
layers = np.arange(32)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Attention dimension ratios
plt.bar(layers - 0.2, attention_ratios, width=0.4, label='Self-attention', color='lightsteelblue', edgecolor='black')

# Plot FFN dimension ratios
plt.bar(layers + 0.2, ffn_ratios, width=0.4, label='FFN', color='lightcoral', hatch='-', edgecolor='black')

# Set labels and title
plt.xlabel('Layer Index')
plt.ylabel('Remaining Ratio (%)')
plt.legend()

# Show the plot
plt.savefig(exp_dir + '/encoder_remaining_ratio.png')
plt.close()