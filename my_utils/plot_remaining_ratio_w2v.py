import torch
import matplotlib.pyplot as plt
import numpy as np

plot_omp = False
sp = 0.7

# Assuming 'model' is your dictionary containing the state_dict
exp_dir = f"/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/wav2vec2-large_train100_sp{sp}_pregnone_thre0.2"
model = torch.load(exp_dir + '/ckpts/pruned_hubert_base.pth')
reg_dir = f"/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/wav2vec2-large_train100_sp{sp}_pregl2_thre0.5"
reg_model = torch.load(reg_dir + '/ckpts/pruned_hubert_base.pth')
if plot_omp:
    omp_model = torch.load(f'/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/OMP/wav2vec2_asr-large-ls960-OMP-sp{sp}/wav2vec2_asr-large-ls960-OMP-sp{sp}.hf.pth')

# # Calculate remaining ratio of attention, self-attention, and FFN dimensions for each layer
# cnn_ratios = [model['state_dict']['feature_extractor.conv_layers.0.conv.weight'].size()[0] * 100 / (512)]
# cnn_ratios += [model['state_dict'][f'feature_extractor.conv_layers.{i}.conv.weight'].size()[0] * model['state_dict'][f'feature_extractor.conv_layers.{i}.conv.weight'].size()[1] * 100 / (512*512) for i in range(1, 7)]
# reg_cnn_ratios = [reg_model['state_dict']['feature_extractor.conv_layers.0.conv.weight'].size()[0] * 100 / (512)]
# reg_cnn_ratios += [reg_model['state_dict'][f'feature_extractor.conv_layers.{i}.conv.weight'].size()[0] * reg_model['state_dict'][f'feature_extractor.conv_layers.{i}.conv.weight'].size()[1] * 100 / (512*512) for i in range(1, 7)]
# if plot_omp:
#     omp_cnn_ratios = []
#     for i in range(7):
#         conv_weight = omp_model['state_dict'][f'feature_extractor.conv_layers.{i}.conv.weight']
#         num_conv_weight = conv_weight.size()[0] * conv_weight.size()[1] * conv_weight.size()[2]
#         num_zero_elements = torch.sum(conv_weight == 0).item()
#         omp_cnn_ratios.append((num_conv_weight - num_zero_elements) * 100 / num_conv_weight)

attention_ratios = [model['state_dict'][f'encoder.transformer.layers.{i}.attention.k_proj.weight'].size()[0] * 100 / 1024 for i in range(24)]
reg_attention_ratios = [reg_model['state_dict'][f'encoder.transformer.layers.{i}.attention.k_proj.weight'].size()[0] * 100 / 1024 for i in range(24)]
if plot_omp:
    omp_attention_ratios = []
    for i in range(24):
        k_proj_weight = omp_model['state_dict'][f'encoder.transformer.layers.{i}.attention.k_proj.weight']
        num_k_proj_weight = k_proj_weight.size()[0] * k_proj_weight.size()[1]
        num_zero_elements = torch.sum(k_proj_weight == 0).item()

        v_proj_weight = omp_model['state_dict'][f'encoder.transformer.layers.{i}.attention.v_proj.weight']
        num_v_proj_weight = v_proj_weight.size()[0] * v_proj_weight.size()[1]
        num_zero_elements += torch.sum(v_proj_weight == 0).item()

        q_proj_weight = omp_model['state_dict'][f'encoder.transformer.layers.{i}.attention.q_proj.weight']
        num_q_proj_weight = q_proj_weight.size()[0] * q_proj_weight.size()[1]
        num_zero_elements += torch.sum(q_proj_weight == 0).item()

        omp_attention_ratios.append((num_k_proj_weight + num_v_proj_weight + num_q_proj_weight - num_zero_elements) * 100 / (num_k_proj_weight + num_v_proj_weight + num_q_proj_weight))

ffn_ratios = [model['state_dict'][f'encoder.transformer.layers.{i}.feed_forward.intermediate_dense.weight'].size()[0] * 100 / 4096 for i in range(24)]
reg_ffn_ratios = [reg_model['state_dict'][f'encoder.transformer.layers.{i}.feed_forward.intermediate_dense.weight'].size()[0] * 100 / 4096 for i in range(24)]
if plot_omp:
    omp_ffn_ratios = []
    for i in range(24):
        intermediate_dense_weight = omp_model['state_dict'][f'encoder.transformer.layers.{i}.feed_forward.intermediate_dense.weight']
        num_intermediate_dense_weight = intermediate_dense_weight.size()[0] * intermediate_dense_weight.size()[1]
        num_zero_elements = torch.sum(intermediate_dense_weight == 0).item()

        output_dense_weight = omp_model['state_dict'][f'encoder.transformer.layers.{i}.feed_forward.output_dense.weight']
        num_output_dense_weight = output_dense_weight.size()[0] * output_dense_weight.size()[1]
        num_zero_elements += torch.sum(output_dense_weight == 0).item()

        omp_ffn_ratios.append((num_intermediate_dense_weight + num_output_dense_weight - num_zero_elements) * 100 / (num_intermediate_dense_weight + num_output_dense_weight))

# Create x-axis values (layer indices)
cnn_layers = np.arange(1, 8)  # Adjust starting point to 1
transformer_layers = np.arange(1, 25)

# Plotting Subplots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

if plot_omp:
    bar_width = 0.2  # Adjust the width of the bars
    bar_range = 0.2
    omp_bar = -0.2
    org_bar = 0
    pruned_bar = 0.2
else:
    bar_width = 0.25  # Adjust the width of the bars
    org_bar = -bar_width / 2
    pruned_bar = bar_width / 2

# # Plot CNN dimension ratios
# if plot_omp:
#     axs.bar(cnn_layers + omp_bar, omp_cnn_ratios, width=bar_width, label='w/ OMP', color='lightgreen', hatch='\\', edgecolor='black')
# axs.bar(cnn_layers + org_bar, cnn_ratios, width=bar_width, label='w/o GGPR', color='lightsteelblue', hatch='/', edgecolor='black')
# axs.bar(cnn_layers + pruned_bar, reg_cnn_ratios, width=bar_width, label='w/ GGPR', color='lightcoral', hatch='-', edgecolor='black')
# axs.set_ylabel('Remaining Ratio (%)')
# axs.set_xlabel('CNN Layer Index')
# axs.legend()
# axs.set_xticks(cnn_layers)
# axs.set_xticklabels(cnn_layers)

exp_dir = "exp/plot_pngs"

# # Save the plot
# plt.tight_layout()  # Adjust layout for better spacing
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
# plt.savefig(exp_dir + '/CNN_remaining_ratio_subplots.png')
# plt.close()

# Plotting Subplots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

# Plot FFN dimension ratios
if plot_omp:
    axs.bar(transformer_layers + omp_bar, omp_ffn_ratios, width=bar_width, label='w/ OMP', color='lightgreen', hatch='\\', edgecolor='black')
axs.bar(transformer_layers + org_bar, ffn_ratios, width=bar_width, label='w/o GGPR', color='lightsteelblue', hatch='/', edgecolor='black')
axs.bar(transformer_layers + pruned_bar, reg_ffn_ratios, width=bar_width, label='w/ GGPR', color='lightcoral', hatch='-', edgecolor='black')
axs.set_ylabel('Remaining Ratio (%)')
axs.set_xlabel('FFN Layer Index')
axs.legend()
axs.set_xticks(transformer_layers)
axs.set_xticklabels(transformer_layers)
axs.set_ylim(0, 100)

# Save the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.savefig(exp_dir + '/FFN_remaining_ratio_subplots.png')
plt.close()

# Plotting Subplots
fig, axs = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

# Plot Attention dimension ratios
if plot_omp:
    axs.bar(transformer_layers + omp_bar, omp_attention_ratios, width=bar_width, label='w/ OMP', color='lightgreen', hatch='\\', edgecolor='black')
axs.bar(transformer_layers + org_bar, attention_ratios, width=bar_width, label='w/o GGPR', color='lightsteelblue', hatch='/', edgecolor='black')
axs.bar(transformer_layers + pruned_bar, reg_attention_ratios, width=bar_width, label='w/ GGPR', color='lightcoral', hatch='-', edgecolor='black')
axs.set_xlabel('Attention Layer Index')
axs.set_ylabel('Remaining Ratio (%)')
axs.legend()
axs.set_xticks(transformer_layers)
axs.set_xticklabels(transformer_layers)
axs.set_ylim(0, 100)

# Save the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.savefig(exp_dir + '/ATT_remaining_ratio_subplots.png')
plt.close()
