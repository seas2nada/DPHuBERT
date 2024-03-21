import sys
sys.path.append(".")
import os

import torchaudio

import logging
import pathlib
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, l1_loss

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite.utilities.rank_zero import _get_rank

from lightning import (
    DistillModule,
    DistillLoss,
    CtcLoss,
)
from wav2vec2.model import (
    wav2vec2_model,
)

from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

def _init_layer_transform(module: nn.Linear):
    module.weight.data.copy_(torch.eye(len(module.weight)))
    module.bias.data.fill_(0)


def run_train(args):

    # Create teacher model
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher_model = wav2vec2_model(**teacher_ckpt['config'])

    # Load teacher model
    with torch.no_grad():
        for name, p in teacher_model.named_parameters():
            if "dummy_weight" in name:
                continue
            if name == "encoder.transformer.pos_conv_embed.conv.weight_g":
                name = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0"
            if name == "encoder.transformer.pos_conv_embed.conv.weight_v":
                name = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1"
            p.copy_(teacher_ckpt['state_dict'][name])

    # Freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()
    
    # Create student model
    student_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    pruning_units = args.pruning_units.split(",")
    student_config = student_ckpt['config']
    student_config.update(
        dict(
            extractor_prune_conv_channels = "conv" in pruning_units,
            encoder_prune_attention_heads = "head" in pruning_units,
            encoder_prune_attention_layer = "attlayer" in pruning_units,
            encoder_prune_feed_forward_intermediate = "interm" in pruning_units,
            encoder_prune_feed_forward_layer = "ffnlayer" in pruning_units,
            mask_prob = args.mask_prob,
            mask_channel_prob = args.mask_channel_prob,
        )
    )
    student_model = wav2vec2_model(**student_config)

    # Load student model
    with torch.no_grad():
        for name, p in student_model.named_parameters():
            if "dummy_weight" in name or "hard_concrete" in name:
                continue
            if name == "encoder.transformer.pos_conv_embed.conv.weight_g":
                name = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0"
            if name == "encoder.transformer.pos_conv_embed.conv.weight_v":
                name = "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1"
            p.copy_(student_ckpt['state_dict'][name])

    # Create linear layers which transform student hiddens to teacher hiddens
    distill_layer_groups = [[int(l) for l in g.split(",")] for g in args.distill_layers.split(".")]
    distill_layers = []
    for g in distill_layer_groups:
        distill_layers.extend(g)
    student_embed_dim = student_model.encoder.feature_projection.projection.out_features
    teacher_embed_dim = teacher_model.encoder.feature_projection.projection.out_features

    if args.distill_mode == "layer2layer":
        distill_linear_projs = nn.ModuleList()
        for g in distill_layer_groups:      # layers in the same group share a linear layer
            tmp_linear = nn.Linear(student_embed_dim, teacher_embed_dim)
            _init_layer_transform(tmp_linear)
            for _ in range(len(g)):
                distill_linear_projs.append(tmp_linear)
    elif args.distill_mode == "predlayer":      # same as DistilHuBERT
        # use independent linear layers, cannot be shared
        distill_linear_projs = nn.ModuleList(
            nn.Sequential(
                nn.Linear(student_embed_dim, teacher_embed_dim),
                nn.GELU(),
            ) for _ in range(len(distill_layers))
        )
    else:
        raise ValueError(f"Invalid distill mode: {args.distill_mode}")

    # Create CtcLoss module
    ctc_loss_criterion = CtcLoss(
        args.tsv_dir / "dict.ltr.txt",
    )

    # Create DistillLoss module
    distill_loss_criterion = DistillLoss(
        l2_weight=args.l2_weight,
        l1_weight=args.l1_weight,
        cos_weight=args.cos_weight,
        cos_type=args.cos_type,
    )

    distill_module = DistillModule(
        teacher_model=teacher_model,
        student_model=student_model,
        distill_mode=args.distill_mode,
        distill_layers=distill_layers,
        distill_linear_projs=distill_linear_projs,
        distill_loss=distill_loss_criterion,
        ctc_loss=ctc_loss_criterion,
        ctc_weight=args.ctc_weight,
        distill_weight=args.distill_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
        use_reg=True,
        reg_learning_rate=args.reg_learning_rate,
        target_sparsity=args.target_sparsity,
        sparsity_warmup_updates=args.sparsity_warmup_updates,
        tsv_dir=args.tsv_dir,
        label_dir=args.label_dir,
        train_subset=args.train_subset,
        seconds_per_batch=args.seconds_per_batch,
        num_workers=args.num_workers,
        param_reg_type=args.param_reg_type,
        threshold=args.threshold,
    )

    return distill_module


def _parse_args():
    parser = ArgumentParser(
        description="Joint distillation and pruning of HuBERT",
    )

    # dataset and dataloader related
    parser.add_argument(
        "--tsv_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the directory containing tsv files.",
    )
    parser.add_argument(
        "--label_dir",
        type=pathlib.Path,
        help="Path to the directory containing label files.",
        default=None
    )
    parser.add_argument(
        "--train_subset",
        default="train100",
        type=str,
        help="The subset name for training. (Default: 'train100')",
    )
    parser.add_argument(
        "--seconds_per_batch",
        default=87.5,
        type=float,
        help="Number of seconds of audio in a mini-batch. (Default: 87.5)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers in DataLoader."
    )

    # general training related
    parser.add_argument(
        "--resume_checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to the feature and label directories. (Default: None)",
    )
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--log_interval",
        default=50,
        type=int,
        help="Log interval in steps."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0002,
        type=float,
        help="The peak learning rate. (Default: 0.0002)",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay (L2 penalty) (Default: 0.0)",
    )
    parser.add_argument(
        "--warmup_updates",
        default=15000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 15000)",
    )
    parser.add_argument(
        "--max_updates",
        default=50000,
        type=int,
        help="Total number of training steps. (Default: 50000)",
    )
    parser.add_argument(
        "--clip_norm",
        default=10.0,
        type=float,
        help="The gradient norm value to clip. (Default: 10.0)",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--accum_grad",
        default=1,
        type=int,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Precision for training."
    )

    # distillation related
    parser.add_argument(
        "--teacher_ckpt",
        default=pathlib.Path("pretrained_ckpts/hubert-base-ls960.pth"),
        type=pathlib.Path,
        help="Path to the teacher model checkpoint."
    )
    parser.add_argument(
        "--student_ckpt",
        default=pathlib.Path("pretrained_ckpts/hubert-base-ls960.pth"),
        type=pathlib.Path,
        help="Path to the student model checkpoint (for initialization)."
    )
    parser.add_argument(
        "--distill_layers",
        default="0.4,8,12",
        type=str,
        help="Distill layer indices (use period to separate groups and comma to separate layers within a group)."
    )
    parser.add_argument(
        "--distill_mode",
        type=str,
        default="layer2layer",
        choices=["layer2layer", "predlayer"],
        help="Distill mode, either layer2layer or predlayer."
    )
    parser.add_argument(
        "--l2_weight",
        default=0.0,
        type=float,
        help="Weight of MSE loss."
    )
    parser.add_argument(
        "--l1_weight",
        default=1.0,
        type=float,
        help="Weight of L1 loss."
    )
    parser.add_argument(
        "--cos_weight",
        default=1.0,
        type=float,
        help="Weight of cosine similarity loss."
    )
    parser.add_argument(
        "--cos_type",
        default="raw",
        type=str,
        choices=["raw", "log_sig"],
        help="Type of the cosine similarity loss."
    )

    # pruning related
    parser.add_argument(
        "--pruning_units",
        default="conv,head,interm,attlayer,ffnlayer",
        type=str,
        help="Pruning units as a comma-separated list."
    )
    parser.add_argument(
        "--reg_learning_rate",
        default=0.02,
        type=float,
        help="Regularization learning rate."
    )
    parser.add_argument(
        "--target_sparsity",
        default=0.75,
        type=float,
        help="Target sparsity."
    )
    parser.add_argument(
        "--sparsity_warmup_updates",
        default=5000,
        type=int,
        help="Warmup updates for the target sparsity."
    )

    parser.add_argument(
        "--ctc_weight",
        default=0.001,
        type=float,
        help="Weight for ctc loss."
    )
    parser.add_argument(
        "--distill_weight",
        default=1,
        type=float,
        help="Weight for ctc loss."
    )

    parser.add_argument(
        "--mask_prob",
        default=0.75,
        type=float,
        help="Weight for ctc loss."
    )
    parser.add_argument(
        "--mask_channel_prob",
        default=0.65,
        type=float,
        help="Weight for ctc loss."
    )
    parser.add_argument(
        "--param_reg_type",
        default="ema",
        type=str,
        help="Method for parameters regularization"
    )

    # wandb logger args
    parser.add_argument(
        "--project_name",
        default="dphubert2024",
        type=str,
        help="project name"
    )
    parser.add_argument(
        "--threshold",
        default="0.2",
        type=float,
    )

    return parser.parse_args()

def calculate_metrics(teacher_hiddens, student_hiddens):
    cosine_similarities = []
    l1_distances = []

    # Ensure you're iterating over the same number of layers
    num_layers = len(teacher_hiddens)
    assert len(teacher_hiddens) == len(student_hiddens)

    for idx in range(num_layers):
        teacher_layer = teacher_hiddens[idx, :, :]  # Assuming shape [batch, layer, time, feature]
        student_layer = student_hiddens[idx, :, :]  # Adjust if your shape is different

        # Flatten the tensors to compute cosine similarity and L2 distance across all dimensions except the batch
        teacher_flat = teacher_layer.view(teacher_layer.size(1), -1)
        student_flat = student_layer.view(student_layer.size(1), -1)

        # Calculate cosine similarity for each item in the batch, then average
        cos_sim = cosine_similarity(teacher_flat, student_flat, dim=1).mean().item()
        cosine_similarities.append(cos_sim)

        # Calculate L1 distance for each item in the batch, then average
        l1_dist = l1_loss(teacher_flat, student_flat, reduction='mean').item()
        l1_distances.append(l1_dist)

    return cosine_similarities, l1_distances

import matplotlib.pyplot as plt

def plot_metrics(cosine_similarities, cosine_similarities_none, l1_distances, l1_distances_none, out_plot_path):
    layers = list(range(len(cosine_similarities)))

    # Accessible color selection
    colors = {
        'cosine_sim': '#1E88E5',  # Vivid pinkish-red, good for color blindness
        'cosine_sim_none': '#D81B60',  # Strong blue, distinguishable from pinkish-red
        'l1_dist': '#1E88E5',  # Amber, stands out against both blue and pinkish-red
        'l1_dist_none': '#D81B60'  # Deep teal, contrast with amber
    }
    
    plt.figure(figsize=(14, 6))

    # Plot Cosine Similarities Comparison
    plt.subplot(1, 2, 1)
    plt.plot(layers, cosine_similarities, marker='o', linestyle='-', color=colors['cosine_sim'], label='w/ GGPR')
    plt.plot(layers, cosine_similarities_none, marker='x', linestyle='--', color=colors['cosine_sim_none'], label='w/o GGPR')
    # plt.title('Average Cosine Similarity Comparison')
    plt.xlabel('Layer Index', fontsize=24)
    plt.ylabel('Cosine Similarity', fontsize=24)
    plt.legend(fontsize=18)
    plt.xticks(np.arange(0, 25, 4), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot L1 Distances Comparison
    plt.subplot(1, 2, 2)
    plt.plot(layers, l1_distances, marker='o', linestyle='-', color=colors['cosine_sim'], label='w/ GGPR')
    plt.plot(layers, l1_distances_none, marker='x', linestyle='--', color=colors['cosine_sim_none'], label='w/o GGPR')
    # plt.title('Average L1 Distance Comparison')
    plt.xlabel('Layer Index', fontsize=24)
    plt.ylabel('L1 Distance', fontsize=24)
    plt.legend(fontsize=18)
    plt.xticks(np.arange(0, 25, 4), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save the plot to the specified output directory
    plt.savefig(os.path.join(out_plot_path, 'comparison_metrics_average.png'))
    plt.close()

def cli_main():
    args = _parse_args()
    # distill_module = run_train(args)
    # distill_module_none = run_train(args)
    out_plot_path = args.exp_dir

    # model = torch.load("/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/w2v2large_960to960_dp_pregl2/ckpts/epoch=10-step=50000.ckpt")
    # with torch.no_grad():
    #     for name, p in distill_module.named_parameters():
    #         p.copy_(model['state_dict'][name])

    # model = torch.load("/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/exp/w2v2large_960to960_dp_nopreg/ckpts/epoch=10-step=50000.ckpt")
    # with torch.no_grad():
    #     for name, p in distill_module_none.named_parameters():
    #         p.copy_(model['state_dict'][name])

    # with open("/home/ubuntu/Workspace/DB/LibriSpeech/DPHuBERT/data/librispeech/test_other/test_other.tsv", "r") as f:
    #     lines = f.readlines()
    # parent_dir = next(iter(lines)).strip()

    # cosine_similarities_sum = torch.zeros(25)
    # l1_distances_sum = torch.zeros(25)
    # cosine_similarities_none_sum = torch.zeros(25)
    # l1_distances_none_sum = torch.zeros(25)
    # num_lines = len(lines) - 1  # Assuming 'lines' contains all the lines including the header

    # for wavepath in tqdm(lines[1:]):
    #     wavepath = wavepath.strip().split('\t')[0]
    #     wavepath = os.path.join(parent_dir, wavepath)

    #     waveforms = torchaudio.load(wavepath)[0]
    #     lengths = torch.tensor([len(waveforms[0])])
    #     st_waveforms = waveforms

    #     distill_module.teacher_model.eval()
    #     distill_module.student_model.eval()
        
    #     with torch.no_grad():
    #         teacher_hiddens, teacher_lengths = distill_module.teacher_model.extract_features(waveforms, lengths, mask=False)
    #         teacher_last_hidden = teacher_hiddens[-1]
    #         teacher_hiddens = torch.stack(
    #             [teacher_hiddens[idx] for idx in range(25)], dim=1
    #         )   # (batch, layer, time, feature)

    #         student_hiddens, student_lengths = distill_module.student_model.extract_features(st_waveforms, lengths)
    #         student_last_hidden = student_hiddens[-1]
    #         student_hiddens = torch.stack(
    #             [student_hiddens[idx] for idx in range(25)], dim=1
    #         )   # (batch, layer, time, feature)

    #     cosine_similarities, l1_distances = calculate_metrics(teacher_hiddens[0], student_hiddens[0])

    #     with torch.no_grad():
    #         student_hiddens, student_lengths = distill_module_none.student_model.extract_features(st_waveforms, lengths)
    #         student_last_hidden = student_hiddens[-1]
    #         student_hiddens = torch.stack(
    #             [student_hiddens[idx] for idx in range(25)], dim=1
    #         )   # (batch, layer, time, feature)

    #     cosine_similarities_none, l1_distances_none = calculate_metrics(teacher_hiddens[0], student_hiddens[0])

    #     # Accumulate metrics
    #     cosine_similarities_sum += torch.tensor(cosine_similarities)
    #     l1_distances_sum += torch.tensor(l1_distances)
    #     cosine_similarities_none_sum += torch.tensor(cosine_similarities_none)
    #     l1_distances_none_sum += torch.tensor(l1_distances_none)

    # # Average the metrics over all lines
    # average_cosine_similarities = cosine_similarities_sum / num_lines
    # average_l1_distances = l1_distances_sum / num_lines
    # average_cosine_similarities_none = cosine_similarities_none_sum / num_lines
    # average_l1_distances_none = l1_distances_none_sum / num_lines

    average_cosine_similarities_none = [0.6801, 0.6622, 0.6471, 0.6484, 0.6578, 0.6720, 0.6874, 0.7140, 0.7478, 0.7108, 0.7056, 0.7160, 0.7313, 0.7420, 0.7491, 0.7596, 0.7655, 0.7307, 0.7100, 0.6976, 0.6528, 0.5692, 0.4637, 0.4368, 0.5946]
    average_cosine_similarities = [0.9481, 0.9065, 0.8906, 0.8960, 0.8975, 0.9067, 0.9159, 0.9249, 0.9350, 0.9300, 0.9281, 0.9324, 0.9383, 0.9417, 0.9405, 0.9416, 0.9286, 0.9081, 0.9040, 0.8896, 0.8596, 0.8021, 0.6760, 0.6142, 0.6849]
    average_l1_distances_none = [0.2057, 0.2530, 0.2656, 0.2652, 0.2627, 0.2539, 0.2427, 0.2304, 0.2122, 0.2342, 0.2340, 0.2246, 0.2131, 0.2048, 0.2055, 0.2068, 0.2041, 0.2262, 0.2584, 0.2865, 0.3220, 0.3859, 0.4293, 0.4296, 0.3165]
    average_l1_distances = [0.1083, 0.1496, 0.1680, 0.1651, 0.1665, 0.1516, 0.1377, 0.1277, 0.1210, 0.1220, 0.1219, 0.1142, 0.1046, 0.0983, 0.0989, 0.1019, 0.1158, 0.1255, 0.1514, 0.1732, 0.2087, 0.2688, 0.3331, 0.3483, 0.2835]

    # Plot the metrics
    plot_metrics(average_cosine_similarities, average_cosine_similarities_none, average_l1_distances, average_l1_distances_none, out_plot_path)

    # with open("average_cosine_similarities_none.txt", "w") as f:
    #     f.write(str(average_cosine_similarities_none))
    # with open("average_l1_distances_none.txt", "w") as f:
    #     f.write(str(average_l1_distances_none))
    # with open("average_cosine_similarities.txt", "w") as f:
    #     f.write(str(average_cosine_similarities))
    # with open("average_l1_distances.txt", "w") as f:
    #     f.write(str(average_l1_distances))

if __name__ == "__main__":
    cli_main()