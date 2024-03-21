import json
import pathlib
import torch
from argparse import ArgumentParser

from whisper_test.model import (
    whisper_model,
)


def prune_from_ckpt(distilled_ckpt, original_ckpt):
    ckpt = torch.load(distilled_ckpt, map_location='cpu')
    student_model_state_dict = {
        k[len("student_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")
    }
    distill_linear_projs_state_dict = {
        k[len("distill_linear_projs."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("distill_linear_projs.")
    }
    config = torch.load(original_ckpt, map_location='cpu')['config']
    config.update(
        dict(
            prune_conv_channels = False,
            prune_ff = True,
            prune_ff_layer = False,
            prune_heads = True,
            prune_layer = False,
        )
    )
    model = whisper_model(**config)
    model.load_state_dict(student_model_state_dict, strict=True)

    # encoder_attention_heads: [20]*32

    conv_config, encoder_ffn_dim, encoder_attention_heads, decoder_ffn_dim, decoder_attention_heads, cross_attention_heads = model.prune()
    pruned_config = config.copy()
    pruned_config.update(
        {
            'conv_in_channels': conv_config['conv_in_channels'],
            'conv_out_channels': conv_config['conv_out_channels'],
            "encoder_ffn_dim": encoder_ffn_dim,
            "encoder_attention_heads": encoder_attention_heads,
            "decoder_ffn_dim": decoder_ffn_dim,
            "decoder_attention_heads": decoder_attention_heads,
            "cross_attention_heads": cross_attention_heads,
        }
    )
    pruned_config.update(
        {
            "prune_conv_channels": False,
            "prune_ff": False,
            "prune_ff_layer": False,
            "prune_heads": False,
            "prune_layer": False,
        }
    )
    print(json.dumps(pruned_config, indent=4))

    ret = {
        "state_dict": model.state_dict(),
        "config": pruned_config,
        "distill_linear_projs": distill_linear_projs_state_dict,
    }
    return ret


def load_pruned_model(ckpt_path, org_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = whisper_model(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"], strict=True)

    org_ckpt = torch.load(org_path, map_location='cpu')
    org_model = whisper_model(**org_ckpt["config"])
    num_org_model_params = sum(p.numel() for p in org_model.parameters())
    num_pruned_model_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in original model: {num_org_model_params}")
    print(f"Number of parameters in pruned model: {num_pruned_model_params}")
    print(f"Sparsity: {100 - num_pruned_model_params / num_org_model_params * 100:.2f}%")
    return model


def parse_args():
    parser = ArgumentParser(description="Prune and save distilled model.")
    parser.add_argument(
        "--distilled_ckpt",
        type=pathlib.Path,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--original_ckpt",
        type=pathlib.Path,
        help="Path to the original checkpoint."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    out_path = args.distilled_ckpt.parent / "pruned_hubert_base.pth"
    torch.save(
        prune_from_ckpt(
            distilled_ckpt=args.distilled_ckpt,
            original_ckpt=args.original_ckpt
        ),
        out_path
    )

    # Check if loading from ckpt works
    load_pruned_model(out_path, args.original_ckpt)

    print(f"Successfully saved pruned model weights and config to: {out_path}")
