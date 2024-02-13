"""Convert fairseq's wav2vec2 to our format."""

import torch
import fairseq
from torchaudio.models.wav2vec2.utils import import_fairseq_model

from wav2vec2.model import wav2vec2_model
from transformers.models.wav2vec2 import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model


if __name__ == "__main__":

    out_name = "pretrained/wav2vec2_asr-large-ls960.fairseq.pth"

    fairseq_ckpt = "pretrained/fairseq/wav2vec_big_960h.pt"
    ensemble, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_ckpt])
    original = ensemble[0]
    imported = import_fairseq_model(original)
    print(imported)
    
    # default config of wav2vec2 large
    wav2vec2_large_config = dict(
        extractor_mode="group_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_use_attention=[True] * 24,
        encoder_use_feed_forward=[True] * 24,
        encoder_num_heads=[16] * 24,
        encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[4096] * 24,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=True,     # wav2vec2 large uses pre norm
        encoder_layer_drop=0.05,
        aux_num_out=32,
        normalize_waveform=True,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    # # Create a mapping between parameter names in loaded model and custom model
    # parameter_mapping = {
    #     'encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0': 'encoder.transformer.pos_conv_embed.conv.weight_g',
    #     'encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1': 'encoder.transformer.pos_conv_embed.conv.weight_v',
    #     # Add more mappings as needed
    # }

    # # Copy weights from loaded model to custom model using the mapping
    # state_dict = imported.state_dict()
    # for source_param, target_param in parameter_mapping.items():
    #     state_dict[target_param] = state_dict.pop(source_param)

    torch.save(
        {
            'state_dict': state_dict,
            'config': wav2vec2_large_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")