"""Convert hf's wav2vec2 to our format."""

import torch
import fairseq
from torchaudio.models.wav2vec2.utils import import_fairseq_model

from wav2vec2.model import wav2vec2_model

from transformers import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model


if __name__ == "__main__":
    out_name = "pretrained/wav2vec2_asr-base-ls100.hf.pth"

    original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h")
    imported = import_huggingface_model(original)

    # default config of wav2vec2 base
    wav2vec2_base_config = dict(
        extractor_mode="group_norm",    # hubert/w2v2 base only uses a group norm at the first conv layer
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_num_heads=[12] * 12,
        encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,     # hubert/w2v2 base uses post norm
        encoder_layer_drop=0.05,
        aux_num_out=32,
        normalize_waveform=False,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': wav2vec2_base_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")
