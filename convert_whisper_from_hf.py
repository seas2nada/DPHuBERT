"""Convert fairseq's wav2vec2 to our format."""

import torch

# TODO: whisper
from wav2vec2.model import wav2vec2_model
from transformers.models.whisper import WhisperForConditionalGeneration


if __name__ == "__main__":
    out_name = "pretrained/whisper-large-v3.hf.pth"

    original = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    print(original)
    # print(original.state_dict().keys())
    exit()
    
    # default config of wav2vec2 large
    wav2vec2_large_config = dict(
        extractor_mode="group_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=True,
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
        encoder_layer_norm_first=False,     # wav2vec2 large uses post norm
        encoder_layer_drop=0.05,
        aux_num_out=32,
        normalize_waveform=True,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': wav2vec2_large_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")
