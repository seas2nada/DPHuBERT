"""Convert hf's whisper to our format."""

import torch

from whisper_test.model import whisper_model
from transformers.models.whisper import WhisperForConditionalGeneration


if __name__ == "__main__":
    out_name = "pretrained/whisper-large-v3-renew.hf.pth"

    imported = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    print(imported)
    # print(original.state_dict().keys())
    
    # default config of whisper large
    whisper_large_config = dict(
        vocab_size=51866,
        num_mel_bins=128,
        encoder_layers=32,
        decoder_layers=32,
        decoder_ffn_dim=[5120] * 32,
        encoder_ffn_dim=[5120] * 32,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50258,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
        bos_token_id=50257,
        eos_token_id=50257,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50257],
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        apply_spec_augment=False,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        median_filter_width=7,
        conv_in_channels=[128, 1280],
        conv_out_channels=[1280, 1280],
        encoder_attention_heads=[20] * 32,
        decoder_attention_heads=[20] * 32,
        cross_attention_heads=[20] * 32,
        prune_conv_channels=False,
        prune_ff=False,
        prune_ff_layer=False,
        prune_heads=False,
        prune_layer=False,
    )

    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': whisper_large_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = whisper_model(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")
