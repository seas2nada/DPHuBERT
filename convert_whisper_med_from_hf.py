"""Convert hf's whisper to our format."""

import torch

from whisper_test.model import whisper_model
from transformers.models.whisper import WhisperForConditionalGeneration
from transformers import AutoConfig

if __name__ == "__main__":
    out_name = "pretrained/whisper-medium.hf.pth"

    config = AutoConfig.from_pretrained("openai/whisper-medium")

    imported = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    print(imported)
    # print(original.state_dict().keys())
    
    # default config of whisper large
    whisper_large_config = dict(
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=24,
        decoder_layers=24,
        decoder_ffn_dim=[4096] * 24,
        encoder_ffn_dim=[4096] * 24,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50258,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
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
        suppress_tokens=[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362],
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
        conv_in_channels=[128, 1024],
        conv_out_channels=[1024, 1024],
        encoder_attention_heads=[16] * 24,
        decoder_attention_heads=[16] * 24,
        cross_attention_heads=[16] * 24,
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
