# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" From https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py"""

import logging
import os
import sys
import warnings
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

import transformers
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutput,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

from whisper.components import WhisperModel

from dataset.whisper_decoder_utils import shift_tokens_right

class WhisperPruneConfig(WhisperConfig):
    def __init__(
        self,
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=4,
        decoder_layers=4,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50257,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=384,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],
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
    ):

        super().__init__(
            vocab_size = vocab_size,
            num_mel_bins = num_mel_bins,
            encoder_layers = encoder_layers,
            encoder_attention_heads = encoder_attention_heads,
            decoder_attention_heads = decoder_attention_heads,
            decoder_layers = decoder_layers,
            decoder_ffn_dim = decoder_ffn_dim,
            encoder_ffn_dim = encoder_ffn_dim,
            encoder_layerdrop = encoder_layerdrop,
            decoder_layerdrop = decoder_layerdrop,
            decoder_start_token_id = decoder_start_token_id,
            use_cache = use_cache,
            is_encoder_decoder = is_encoder_decoder,
            activation_function = activation_function,
            d_model = d_model,
            dropout = dropout,
            attention_dropout = attention_dropout,
            activation_dropout = activation_dropout,
            init_std = init_std,
            scale_embedding = scale_embedding,
            max_source_positions = max_source_positions,
            max_target_positions = max_target_positions,
            pad_token_id = pad_token_id,
            bos_token_id = bos_token_id,
            eos_token_id = eos_token_id,
            suppress_tokens = suppress_tokens,
            begin_suppress_tokens = begin_suppress_tokens,
            use_weighted_layer_sum = use_weighted_layer_sum,
            classifier_proj_size = classifier_proj_size,
            apply_spec_augment = apply_spec_augment,
            mask_time_prob = mask_time_prob,
            mask_time_length = mask_time_length,
            mask_time_min_masks = mask_time_min_masks,
            mask_feature_prob = mask_feature_prob,
            mask_feature_length = mask_feature_length,
            mask_feature_min_masks = mask_feature_min_masks,
            median_filter_width = median_filter_width,
        )

        self.prune_conv_channels = prune_conv_channels
        self.prune_ff = prune_ff
        self.prune_ff_layer = prune_ff_layer
        self.prune_heads = prune_heads
        self.prune_layer = prune_layer
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.cross_attention_heads = cross_attention_heads
    

class WhisperForConditionalGeneration(WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def prune(self):
        self.eval()     # must be in eval mode
        return self.model.prune()
    
    def get_num_params(self):
        return self.model.get_num_params()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids: Optional[torch.Tensor] = None,
        num_segment_frames: Optional[int] = None,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ):
        # generation input: dict_keys(['input_features', 'labels'])

        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)

        return_dict_in_generate = False

        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        if num_segment_frames is None:
            num_segment_frames = input_stride * self.config.max_source_positions

        # 1. Check whether we're in shortform or longform mode
        if input_features is not None:
            total_input_frames = input_features.shape[-1]
        elif "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            total_input_frames = encoder_outputs_shape[1] * input_stride
        else:
            raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

        is_shortform = total_input_frames <= num_segment_frames

        # 2. Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        if return_timestamps is True:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )
            generation_config.return_timestamps = return_timestamps
        elif not is_shortform:
            if return_timestamps is False:
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
                )

            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the generation config to have `no_timestamps_token_id` correctly. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                    "or make sure to pass no more than 3000 mel input features."
                )

            logger.info("Setting `return_timestamps=True` for long-form generation.")
            generation_config.return_timestamps = True
        else:
            generation_config.return_timestamps = False

        # 3. Make sure to correctly set language-related parameters
        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language
        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

        # 4. Add forced decoder ids depending on passed `language`, `task`,`prompt_ids`, `return_token_timestamps` and `return_timestamps`
        forced_decoder_ids = None
        # Legacy code for backward compatibility
        if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get("forced_decoder_ids", None)

        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{generation_config.language}|>"
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(
                        f"Unsupported language: {generation_config.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                    )
                if language_token not in generation_config.lang_to_id:
                    raise ValueError(
                        f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                        "(You should just add it to the generation config)"
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            elif hasattr(generation_config, "task_to_id"):
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids

        if prompt_ids is not None:
            if kwargs.get("decoder_start_token_id") is not None:
                raise ValueError(
                    "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
                )
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
            # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
            text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1 :]
            # Set the decoder_start_token_id to <|startofprev|>
            kwargs.update({"decoder_start_token_id": decoder_start_token_id})

            # If the user passes `max_new_tokens`, increase its number to account for the prompt
            if kwargs.get("max_new_tokens", None) is not None:
                kwargs["max_new_tokens"] += len(text_prompt_ids)
                if kwargs["max_new_tokens"] >= self.config.max_target_positions:
                    raise ValueError(
                        f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
                        f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
                        f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
                        f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                        "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                        f"so that their combined length is less that {self.config.max_target_positions}."
                    )

            # Reformat the forced_decoder_ids to incorporate the prompt
            non_prompt_forced_decoder_ids = (
                kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
            )
            forced_decoder_ids = [
                *text_prompt_ids,
                generation_config.decoder_start_token_id,
                *[token for _rank, token in non_prompt_forced_decoder_ids],
            ]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids

        if return_token_timestamps:
            kwargs["output_attentions"] = True
            return_dict_in_generate = True
            kwargs["output_scores"] = True

            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            if kwargs.get("num_frames") is not None:
                generation_config.num_frames = kwargs.pop("num_frames")

        if generation_config.return_timestamps is True:
            last_forced_decoder_ids = (
                generation_config.forced_decoder_ids[-1][-1]
                if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids
                else None
            )
            if last_forced_decoder_ids == self.generation_config.no_timestamps_token_id:
                # remove no_timestamp to be forcefully generated if we want to return timestamps
                # this is also important to make sure `WhisperTimeStampLogitsProcessor` functions correctly
                forced_decoder_ids = generation_config.forced_decoder_ids[:-1]
                # Make sure that if list is empty we set it to None
                generation_config.forced_decoder_ids = None if len(forced_decoder_ids) == 0 else forced_decoder_ids

            timestamp_processor = [WhisperTimeStampLogitsProcessor(generation_config)]
            logits_processor = (
                timestamp_processor if logits_processor is None else timestamp_processor + logits_processor
            )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            outputs = super().generate(
                input_features,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex. We need to chunk the audio input depending on when the model generated
        # timestamp tokens
        # 6.1 Set running parameters for while loop
        if not return_segments and return_dict_in_generate:
            raise ValueError(
                "Make sure to set `return_segments=True` to return generation outputs as part of the `'segments' key.`"
            )

        # if input is longer than 30 seconds we default to long-form generation
        timestamp_begin = self.generation_config.no_timestamps_token_id + 1
        # input stride is mel frames per encoder output vector which is the product of all conv strides
        batch_size = input_features.shape[0]

        if batch_size > 1 and attention_mask is None:
            raise ValueError(
                "When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        elif batch_size > 1:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((1,), dtype=torch.long)

        current_segments = [[] for _ in range(batch_size)]
        cur_to_prev_index_map = list(range(batch_size))

        # batch size can decrease during the run
        cur_bsz = prev_bsz = batch_size

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            prev_bsz = cur_bsz

            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            new_cur_to_prev_index_map = []
            for i in range(prev_bsz):
                prev_i = cur_to_prev_index_map[i]
                if seek[prev_i] >= max_frames[prev_i]:
                    cut_index = i + (cur_bsz - prev_bsz)
                    cur_bsz -= 1
                    input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
                else:
                    # cut out index that goes away
                    new_cur_to_prev_index_map.append(prev_i)

            # 6.4  Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            cur_to_prev_index_map = new_cur_to_prev_index_map
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.5 Make sure that all inputs are padded to the same input length
            segment_input = []
            for i in range(cur_bsz):
                prev_i = cur_to_prev_index_map[i]
                segment_input_slice = input_features[
                    i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]
                ]

                if segment_input_slice.shape[-1] < num_segment_frames:
                    # pad to 3000 if necessary
                    segment_input_slice = F.pad(
                        segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                    )

                segment_input.append(segment_input_slice)

            segment_input = torch.cat(segment_input, dim=0)

            # 6.6 Batch generate current chunk
            seek_outputs = super().generate(
                segment_input,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            if return_dict_in_generate:
                seek_sequences = seek_outputs["sequences"]
                seek_outputs = [
                    {k: v[i] for k, v in seek_outputs.items()}
                    for i in range(next(iter(seek_outputs.values())).size(0))
                ]
            else:
                seek_sequences = seek_outputs

            # 6.7 Loop over each decoded audio individually as each decoding can be of a different length
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = cur_to_prev_index_map[i]

                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]
                if is_not_final and seek_sequence[-1] == self.generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]

                # remove all padding tokens
                if seek_sequence[-1] == self.generation_config.pad_token_id:
                    num_paddings = (seek_sequence == self.generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    cur_bsz=cur_bsz,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        sequences = []
        max_total_length = 0
        for current_segment_list in current_segments:
            sequences.append(torch.cat([d["tokens"] for d in current_segment_list], dim=-1))
            max_total_length = max(max_total_length, len(sequences[-1]))

        for i in range(batch_size):
            sequences[i] = F.pad(
                sequences[i], pad=(0, max_total_length - len(sequences[i])), value=self.generation_config.pad_token_id
            )

        sequences = torch.stack(sequences, dim=0)

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": current_segments}

        return sequences

    @staticmethod
    def _retrieve_segment(
        seek_sequence,
        seek_outputs,
        time_offset,
        timestamp_begin,
        seek_num_frames,
        cur_bsz,
        time_precision,
        input_stride,
        prev_idx,
        idx,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == cur_bsz * [[False, True]]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice + 1 : current_slice + 1]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            last_timestamp_pos = seek_num_frames[prev_idx]
            if timestamps.numel() > 0 and timestamps[-1].item() != timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin

            segments = [
                {
                    "start": time_offset[prev_idx],
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "result": seek_outputs[idx],
                }
            ]
            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class WhisperDecoderWrapper(WhisperPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False
        self.decoder = WhisperDecoder(config)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

def whisper_model(
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
) -> WhisperForConditionalGeneration:
    
    config = WhisperPruneConfig(
        vocab_size = vocab_size,
        num_mel_bins = num_mel_bins,
        encoder_layers = encoder_layers,
        decoder_layers = decoder_layers,
        decoder_ffn_dim = decoder_ffn_dim,
        encoder_ffn_dim = encoder_ffn_dim,
        encoder_layerdrop = encoder_layerdrop,
        decoder_layerdrop = decoder_layerdrop,
        decoder_start_token_id = decoder_start_token_id,
        use_cache = use_cache,
        is_encoder_decoder = is_encoder_decoder,
        activation_function = activation_function,
        d_model = d_model,
        dropout = dropout,
        attention_dropout = attention_dropout,
        activation_dropout = activation_dropout,
        init_std = init_std,
        scale_embedding = scale_embedding,
        max_source_positions = max_source_positions,
        max_target_positions = max_target_positions,
        pad_token_id = pad_token_id,
        bos_token_id = bos_token_id,
        eos_token_id = eos_token_id,
        suppress_tokens = suppress_tokens,
        begin_suppress_tokens = begin_suppress_tokens,
        use_weighted_layer_sum = use_weighted_layer_sum,
        classifier_proj_size = classifier_proj_size,
        apply_spec_augment = apply_spec_augment,
        mask_time_prob = mask_time_prob,
        mask_time_length = mask_time_length,
        mask_time_min_masks = mask_time_min_masks,
        mask_feature_prob = mask_feature_prob,
        mask_feature_length = mask_feature_length,
        mask_feature_min_masks = mask_feature_min_masks,
        median_filter_width = median_filter_width,
        conv_in_channels = conv_in_channels,
        conv_out_channels = conv_out_channels,
        encoder_attention_heads = encoder_attention_heads,
        decoder_attention_heads = decoder_attention_heads,
        cross_attention_heads = cross_attention_heads,
        prune_conv_channels = prune_conv_channels,
        prune_ff = prune_ff,
        prune_ff_layer = prune_ff_layer,
        prune_heads = prune_heads,
        prune_layer = prune_layer
    )

    return WhisperForConditionalGeneration(config)