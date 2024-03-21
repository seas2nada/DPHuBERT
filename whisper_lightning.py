import math
import pathlib
from typing import Optional, List, Union, Any, Dict
from collections.abc import Mapping
import contextlib
import os

import pytorch_lightning as pl
import numpy as np
import random
import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import BatchSampler, RandomSampler

from fairseq.data import Dictionary

from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction

from torch_ema import ExponentialMovingAverage

import evaluate

id_to_lang = {0: "en", 1: "es", 2: "fr", 3: "pt", 4: "de", 5: "tr", 6: "ko", 7: "it", 8: "ro", 9: "ja", 10: "zh", 11: "ru"}

class LinearDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        elif self._step_count >= self.max_updates:
            return [0.0 for _ in self.base_lrs]
        else:
            pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
            return [base_lr * pct_remaining for base_lr in self.base_lrs]


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]


class DistillLoss(nn.Module):
    def __init__(self, l2_weight, l1_weight, cos_weight, cos_type):
        super().__init__()
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight
        self.cos_type = cos_type
        assert cos_type in ["raw", "log_sig"], cos_type

        if l2_weight != 0:
            self.mse_loss = nn.MSELoss()
        if l1_weight != 0:
            self.l1_loss = nn.L1Loss()
        if cos_weight != 0:
            self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def __repr__(self) -> str:
        return "{}(l2={}, l1={}, {}_cos={})".format(
            self.__class__.__name__,
            self.l2_weight,
            self.l1_weight,
            self.cos_type,
            self.cos_weight,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (batch, layer, time, feature)
            target: same shape as input
        """
        loss_mse = 0
        loss_l1 = 0
        loss_cos = 0
        if self.l2_weight != 0:
            loss_mse = self.mse_loss(input, target)
        if self.l1_weight != 0:
            loss_l1 = self.l1_loss(input, target)
        if self.cos_weight != 0:    # maximize cosine similarity
            if self.cos_type == "raw":
                loss_cos = -self.cos_sim(input, target).mean()
            elif self.cos_type == "log_sig":
                loss_cos = -self.cos_sim(input, target).sigmoid().log().mean()
            else:
                raise ValueError

        loss = self.l2_weight * loss_mse + self.l1_weight * loss_l1 + self.cos_weight * loss_cos

        return loss, (loss_mse, loss_l1, loss_cos)

class DistillModule(pl.LightningModule):
    def __init__(
        self,
        *,
        teacher_model,
        student_model,
        distill_mode: str,  # "layer2layer", "predlayer"
        distill_layers: List[int],  # layer indices to align, from 0 to num_layers
        distill_linear_projs: nn.ModuleList, # list of linear layers which transform student to teacher
        distill_loss: DistillLoss,
        ce_weight: float,
        distill_weight: float,
        learning_rate: float,
        weight_decay: float,
        warmup_updates: int,
        max_updates: int,
        use_reg: bool,  # whether to use the L0 regularization
        reg_learning_rate: Optional[float],   # lr for loga and lambda
        target_sparsity: Optional[float],
        sparsity_warmup_updates: Optional[int],   # linearly increase the target sparsity
        tsv_dir: Union[str, pathlib.Path],
        train_subset: str,
        seconds_per_batch: float,
        num_workers: int,
        label_dir: Union[str, pathlib.Path] = None,
        param_reg_type: str = "ema",
        threshold: float = 0.2,
        language: str = "tr",
        whisper_model_name: str = "openai/whisper-large-v3",
        exp_dir: str = "exp",
        dataset: str = "mozilla-foundation/common_voice_16_1",
        eval_metric: str = "wer",
    ):
        super().__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model

        self.original_num_params = sum(p.numel() for p in teacher_model.parameters())

        assert distill_mode in ["layer2layer", "predlayer"], distill_mode
        assert len(distill_layers) == len(distill_linear_projs)
        self.distill_mode = distill_mode
        self.distill_layers = distill_layers
        self.distill_linear_projs = distill_linear_projs
        self.distill_loss = distill_loss

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates

        self.use_reg = use_reg
        self.reg_learning_rate = reg_learning_rate
        self.target_sparsity = target_sparsity
        self.sparsity_warmup_updates = sparsity_warmup_updates

        # lambdas for Lagrangian
        if self.use_reg:
            self.lambda1 = nn.Parameter(torch.tensor(0.0))
            self.lambda2 = nn.Parameter(torch.tensor(0.0))

        # dataset related
        self.tsv_dir = tsv_dir
        self.label_dir = label_dir
        self.train_subset = train_subset
        self.seconds_per_batch = seconds_per_batch
        self.num_workers = num_workers

        # supervised training
        self.pad = True
        self.rand_crop = not self.pad
        self.distill_weight = distill_weight

        self.val_wer_sum = 0.0
        self.val_batch_count = 0

        self.param_reg_type = param_reg_type
        if param_reg_type == "ema":
            self.ema = ExponentialMovingAverage(self.student_model.parameters(), decay=0.99)
        elif param_reg_type == "l2":
            self.teacher_state_dict = self.teacher_model.state_dict()
            with torch.no_grad():
                for n, p in self.student_model.named_parameters():
                    if "log_alpha" in n:
                        self.teacher_state_dict[n] = p.data
            self.M = dict()
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(whisper_model_name)
        self.language = language
        self.whisper_model_name = whisper_model_name
        self.metric = evaluate.load(eval_metric)
        self.eval_metric = eval_metric
        self.exp_dir = exp_dir
        self.pred_str = []
        self.label_str = []
        self.dataset = dataset.split("/")[-1]

        self.ce_weight = ce_weight

    def configure_optimizers(self):
        main_params = [p for n, p in self.student_model.named_parameters() if "log_alpha" not in n]
        main_params.extend(list(self.distill_linear_projs.parameters()))
        pgs = [
            {
                'params': main_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'main_params',
            },
        ]
        if self.use_reg:
            pgs.extend(
                [
                    {
                        'params': [p for n, p in self.student_model.named_parameters() if "log_alpha" in n],
                        'lr': self.reg_learning_rate,
                        'weight_decay': 0.0,
                        'name': 'log_alpha',
                    },
                    {
                        'params': [self.lambda1, self.lambda2],
                        'lr': -self.reg_learning_rate,
                        'weight_decay': 0.0,
                        'name': 'lambda',
                    },
                ]
            )
        optimizer = torch.optim.AdamW(pgs)
        lr_scheduler = LinearDecayLRScheduler(
            optimizer, warmup_updates=self.warmup_updates, max_updates=self.max_updates
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def optimizer_step(self, *args, **kwargs):
        if self.param_reg_type == "l2":
            for n, p in self.student_model.named_parameters():
                if "log_alpha" in n:
                    self.teacher_state_dict[n] = p.data

        super().optimizer_step(*args, **kwargs)
        
        if self.param_reg_type == "ema":
            if self.ema.shadow_params[0].device != self.device:
                for i, p in enumerate(self.ema.shadow_params):
                    self.ema.shadow_params[i] = p.to(self.device)
                self.ema.update(self.student_model.parameters())

    def _get_target_sparsity(self):
        if self.global_step >= self.sparsity_warmup_updates:
            return self.target_sparsity
        return self.target_sparsity * (self.global_step / self.sparsity_warmup_updates)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        return inputs

    def _step(self, batch, batch_idx, mode):
        batch = self._prepare_inputs(batch)
        self.teacher_model.eval()
        batch['output_hidden_states'] = True
        if 'language' in batch.keys():
            batch.pop('language', None)

        with torch.no_grad():
            output = self.teacher_model(**batch)
            teacher_pred = output.logits
            teacher_enc_hiddens = output.encoder_hidden_states
            teacher_dec_hiddens = output.decoder_hidden_states
            teacher_enc_hiddens = torch.stack(
                [teacher_enc_hiddens[idx] for idx in self.distill_layers], dim=1
            )   # (batch, layer, time, feature)
            teacher_dec_hiddens = torch.stack(
                [teacher_dec_hiddens[idx] for idx in self.distill_layers], dim=1
            )   # (batch, layer, time, feature)
        
        self.student_model.train()
        output = self.student_model(**batch)
        student_pred = output.logits
        student_enc_hiddens = output.encoder_hidden_states
        student_dec_hiddens = output.decoder_hidden_states
        new_student_enc_hiddens = []
        new_student_dec_hiddens = []
        for idx, proj in zip(self.distill_layers, self.distill_linear_projs):
            new_student_enc_hiddens.append(proj(student_enc_hiddens[idx]))
            new_student_dec_hiddens.append(proj(student_dec_hiddens[idx]))
            
        student_enc_hiddens = torch.stack(new_student_enc_hiddens, dim=1)   # (batch, layer, time, feature)
        student_dec_hiddens = torch.stack(new_student_dec_hiddens, dim=1)   # (batch, layer, time, feature)

        loss_distill_enc, (loss_mse_enc, loss_l1_enc, loss_cos_enc) = self.distill_loss(student_enc_hiddens, teacher_enc_hiddens)
        loss_distill_dec, (loss_mse_dec, loss_l1_dec, loss_cos_dec) = self.distill_loss(student_dec_hiddens, teacher_dec_hiddens)
        loss_distill_pred, (loss_mse_pred, loss_l1_pred, loss_cos_pred) = self.distill_loss(student_pred, teacher_pred)
        loss_distill = loss_distill_enc + loss_distill_dec + loss_distill_pred
        loss_mse = loss_mse_enc + loss_mse_dec + loss_mse_pred
        loss_l1 = loss_l1_enc + loss_l1_dec + loss_l1_pred
        loss_cos = loss_cos_enc + loss_cos_dec + loss_cos_pred

        if self.use_reg:
            cur_target_sparsity = self._get_target_sparsity()
            cur_expected_sparsity = 1. - self.student_model.get_num_params() / self.original_num_params
            loss_reg = self.lambda1 * (cur_expected_sparsity - cur_target_sparsity) \
                + self.lambda2 * (cur_expected_sparsity - cur_target_sparsity)**2
        else:
            loss_reg = 0
        
        loss_sup = output.loss

        # Params L2-reg loss
        l2_reg = 0.0
        if self.param_reg_type == "l2":
            for n, p in self.student_model.named_parameters():
                reg_p = self.teacher_state_dict[n].to(p.device)
                if n in self.M:
                    M = self.M[n]
                else:
                    M = 0
                if "log_alpha" not in n:
                    l2_reg += torch.norm(p - (M * reg_p + (1-M) * p), p=2)
                elif "log_alpha" in n and self.global_step >= self.sparsity_warmup_updates:
                    l2_reg += torch.norm(p - (M * reg_p + (1-M) * p), p=2)
        
        self.l2_weight = 0.0006

        loss = self.distill_weight * loss_distill + loss_reg + self.ce_weight * loss_sup + self.l2_weight * l2_reg

        self.log_dict(
            {
                f"{mode}_loss": loss,   # total loss
                f"{mode}_loss_distill": loss_distill,   # distill total loss
                f"{mode}_loss_mse": loss_mse,
                f"{mode}_loss_l1": loss_l1,
                f"{mode}_loss_cos": loss_cos,
                f"{mode}_loss_reg": loss_reg,   # sparsity loss
                f"{mode}_loss_sup": loss_sup,   # supervised loss
                f"{mode}_l2_reg": l2_reg,
            }
        )
        if mode == "train" and self.use_reg:
            self.log_dict(
                {
                    'sparsity_expected': cur_expected_sparsity,
                    'sparsity_target': cur_target_sparsity,
                    'lambda1': self.lambda1,
                    'lambda2': self.lambda2,
                },
            )

        return loss

    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        return self.autocast_smart_context_manager()

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        ctx_manager = contextlib.nullcontext()

        return ctx_manager

    def _validation_step(self, batch, batch_idx, mode="valid"):
        self.student_model.eval()
        batch["output_hidden_states"] = False
        generation_inputs = batch.copy()

        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        labels = generation_inputs["labels"]
        if "language" in generation_inputs.keys():
            self.language = id_to_lang[generation_inputs["language"][0].item()]
            generation_inputs.pop("language", None)
        
        gen_kwargs = {'max_length': 225, 'synced_gpus': False, 'language': self.language, 'task': 'transcribe'}
        generated_tokens = self.student_model.generate(**generation_inputs, **gen_kwargs)
        
        self.tokenizer.set_prefix_tokens(language=self.language, task="transcribe")
        metrics = self.compute_metrics(EvalPrediction(predictions=generated_tokens, label_ids=labels))

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = self.student_model(**generation_inputs)
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()

        return loss, metrics

    def compute_metrics(self, pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True, group_tokens=True)
        
        for i in range(len(pred_str)):
            pred_str[i] = pred_str[i].lower()
            if self.language == "en":
                pred_str[i] = self.tokenizer._normalize(pred_str[i])
                label_str[i] = self.tokenizer._normalize(label_str[i])

        wer = self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "pred_str": pred_str, "label_str": label_str}
    
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.student_model.config.pad_token_id is not None:
                pad_token_id = self.student_model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    
    def on_after_backward(self):
        # This hook is called after loss.backward() and before optimizer.step()
        if self.param_reg_type == "l2":
            all_gradients = []

            # Loop through all the named parameters in the model
            for n, p in self.student_model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # Flatten the gradient to work with it as a single vector
                    grad_flat = p.grad.flatten()
                    all_gradients.append(grad_flat)

            # Concatenate all the gradients into a single tensor
            all_gradients = torch.cat(all_gradients)
            sorted_gradients = torch.sort(all_gradients.abs())[0]

            # Determine the threshold for the lowest 10% of gradient elements
            percentile_index = int(self.threshold * len(sorted_gradients))
            threshold = sorted_gradients[percentile_index]

            for n, p in self.student_model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # # Flatten the gradient to work with it as a single vector
                    # grad_flat = p.grad.flatten()
                    
                    # # Determine the threshold for the lowest 20% of gradient elements
                    # threshold = grad_flat.abs().quantile(0.1)
                    # # Create a mask where elements with gradients in the highest 80% are set to 1, others are set to 0
                    # mask = (grad_flat.abs() >= threshold).float().view_as(p)
                    
                    # Flatten the gradient to work with it as a single vector
                    grad_flat = p.grad.flatten()

                    # Create a mask where elements with gradients in the highest 80% are set to 1, others are set to 0
                    # mask = (grad_flat.abs() >= threshold).float().view_as(p) * 0.998 + 0.001
                    mask = (grad_flat.abs() >= threshold).float().view_as(p)

                    self.M[n] = mask

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._validation_step(batch, batch_idx, mode="valid")

        self.pred_str.extend(metrics["pred_str"])
        self.label_str.extend(metrics["label_str"])

        return loss

    def validation_epoch_end(self, outputs):
        # Write decoding results to a log file
        log_file_name = "decoding_log.txt"
        log_file_prefix = self.dataset + "_" + str(self.language)
        log_file_name = log_file_prefix + "-" + log_file_name

        log_file_path = os.path.join(self.exp_dir, log_file_name)
        if self.eval_metric == "wer":
            with open(log_file_path, 'w', encoding='utf-8') as log_file:

                # Calculate average WER at the end of the validation epoch
                wrongs = 0
                total_len = 0
                for pred, label in zip(self.pred_str, self.label_str):
                    pred = pred.lstrip(' ')
                    wrongs += editdistance.eval(pred.split(" "), label.split(" "))
                    total_len += len(label.split(" "))

                    log_file.write(f"  Prediction: {pred}\n")
                    log_file.write(f"  Reference: {label}\n")
                    log_file.write("\n")
        
        elif self.eval_metric == "cer":
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                # Calculate average CER at the end of the validation epoch
                wrongs = 0
                total_len = 0
                for pred, label in zip(self.pred_str, self.label_str):
                    pred = pred.replace(" ", "")
                    label = label.replace(" ", "")
                    
                    wrongs += editdistance.eval(pred, label)
                    total_len += len(label)

                    log_file.write(f"  Prediction: {pred}\n")
                    log_file.write(f"  Reference: {label}\n")
                    log_file.write("\n")

        avg_metric = wrongs / total_len

        metric_file_name = log_file_prefix + "-" + f"{self.eval_metric}.txt"
        metric_file_path = os.path.join(self.exp_dir, metric_file_name)
        with open(metric_file_path, 'w', encoding='utf-8') as f:            
            f.write(str(avg_metric))

        # Log the average WER using self.log
        self.log(f"val_{self.eval_metric}", avg_metric, on_step=False, on_epoch=True)

        # Reset the tracking variables for the next epoch
        self.pred_str = []
        self.label_str = []