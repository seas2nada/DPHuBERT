"""Perform distillation and pruning."""

import logging
import pathlib
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import strategies, plugins
from lightning_lite.utilities.rank_zero import _get_rank

from whisper_lightning import (
    DistillModule,
    DistillLoss,
)
from whisper.model import (
    whisper_model,
)

from pytorch_lightning.loggers import WandbLogger

import datasets
from datasets import load_dataset, concatenate_datasets

from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.utils import is_datasets_available

from dataset.audio_dataset import (
    BucketizeBatchSampler,
    DistributedBatchSampler,
    CollateFnWhisper,
    WhisperDataset,
    DataCollatorSpeechSeq2SeqWithPadding
)

_LG = logging.getLogger(f"{__name__}:{_get_rank()}")

class WhisperData:
    def __init__(self, whisper_model_name, language="en", batch_size=8, cache_file_name="/home/ubuntu/Workspace/huggingface/commonvoice"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_name)
        self.processor = AutoProcessor.from_pretrained(whisper_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(whisper_model_name)
        
        self.language = language
        self.batch_size = batch_size

        self.cache_file_name = cache_file_name

    def prepare_dataloader(self, dataset, split="test"):

        def prepare_dataset(batch):
            # process audio
            sample = batch['audio']
            inputs = self.feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=True)
            
            # process audio length
            batch['input_features'] = inputs.get('input_features')[0]
            batch["input_length"] = len(sample["array"])
            batch["attention_mask"] = inputs.get("attention_mask")[0]

            # process targets
            input_str = batch['sentence'].lower()
            language = batch["locale"] if "locale" in batch else batch["language"]
            self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
            batch["labels"] = self.tokenizer(input_str).input_ids
            batch["language"] = language
            return batch

        dataset_sampling_rate = next(iter(dataset))['audio']['sampling_rate']
        if dataset_sampling_rate != self.feature_extractor.sampling_rate:
            dataset = dataset.cast_column(
                'audio', datasets.features.Audio(sampling_rate=self.feature_extractor.sampling_rate)
            )

        cache_file_name = self.cache_file_name + "_" + split + ".arrow"
        vectorized_datasets = dataset.map(
            prepare_dataset,
            num_proc=1,
            cache_file_name=cache_file_name,
            desc="preprocess dataset",
        )

        # filter data that is shorter than min_input_length or longer than
        # max_input_length
        max_input_length = 30 * self.feature_extractor.sampling_rate
        min_input_length = 0.0
        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=1,
            input_columns=["input_length"],
        )

        if is_datasets_available() and isinstance(vectorized_datasets, datasets.Dataset):
            lengths = (
                vectorized_datasets["input_length"]
                if "input_length" in vectorized_datasets.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        sampler = LengthGroupedSampler(
            self.batch_size,
            dataset=vectorized_datasets,
            lengths=lengths,
            model_input_name=model_input_name,
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=50258,
            forward_attention_mask=True,
        )

        # batch_sampler = BatchSampler(RandomSampler(vectorized_datasets), batch_size=32, drop_last=False)
        dataloader = DataLoader(vectorized_datasets, batch_size=self.batch_size, collate_fn=data_collator, num_workers=12, sampler=sampler, drop_last=False)

        return dataloader

    def train_dataloader(self):
        cv_16 = []
        for language in self.language.split("+"):
            cv_16.append(load_dataset("mozilla-foundation/common_voice_16_1", language, split="train"))
        cv_16 = concatenate_datasets(cv_16)
        return self.prepare_dataloader(cv_16, "train")

    def val_dataloader(self):
        cv_16 = load_dataset("mozilla-foundation/common_voice_16_1", "es", split="validation")
        return self.prepare_dataloader(cv_16, "valid")

    def test_dataloader(self):
        cv_16 = load_dataset("mozilla-foundation/common_voice_16_1", self.language, split="test")
        return self.prepare_dataloader(cv_16)


def _init_layer_transform(module: nn.Linear):
    module.weight.data.copy_(torch.eye(len(module.weight)))
    module.bias.data.fill_(0)


def run_train(args):
    pl.seed_everything(2022)

    # Callbacks
    lr_monitor = LearningRateMonitor()  # log learning rates for all param groups
    model_checkpoint = ModelCheckpoint(dirpath=args.exp_dir / "ckpts", verbose=True)   # only save the latest epoch
    callbacks = [lr_monitor, model_checkpoint]

    wandb_logger = WandbLogger(name=args.exp_dir.name.split('/')[-1], project=args.project_name, save_dir="logs")

    # strategy = strategies.DeepSpeedStrategy(partition_activations=True)

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        callbacks=callbacks,
        max_steps=args.max_updates,
        strategy="ddp",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accumulate_grad_batches=args.accum_grad,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=args.clip_norm,
        log_every_n_steps=args.log_interval,
        precision=16,
        logger=wandb_logger,
    )

    # Create teacher model
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher_model = whisper_model(**teacher_ckpt['config'])
    _LG.info(f"Teacher model:\n{teacher_model}")
    teacher_result = teacher_model.load_state_dict(teacher_ckpt['state_dict'], strict=False)
    _LG.info(f"Load pretrained ckpt to teacher: missing {teacher_result.missing_keys}, unexpected {teacher_result.unexpected_keys}")

    # Load teacher model
    with torch.no_grad():
        for name, p in teacher_model.named_parameters():
            if "dummy_weight" in name:
                continue
            p.copy_(teacher_ckpt['state_dict'][name])

    # Freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False
    _LG.info("Freeze parameters of the teacher model by setting requires_grad=False")
    teacher_model.eval()
    
    # Create student model
    student_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    pruning_units = args.pruning_units.split(",")
    _LG.info(f"Pruning units: {pruning_units}")
    student_config = student_ckpt['config']
    student_config.update(
        dict(
            prune_conv_channels = False,
            prune_ff = True,
            prune_ff_layer = False,
            prune_heads = True,
            prune_layer = False,
        )
    )
    student_model = whisper_model(**student_config)
    _LG.info(f"Student model:\n{student_model}")
    student_result = student_model.load_state_dict(student_ckpt['state_dict'], strict=False)
    _LG.info(f"Load pretrained ckpt to student: missing {student_result.missing_keys}, unexpected {student_result.unexpected_keys}")

    # Load student model
    with torch.no_grad():
        for name, p in student_model.named_parameters():
            if "dummy_weight" in name or "hard_concrete" in name:
                continue
            p.copy_(student_ckpt['state_dict'][name])

    # Freeze the parameters of proj_out
    for param in student_model.proj_out.parameters():
        param.requires_grad = False

    # apply gradient checkpointing for student model
    student_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    # Load generation configs
    teacher_model.generation_config = GenerationConfig.from_pretrained(args.whisper_model_name)
    student_model.generation_config = GenerationConfig.from_pretrained(args.whisper_model_name)

    # Create linear layers which transform student hiddens to teacher hiddens
    distill_layer_groups = [[int(l) for l in g.split(",")] for g in args.distill_layers.split(".")]
    _LG.info(f"Distill transformer layers: {distill_layer_groups}")
    distill_layers = []
    for g in distill_layer_groups:
        distill_layers.extend(g)
    for i, idx in enumerate(distill_layers):
        distill_layers[i] = idx - 1

    student_embed_dim = student_model.proj_out.in_features
    teacher_embed_dim = teacher_model.proj_out.in_features

    distill_linear_projs = nn.ModuleList()
    for g in distill_layer_groups:      # layers in the same group share a linear layer
        tmp_linear = nn.Linear(student_embed_dim, teacher_embed_dim)
        _init_layer_transform(tmp_linear)
        for _ in range(len(g)):
            distill_linear_projs.append(tmp_linear)

    # Create DistillLoss module
    distill_loss_criterion = DistillLoss(
        l2_weight=args.l2_weight,
        l1_weight=args.l1_weight,
        cos_weight=args.cos_weight,
        cos_type=args.cos_type,
    )
    _LG.info(f"Distill loss module:\n{distill_loss_criterion}")

    distill_module = DistillModule(
        teacher_model=teacher_model,
        student_model=student_model,
        distill_mode=args.distill_mode,
        distill_layers=distill_layers,
        distill_linear_projs=distill_linear_projs,
        distill_loss=distill_loss_criterion,
        ce_weight=args.ce_weight,
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
        language=args.language,
        whisper_model_name=args.whisper_model_name,
        exp_dir=args.exp_dir,
    )

    whisper_data = WhisperData(args.whisper_model_name, language=args.language, batch_size=args.batch_size, cache_file_name=args.cache_file_name)
    train_dataloader = whisper_data.train_dataloader()
    val_dataloader = whisper_data.val_dataloader()

    trainer.fit(
        distill_module, 
        ckpt_path=args.resume_checkpoint,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


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
        default="bf16",
        type=str,
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
        "--ce_weight",
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

    # whisper model
    parser.add_argument(
        "--language",
        default="tr",
        type=str
    )
    # whisper model
    parser.add_argument(
        "--whisper_model_name",
        default="openai/whisper-large-v3",
        type=str
    )
    # whisper model
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int
    )
    parser.add_argument(
        "--cache_file_name",
        default="/home/ubuntu/Workspace/huggingface/commonvoice",
        type=str
    )
    

    
    return parser.parse_args()


def _init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    if _get_rank() == 0:
        _LG.setLevel(logging.INFO)
    else:
        _LG.setLevel(logging.WARN)


def cli_main():
    _init_logger()
    args = _parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
