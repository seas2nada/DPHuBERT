import sys
sys.path.append('.')
import os

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
from datasets import load_dataset

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
    def __init__(self, whisper_model_name, language="en", batch_size=8, dataset="google/fleurs"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_name)
        self.processor = AutoProcessor.from_pretrained(whisper_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(whisper_model_name)
        
        self.language = language
        self.batch_size = batch_size
        self.dataset = dataset

    def prepare_dataloader(self, dataset):

        def prepare_dataset(batch):
            # process audio
            sample = batch['audio']
            inputs = self.feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=True)
            
            # process audio length
            batch['input_features'] = inputs.get('input_features')[0]
            batch["input_length"] = len(sample["array"])
            batch["attention_mask"] = inputs.get("attention_mask")[0]

            # process targets
            input_str = batch['sentence'].lower() if 'sentence' in batch else batch['transcription'].lower()
            language = batch["locale"] if "locale" in batch else batch["language"].lower()
            if " " in language:
                language = language.split(" ")[-1]
            if "-" in language:
                language = language.split("-")[0]
            self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
            batch["labels"] = self.tokenizer(input_str).input_ids
            batch["language"] = language
            return batch

        dataset_sampling_rate = next(iter(dataset))['audio']['sampling_rate']
        if dataset_sampling_rate != self.feature_extractor.sampling_rate:
            dataset = dataset.cast_column(
                'audio', datasets.features.Audio(sampling_rate=self.feature_extractor.sampling_rate)
            )

        vectorized_datasets = dataset.map(
            prepare_dataset,
            num_proc=1,
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
        cv_16 = load_dataset(self.dataset, self.language, split="train")
        return self.prepare_dataloader(cv_16)

    def val_dataloader(self):
        cv_16 = load_dataset(self.dataset, self.language, split="validation")
        return self.prepare_dataloader(cv_16)

    def test_dataloader(self):
        cv_16 = load_dataset(self.dataset, self.language, split="test")
        return self.prepare_dataloader(cv_16)

def run_train(args):
    pl.seed_everything(2022)

    # Callbacks
    model_checkpoint = args.model_name

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        strategy="ddp",
        accelerator="gpu",
        devices=1,
        reload_dataloaders_every_n_epochs=1,
        precision=16,
    )

    # Create teacher model
    ckpt = torch.load(args.model_name, map_location="cpu")
    if 'config' not in ckpt.keys():
        org_ckpt = torch.load('pretrained/whisper-medium.hf.pth', map_location="cpu")
        ckpt['config'] = org_ckpt['config']
        ckpt['config'].update(
            dict(
                prune_conv_channels = False,
                prune_ff = True,
                prune_ff_layer = False,
                prune_heads = True,
                prune_layer = False,
            )
        )
    model = whisper_model(**ckpt['config'])
    _LG.info(f"Model:\n{model}")

    # Load model
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in ckpt['state_dict']:
                name = "student_model." + name
            p.copy_(ckpt['state_dict'][name])
    model.eval()
    
    # Load generation configs
    model.generation_config = GenerationConfig.from_pretrained(args.whisper_model_name)
    if "_" in args.language:
        language = args.language.split("_")[0]
    elif "-" in args.language:
        language = args.language.split("-")[0]
    elif args.language == "cmn_hans_cn":
        language = "zh"
    else:
        language = args.language

    distill_module = DistillModule(
        teacher_model=model,
        student_model=model,
        distill_mode="layer2layer",
        distill_layers=[],
        distill_linear_projs=[],
        distill_loss=None,
        ce_weight=None,
        distill_weight=None,
        learning_rate=None,
        weight_decay=None,
        warmup_updates=None,
        max_updates=None,
        use_reg=False,
        reg_learning_rate=None,
        target_sparsity=None,
        sparsity_warmup_updates=None,
        tsv_dir=None,
        label_dir=None,
        train_subset=None,
        seconds_per_batch=None,
        num_workers=None,
        language=language,
        whisper_model_name=args.whisper_model_name,
        exp_dir=args.exp_dir,
        dataset=args.dataset,
        eval_metric=args.eval_metric,
    )

    whisper_data = WhisperData(args.whisper_model_name, language=args.language, batch_size=args.batch_size, dataset=args.dataset)
    val_dataloader = whisper_data.test_dataloader()

    trainer.validate(distill_module, dataloaders=val_dataloader)

def _parse_args():
    parser = ArgumentParser(
        description="Joint distillation and pruning of whisper",
    )
    # distillation related
    parser.add_argument(
        "--model_name",
        default=pathlib.Path("pretrained/whisper-large-v3.hf.pth"),
        type=pathlib.Path,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("exp"),
        type=pathlib.Path,
        help="Path to the model checkpoint parent directory."
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
    # whisper model
    parser.add_argument(
        "--dataset",
        default="google/fleurs",
        type=str
    )
    # whisper model
    parser.add_argument(
        "--eval_metric",
        default="wer",
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
