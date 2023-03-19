# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 04:43:25 2023

@author: Shahir
"""

import os
import collections
from functools import partial
from typing import Any, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import (
    PreTrainedTokenizerBase, DataCollatorForLanguageModeling, DataCollator, DataCollatorForTokenClassification)
from transformers.utils import PaddingStrategy
from transformers import TrainingArguments, Trainer, PreTrainedModel, IntervalStrategy, DataCollator

import tqdm

from cs324_project.datasets import GlueTaskDatasetInfo
from cs324_project.utils import HF_AUTH_TOKEN, get_timestamp_str, get_rel_pkg_path


def _whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        base_data_collator: DataCollator,
        prob: float,
        examples: list[dict[str, Any]]) -> dict[str, Any]:

    # Taken from https://huggingface.co/course/chapter7/3?fw=pt
    masked_examples = []
    for example in examples:
        word_ids = example.pop('word_ids')
        example['labels'] = example['input_ids'].copy()

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, prob, (len(mapping),))
        input_ids = example['input_ids']
        labels = example['labels']
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        example['labels'] = new_labels
        masked_examples.append(example)

    return base_data_collator(masked_examples)


def _random_masking_data_collator(
        base_data_collator: DataCollator,
        examples: list[dict[str, Any]]) -> dict[str, Any]:

    examples = [{k: v for k, v in example.items() if k not in ['word_ids']}
                for example in examples]

    return base_data_collator(examples)


def get_random_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    base_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=prob)
    data_collator = partial(_random_masking_data_collator, base_data_collator)

    return data_collator


def get_whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    base_data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=PaddingStrategy.LONGEST)
    data_collator = partial(_whole_word_masking_data_collator,
                            tokenizer, base_data_collator, prob)

    return data_collator


def get_training_args_mlm(
        batch_size: int = 16,
        num_epochs: int = 1,
        verbose: bool = True) -> TrainingArguments:

    output_dir = os.path.join(get_rel_pkg_path(
        "models/mlm/"), "Model {}".format(get_timestamp_str()))
    if verbose:
        print("Creating training arguments, model output dir:", output_dir)

    args = TrainingArguments(
        output_dir,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        hub_token=HF_AUTH_TOKEN)

    return args


def get_trainer_mlm(
        dataset_info: GlueTaskDatasetInfo,
        model: PreTrainedModel,
        training_args: TrainingArguments,
        data_collator: Optional[DataCollator] = None) -> Trainer:

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_info.datasets_encoded_mlm.train,
        eval_dataset=dataset_info.datasets_encoded_mlm.val,
        tokenizer=dataset_info.tokenizer,
        data_collator=data_collator)

    return trainer


def run_manual_training_mlm(
        *,
        device: torch.device,
        model: nn.Module,
        dataloader_train: DataLoader,
        dataloader_test: DataLoader,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
        num_epochs: int = 1):

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_count = 0

        pbar = tqdm(dataloader_train)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            current_loss = loss.detach().item()
            num_samples = batch['labels'].size(0)
            running_loss += current_loss * num_samples
            running_count += num_samples
            avg_loss = running_loss / running_count
            desc = "Avg. Loss: {:.4f}, Current Loss: {:.4f}"
            desc = desc.format(avg_loss, current_loss)
            pbar.set_description(desc)

            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler:
                lr_scheduler.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()
