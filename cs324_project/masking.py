# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 04:43:25 2023

@author: Shahir
"""

import os
import abc
import collections
from functools import partial
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from transformers import (
    PreTrainedTokenizerBase, DataCollatorForLanguageModeling, DataCollator, DataCollatorForTokenClassification,
    BertForMaskedLM)
from transformers.utils import PaddingStrategy
from transformers import TrainingArguments, Trainer, PreTrainedModel, IntervalStrategy, DataCollator

import tqdm

from cs324_project.datasets import GlueTaskDatasetInfo
from cs324_project.utils import HF_AUTH_TOKEN, get_timestamp_str, get_rel_pkg_path


class MaskingMethod(Enum):
    RANDOM = 0
    WHOLE_WORD = 1
    TYPHOON = 2


class BaseMaskingConfig(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def masking_method(
            self):

        pass


@dataclass(frozen=True)
class RandomMaskingConfig(BaseMaskingConfig):
    masking_method: MaskingMethod = field(init=False, default=MaskingMethod.RANDOM)
    prob: float = 0.3


@dataclass(frozen=True)
class WholeWordMaskingConfig(BaseMaskingConfig):
    masking_method: MaskingMethod = field(init=False, default=MaskingMethod.WHOLE_WORD)
    prob: float = 0.3


@dataclass(frozen=True)
class TyphoonMaskingConfig(BaseMaskingConfig):
    masking_method: MaskingMethod = field(init=False, default=MaskingMethod.TYPHOON)
    prob: float = 0.3
    max_prob: float = 0.5
    ema_weight: float = 0.8


@dataclass(frozen=True)
class MaskedLanguageModelTrainingArgs:
    masking_config: Union[RandomMaskingConfig, WholeWordMaskingConfig, TyphoonMaskingConfig]
    output_dir: os.PathLike
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    _hf_training_args: TrainingArguments


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
    data_collator = partial(_whole_word_masking_data_collator, tokenizer, base_data_collator, prob)

    return data_collator


def get_training_args_mlm(
        masking_config: BaseMaskingConfig,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_epochs: int = 1,
        verbose: bool = True) -> MaskedLanguageModelTrainingArgs:

    output_dir = os.path.join(get_rel_pkg_path(
        "models/mlm/"), "Model {}".format(get_timestamp_str()))
    if verbose:
        print("Creating training arguments, model output dir:", output_dir)

    hf_training_args = TrainingArguments(
        output_dir,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        hub_token=HF_AUTH_TOKEN)

    args = MaskedLanguageModelTrainingArgs(
        masking_config=masking_config,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        _hf_training_args=hf_training_args)

    return args


class TyphoonMaskedLanguageModelTrainer(Trainer):
    def __init__(
            self,
            dataset_info: GlueTaskDatasetInfo,
            mlm_args: MaskedLanguageModelTrainingArgs,
            model: PreTrainedModel):

        self.dataset_info = dataset_info
        self.mlm_args = mlm_args
        self.model = model

        self.device = self.mlm_args._hf_training_args.device
        self.first_step_done = False
        self.tokenizer = self.dataset_info.tokenizer
        self.vocab_size = max(self.tokenizer.get_vocab().values()) + 1
        self.token_id_to_weight = torch.zeros(
            self.vocab_size, dtype=torch.float32, device=self.device)
        self.ema_weight = self.mlm_args.masking_config.ema_weight
        self.rng = np.random.default_rng()
        self.base_data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=PaddingStrategy.LONGEST)

        super().__init__(
            model=model,
            args=self.mlm_args._hf_training_args,
            train_dataset=self.dataset_info.datasets_encoded_mlm.train,
            eval_dataset=self.dataset_info.datasets_encoded_mlm.val,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

    def data_collator(
            self,
            examples: list[dict[str, Any]]) -> dict[str, Any]:

        id_masking_rate = self.get_id_to_masking_rate()
        masked_examples = []
        for example in examples:
            example.pop('word_ids')
            example['labels'] = example['input_ids'].copy()

            input_ids = example['input_ids']
            labels = example['labels']
            new_labels = [-100] * len(labels)

            mask_probs = id_masking_rate[input_ids]
            mask = self.rng.random(len(labels)) < mask_probs

            for i, masked in enumerate(mask):
                if masked:
                    new_labels[i] = labels[i]
                    input_ids[i] = self.tokenizer.mask_token_id
            example['labels'] = new_labels

            masked_examples.append(example)

        return self.base_data_collator(masked_examples)

    def training_step(
            self,
            model: BertForMaskedLM,
            inputs: dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        word_embeddings = self.model.bert.embeddings.word_embeddings
        input_ids = inputs.pop('input_ids')
        input_ids_one_hot = F.one_hot(input_ids, self.vocab_size).float()
        input_ids_one_hot.requires_grad = True
        inputs['inputs_embeds'] = torch.matmul(
            input_ids_one_hot, word_embeddings.weight)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        inputs_grad = torch.norm(input_ids_one_hot.grad, dim=2)

        special_token_mask = torch.stack(
            [input_ids == n for n in self.tokenizer.all_special_ids], dim=0)
        special_token_mask = (torch.sum(special_token_mask, dim=0) == 0)
        input_ids_flat = input_ids[special_token_mask]
        inputs_grad_flat = torch.clamp(inputs_grad[special_token_mask], min=0)

        counts = torch.zeros(
            self.vocab_size, dtype=torch.float32, device=self.device)
        counts[input_ids_flat] += 1
        count_mask = (counts != 0)
        new_weights = torch.zeros(
            self.vocab_size, dtype=torch.float32, device=self.device)
        new_weights[input_ids_flat] += inputs_grad_flat
        new_weights[count_mask] /= counts[count_mask]

        if self.first_step_done:
            self.token_id_to_weight[count_mask] *= (1 - self.ema_weight)
            self.token_id_to_weight[count_mask] += new_weights[count_mask] * self.ema_weight
        else:
            self.token_id_to_weight[count_mask] += new_weights[count_mask]
            self.first_step_done = True

        return loss.detach()

    def get_id_to_masking_rate(
            self) -> npt.NDArray[np.floating]:

        weights = self.token_id_to_weight.cpu().numpy()
        mask = weights != 0
        if mask.sum():
            scaled_weights = np.log(weights[mask])
            scaled_weights -= scaled_weights.min()
            weights[mask] = scaled_weights
            weights /= weights.max()

        min_prob = self.mlm_args.masking_config.prob
        max_prob = self.mlm_args.masking_config.max_prob
        weights *= (max_prob - min_prob)
        weights += min_prob
        weights += (min_prob - weights.mean())

        return weights


def get_trainer_mlm(
        dataset_info: GlueTaskDatasetInfo,
        mlm_args: MaskedLanguageModelTrainingArgs,
        model: PreTrainedModel) -> Trainer:

    masking_method = mlm_args.masking_config.masking_method
    if masking_method == mlm_args.masking_config.masking_method == MaskingMethod.TYPHOON:
        trainer = TyphoonMaskedLanguageModelTrainer(
            dataset_info=dataset_info,
            mlm_args=mlm_args,
            model=model)
    else:
        if masking_method == MaskingMethod.RANDOM:
            data_collator = get_random_masking_data_collator(
                tokenizer=dataset_info.tokenizer,
                prob=mlm_args.masking_config.prob)
        elif masking_method == MaskingMethod.WHOLE_WORD:
            data_collator = get_whole_word_masking_data_collator(
                tokenizer=dataset_info.tokenizer,
                prob=mlm_args.masking_config.prob)

        trainer = Trainer(
            model=model,
            args=mlm_args._hf_training_args,
            train_dataset=dataset_info.datasets_encoded_mlm.train,
            eval_dataset=dataset_info.datasets_encoded_mlm.val,
            tokenizer=dataset_info.tokenizer,
            data_collator=data_collator)

    return trainer
