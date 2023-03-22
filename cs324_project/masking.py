# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 04:43:25 2023

@author: Shahir, Hashem, Bruce
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
import torch.nn.functional as F

from transformers import (
    PreTrainedTokenizerBase, DataCollator, DataCollatorForTokenClassification, BertForMaskedLM, TrainingArguments,
    Trainer, PreTrainedModel, IntervalStrategy, DataCollator)
from transformers.utils import PaddingStrategy
from transformers.training_args import OptimizerNames

from cs324_project.datasets import GlueTaskDatasetInfo
from cs324_project.training import DataSaverTrainerCallback
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
class MaskingCorruptionConfig:
    original_prob: float = 0.0
    random_prob: float = 0.0


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
    dist_spread_frac: float = 0.4
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


def _random_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        base_data_collator: DataCollator,
        prob: float,
        examples: list[dict[str, Any]]) -> dict[str, Any]:

    masked_examples = []

    for example in examples:
        example.pop('word_ids')

        special_tokens_mask = np.array(
            tokenizer.get_special_tokens_mask(
                example['input_ids'],
                already_has_special_tokens=True),
            dtype=np.bool_)
        input_ids = np.array(example['input_ids'], dtype=np.int32)
        labels = input_ids.copy()

        mask_probs = np.full(labels.shape, prob)
        mask_probs[special_tokens_mask] = 0
        mask = np.random.random(len(labels)) < prob

        labels[~mask] = -100
        input_ids[mask] = tokenizer.mask_token_id

        example['input_ids'] = input_ids.tolist()
        example['labels'] = labels.tolist()

        masked_examples.append(example)

    return base_data_collator(masked_examples)


def _whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        base_data_collator: DataCollator,
        prob: float,
        examples: list[dict[str, Any]]) -> dict[str, Any]:

    # Modified from https://huggingface.co/course/chapter7/3?fw=pt
    masked_examples = []

    for example in examples:
        # Create a map between words and corresponding token indices
        word_ids = example.pop('word_ids')
        word_id_to_token_indices = collections.defaultdict(list)  # word id to token indices
        current_word_index = -1
        current_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:  # the None check handles special tokens that cannot be masked
                if word_id != current_word_id:
                    current_word_id = word_id
                    current_word_index += 1
                word_id_to_token_indices[current_word_index].append(idx)

        special_tokens_mask = np.array(
            tokenizer.get_special_tokens_mask(
                example['input_ids'],
                already_has_special_tokens=True),
            dtype=np.bool_)
        input_ids = np.array(example['input_ids'], dtype=np.int32)
        labels = input_ids.copy()

        word_mask = np.random.random(len(word_id_to_token_indices)) < prob
        masked_word_ids = np.where(word_mask)[0].tolist()
        masked_token_ids = np.array(sum([word_id_to_token_indices[i] for i in masked_word_ids], []), dtype=np.int32)
        mask = np.zeros(len(labels), dtype=np.bool_)
        mask[masked_token_ids] = 1
        mask[special_tokens_mask] = 0

        labels[~mask] = -100
        input_ids[mask] = tokenizer.mask_token_id

        example['input_ids'] = input_ids.tolist()
        example['labels'] = labels.tolist()

        masked_examples.append(example)

    return base_data_collator(masked_examples)


def get_random_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    base_data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=PaddingStrategy.LONGEST)
    data_collator = partial(_random_masking_data_collator, tokenizer, base_data_collator, prob)

    return data_collator


def get_whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    base_data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=PaddingStrategy.LONGEST)
    data_collator = partial(_whole_word_masking_data_collator, tokenizer, base_data_collator, prob)

    return data_collator


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
        self.token_id_to_weight = np.zeros(self.vocab_size, dtype=np.float32)
        self.ema_weight = self.mlm_args.masking_config.ema_weight
        self.base_data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=PaddingStrategy.LONGEST)

        input_ids_flat = sum(self.dataset_info.datasets_encoded_mlm.train[:]['input_ids'], [])
        input_ids_flat = np.array(input_ids_flat, dtype=np.int32)
        self.token_id_to_dataset_count = self._get_count_dist(input_ids_flat)
        self.token_id_to_dataset_count_mask = (self.token_id_to_dataset_count != 0)
        self.num_tokens_dataset = len(input_ids_flat)

        super().__init__(
            model=model,
            args=self.mlm_args._hf_training_args,
            train_dataset=self.dataset_info.datasets_encoded_mlm.train,
            eval_dataset=self.dataset_info.datasets_encoded_mlm.val,
            tokenizer=self.tokenizer,
            data_collator=self._data_collator,
            callbacks=[DataSaverTrainerCallback()]
        )

    def _get_count_dist(
            self,
            input_ids_flat: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:

        input_ids_unique, input_ids_unique_count = np.unique(input_ids_flat, return_counts=True)
        token_id_to_count = np.zeros(self.vocab_size, dtype=np.float32)
        token_id_to_count[input_ids_unique] = input_ids_unique_count.astype(np.float32)

        return token_id_to_count

    def _data_collator(
            self,
            examples: list[dict[str, Any]]) -> dict[str, Any]:

        id_masking_rate = self.get_id_to_masking_rate()
        masked_examples = []

        for example in examples:
            example.pop('word_ids')

            special_tokens_mask = np.array(
                self.tokenizer.get_special_tokens_mask(
                    example['input_ids'],
                    already_has_special_tokens=True),
                dtype=np.bool_)
            input_ids = np.array(example['input_ids'], dtype=np.int32)
            labels = input_ids.copy()

            mask_probs = id_masking_rate[input_ids]
            mask_probs[special_tokens_mask] = 0
            mask = np.random.random(len(labels)) < mask_probs

            labels[~mask] = -100
            input_ids[mask] = self.tokenizer.mask_token_id

            example['input_ids'] = input_ids.tolist()
            example['labels'] = labels.tolist()

            masked_examples.append(example)

        return self.base_data_collator(masked_examples)

    def training_step(
            self,
            model: BertForMaskedLM,
            inputs: dict[str, torch.Tensor]) -> torch.Tensor:

        model.train()

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        word_embeddings = self.model.bert.embeddings.word_embeddings
        input_ids = inputs.pop('input_ids')
        input_ids_one_hot = F.one_hot(input_ids, self.vocab_size).float()
        input_ids_one_hot.requires_grad = True
        inputs['inputs_embeds'] = torch.matmul(input_ids_one_hot, word_embeddings.weight)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        labels = inputs['labels']

        input_id_valid_mask = torch.stack([input_ids == n for n in self.tokenizer.all_special_ids], dim=0)
        input_id_valid_mask = (torch.sum(input_id_valid_mask, dim=0) == 0)
        label_valid_mask = (labels != -100)

        input_id_counts = self._get_count_dist(input_ids[input_id_valid_mask].cpu().numpy())
        input_id_counts_mask = (input_id_counts != 0)
        label_counts = self._get_count_dist(labels[label_valid_mask].cpu().numpy())
        label_counts_mask = (label_counts != 0)
        combined_counts_mask = np.logical_or(input_id_counts_mask, label_counts_mask)

        new_weights1 = np.zeros(self.vocab_size, dtype=np.float32)
        new_weights2 = np.zeros(self.vocab_size, dtype=np.float32)

        # inputs_grad = torch.norm(input_ids_one_hot.grad, dim=2)

        input_ids_unmasked = input_ids.detach().clone()
        input_ids_unmasked[label_valid_mask] = labels[label_valid_mask]
        inputs_grad = input_ids_one_hot.grad.detach()
        indices = torch.tensor(
            np.array([*np.indices(input_ids.shape), input_ids_unmasked.cpu().numpy()]),
            dtype=torch.int32,
            device=self.device)
        inputs_grad = inputs_grad[indices[0], indices[1], indices[2]]

        np.add.at(
            new_weights1,
            input_ids[input_id_valid_mask].cpu().numpy(),
            -inputs_grad[input_id_valid_mask].cpu().numpy())
        new_weights1[input_id_counts_mask] /= input_id_counts[input_id_counts_mask]

        np.add.at(
            new_weights2,
            labels[label_valid_mask].cpu().numpy(),
            inputs_grad[label_valid_mask].cpu().numpy())
        new_weights2[label_counts_mask] /= label_counts[label_counts_mask]

        new_weights = new_weights1 + new_weights2

        if self.first_step_done:
            self.token_id_to_weight[combined_counts_mask] *= (1 - self.ema_weight)
            self.token_id_to_weight[combined_counts_mask] += new_weights[combined_counts_mask] * self.ema_weight
        else:
            self.token_id_to_weight[combined_counts_mask] += new_weights[combined_counts_mask]
            self.first_step_done = True

        return loss.detach()

    def get_id_to_masking_rate(
            self) -> npt.NDArray[np.floating]:

        weights = self.token_id_to_weight.copy()
        mask = weights != 0
        if mask.sum():
            scaled_weights = weights[mask]
            # scaled_weights -= scaled_weights.min() + 0.1
            # scaled_weights = np.log(scaled_weights)
            scaled_weights -= scaled_weights.max()
            weights[mask] = scaled_weights
            weights -= weights.min()
            weights /= weights.max()
            # weights now ranges from 0 to 1

        target_prob = self.mlm_args.masking_config.prob
        init_min_prob = 1 - self.mlm_args.masking_config.dist_spread_frac
        init_max_prob = 1
        weights *= (init_max_prob - init_min_prob)
        weights += init_min_prob
        # now weights ranges from 1 - dist_spread_frac to 1

        current_mask_freq = np.sum(weights * self.token_id_to_dataset_count) / self.num_tokens_dataset
        weights *= target_prob / current_mask_freq

        current_mask_freq = np.sum(weights * self.token_id_to_dataset_count) / self.num_tokens_dataset
        assert np.abs(current_mask_freq - target_prob).item() < 1e-4
        assert weights.min() >= 0
        assert weights.max() <= 1

        return weights


def get_training_args_mlm(
        masking_config: BaseMaskingConfig,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_epochs: int = 1,
        verbose: bool = True) -> MaskedLanguageModelTrainingArgs:

    output_dir = os.path.join(get_rel_pkg_path("models/mlm/"), "Model {}".format(get_timestamp_str()))
    if verbose:
        print("Creating training arguments, model output dir:", output_dir)

    hf_training_args = TrainingArguments(
        output_dir,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        optim=OptimizerNames.ADAMW_TORCH,
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
            data_collator=data_collator,
            callbacks=[DataSaverTrainerCallback()])

    return trainer
