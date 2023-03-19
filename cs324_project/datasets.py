# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:09:54 2023

@author: Shahir, Hashem, Bruce
"""

from enum import Enum
from dataclasses import dataclass
from functools import partial
from typing import Any, Union, Callable, Optional

from datasets import load_dataset, Dataset, Metric
from evaluate import load as load_metric
from transformers import PreTrainedTokenizerBase, BatchEncoding


class GlueDatasetTask(str, Enum):
    AX = 'ax'
    COLA = 'cola'
    MNLI = 'mnli'
    MNLI_MATCHED = 'mnli_matched'
    MNLI_MISMATCHED = 'mnli_mismatched'
    MRPC = 'mrpc'
    QNLI = 'qnli'
    QQP = 'qqp'
    RTE = 'rte'
    SST2 = 'sst2'
    STSB = 'stsb'
    WNLI = 'wnli'


@dataclass
class GlueTaskDatasets:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class GlueTaskDatasetInfo:
    task: GlueDatasetTask
    datasets: GlueTaskDatasets
    datasets_encoded_mlm: GlueTaskDatasets
    datasets_encoded_sc: GlueTaskDatasets
    tokenizer: PreTrainedTokenizerBase
    metric: Metric
    num_classes: int


DatasetEntryType = dict[str, Any]


GLUE_DATASET_TASK_TO_KEYS = {
    GlueDatasetTask.COLA: ('sentence',),
    GlueDatasetTask.MNLI_MATCHED: ('premise', 'hypothesis'),
    GlueDatasetTask.MNLI_MISMATCHED: ('premise', 'hypothesis'),
    GlueDatasetTask.MRPC: ('sentence1', 'sentence2'),
    GlueDatasetTask.QNLI: ('question', 'sentence'),
    GlueDatasetTask.QQP: ('question1', 'question2'),
    GlueDatasetTask.RTE: ('sentence1', 'sentence2'),
    GlueDatasetTask.SST2: ('sentence',),
    GlueDatasetTask.STSB: ('sentence1', 'sentence2'),
    GlueDatasetTask.WNLI: ('sentence1', 'sentence2')
}


def get_glue_dataset_task_keys(
        task: GlueDatasetTask) -> Union[tuple[str], tuple[str, str]]:

    return GLUE_DATASET_TASK_TO_KEYS[task]


def _encode_glue_dataset_func(
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase,
        add_word_ids: bool,
        examples: DatasetEntryType) -> BatchEncoding:

    sentence_keys = get_glue_dataset_task_keys(task)
    if len(sentence_keys) == 1:
        sentence1_key, = sentence_keys
        result = tokenizer(examples[sentence1_key], truncation=True)
    else:
        sentence1_key, sentence2_key = sentence_keys
        result = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
    if result.is_fast:
        result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]

    return result


def _get_encode_glue_dataset_func(
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase,
        add_word_ids: bool = False) -> Callable[[DatasetEntryType], DatasetEntryType]:

    return partial(_encode_glue_dataset_func, task, tokenizer, add_word_ids)


def encode_glue_dataset_mlm(
        datasets: GlueTaskDatasets,
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase):

    encode_func = _get_encode_glue_dataset_func(
        task=task,
        tokenizer=tokenizer,
        add_word_ids=True)
    remove_columns = list(get_glue_dataset_task_keys(task)) + ['label', 'idx']
    datasets_encoded = GlueTaskDatasets(
        train=datasets.train.map(encode_func, batched=True, remove_columns=remove_columns),
        val=datasets.val.map(encode_func, batched=True, remove_columns=remove_columns),
        test=datasets.test.map(encode_func, batched=True, remove_columns=remove_columns))

    return datasets_encoded


def encode_glue_dataset_sc(
        datasets: GlueTaskDatasets,
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase):

    encode_func = _get_encode_glue_dataset_func(
        task=task,
        tokenizer=tokenizer)
    remove_columns = list(get_glue_dataset_task_keys(task)) + ['idx']
    datasets_encoded = GlueTaskDatasets(
        train=datasets.train.map(encode_func, batched=True, remove_columns=remove_columns),
        val=datasets.val.map(encode_func, batched=True, remove_columns=remove_columns),
        test=datasets.test.map(encode_func, batched=True, remove_columns=remove_columns))

    return datasets_encoded


def load_glue_dataset_info(
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase,
        reduce_fraction: Optional[float] = None) -> GlueTaskDatasetInfo:

    dataset = load_dataset("glue", task.value)
    metric = load_metric("glue", task.value)

    if task == GlueDatasetTask.MNLI_MISMATCHED:
        val_key = 'validation_mismatched'
    elif task == GlueDatasetTask.MNLI:
        val_key = 'validation_matched'
    else:
        val_key = 'validation'
    
    if reduce_fraction:
        dataset['train'] = dataset['train'].train_test_split(int(len(dataset['train']) * reduce_fraction))['test']
        dataset[val_key] = dataset[val_key].train_test_split(int(len(dataset[val_key]) * reduce_fraction))['test']
        dataset['test'] = dataset['test'].train_test_split(int(len(dataset['test']) * reduce_fraction))['test']
        
    task_datasets = GlueTaskDatasets(
        train=dataset['train'],
        val=dataset[val_key],
        test=dataset['test'])
    task_datasets_encoded_mlm = encode_glue_dataset_mlm(
        datasets=task_datasets,
        task=task,
        tokenizer=tokenizer)
    task_datasets_encoded_sc = encode_glue_dataset_sc(
        datasets=task_datasets,
        task=task,
        tokenizer=tokenizer)
    num_classes = task_datasets.train.features['label'].num_classes

    dataset_info = GlueTaskDatasetInfo(
        task=task.value,
        datasets=task_datasets,
        datasets_encoded_mlm=task_datasets_encoded_mlm,
        datasets_encoded_sc=task_datasets_encoded_sc,
        tokenizer=tokenizer,
        metric=metric,
        num_classes=num_classes)

    return dataset_info
