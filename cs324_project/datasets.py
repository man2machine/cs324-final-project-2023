# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:09:54 2023

@author: Shahir, Hashem, Bruce
"""

from enum import Enum
from dataclasses import dataclass
from functools import partial
from typing import Any, Union, Callable

from datasets import load_dataset, Dataset, Metric
from evaluate import load as load_metric
from transformers import PreTrainedTokenizerBase

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
    datasets_encoded: GlueTaskDatasets
    tokenizer: PreTrainedTokenizerBase
    metric: Metric
    num_classes: int

DatasetEntryType = dict[str, Any]

def get_glue_dataset_task_keys(
        task: GlueDatasetTask) -> tuple[str, Union[str, None]]:
    
    task_to_keys = {
        GlueDatasetTask.COLA: ("sentence", None),
        GlueDatasetTask.MNLI_MATCHED: ("premise", "hypothesis"),
        GlueDatasetTask.MNLI_MISMATCHED: ("premise", "hypothesis"),
        GlueDatasetTask.MRPC: ("sentence1", "sentence2"),
        GlueDatasetTask.QNLI: ("question", "sentence"),
        GlueDatasetTask.QQP: ("question1", "question2"),
        GlueDatasetTask.RTE: ("sentence1", "sentence2"),
        GlueDatasetTask.SST2: ("sentence", None),
        GlueDatasetTask.STSB: ("sentence1", "sentence2"),
        GlueDatasetTask.WNLI: ("sentence1", "sentence2")
    }
    
    return task_to_keys[task]

def _preproc_glue_dataset_func(
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase,
        examples: DatasetEntryType):
    
    sentence1_key, sentence2_key = get_glue_dataset_task_keys(task)
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def _get_preproc_glue_dataset_func(
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase) -> Callable[[DatasetEntryType], DatasetEntryType]:
    
    return partial(_preproc_glue_dataset_func, task, tokenizer)

def preproc_glue_dataset(
        datasets: GlueTaskDatasets,
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase):
    
    preproc_func = _get_preproc_glue_dataset_func(task, tokenizer)
    encoded_datasets = GlueTaskDatasets(
        train=datasets.train.map(preproc_func, batched=True),
        val=datasets.val.map(preproc_func, batched=True),
        test=datasets.test.map(preproc_func, batched=True))
    
    return encoded_datasets

def load_glue_dataset_info(
        task: GlueDatasetTask,
        tokenizer: PreTrainedTokenizerBase) -> GlueTaskDatasetInfo:
    
    dataset = load_dataset('glue', task)
    metric = load_metric('glue', task)
    
    if task == GlueDatasetTask.MNLI_MISMATCHED:
        val_key = 'validation_mismatched'
    elif task == GlueDatasetTask.MNLI: 
        val_key = 'validation_matched'
    else:
        val_key = 'validation'
    
    task_datasets = GlueTaskDatasets(
        train=dataset['train'],
        val=dataset[val_key],
        test=dataset['test'])
    task_datasets_encoded = preproc_glue_dataset(
        datasets=task_datasets,
        task=task,
        tokenizer=tokenizer)
    num_classes = task_datasets.train.features['label'].num_classes
    
    dataset_info = GlueTaskDatasetInfo(
        task=task.value,
        datasets=task_datasets,
        datasets_encoded=task_datasets_encoded,
        tokenizer=tokenizer,
        metric=metric,
        num_classes=num_classes)
    
    return dataset_info
