# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:28 2023

@author: Shahir, Hashem, Bruce
"""

import os
from functools import partial
from typing import Union, Callable
import pathlib

import numpy as np

from datasets import Metric
from transformers import TrainingArguments, Trainer, EvalPrediction, PreTrainedModel, BertModel

from cs324_project.datasets import GlueDatasetTask, GlueTaskDatasetInfo
from cs324_project.utils import HF_AUTH_TOKEN, get_timestamp_str, get_rel_pkg_path


def get_training_args(
        task: GlueDatasetTask,
        batch_size: int = 16,
        num_epochs: int = 1):

    if task == GlueDatasetTask.STSB:
        metric_name = 'pearson'
    elif task == GlueDatasetTask.COLA:
        metric_name = 'matthews_correlation'
    else:
        metric_name = 'accuracy'
    output_dir = os.path.join(get_rel_pkg_path("models/"), "Model {}".format(get_timestamp_str()))

    args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        hub_token=HF_AUTH_TOKEN)
    
    return args

def _compute_metrics_func(
        task: GlueDatasetTask,
        metric: Metric,
        eval_pred: EvalPrediction) -> Union[dict, None]:
    predictions, labels = eval_pred
    if task != GlueDatasetTask.STSB:
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def _get_compute_metrics_func_func(
        task: GlueDatasetTask,
        metric: Metric) -> Callable[[EvalPrediction], Union[dict, None]]:
    
    return partial(_compute_metrics_func, task, metric)

def get_trainer(
        dataset_info: GlueTaskDatasetInfo,
        model: PreTrainedModel,
        training_args: TrainingArguments) -> Trainer:
    
    func = _get_compute_metrics_func_func(dataset_info.task, dataset_info.metric)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_info.datasets_encoded.train,
        eval_dataset=dataset_info.datasets_encoded.val,
        tokenizer=dataset_info.tokenizer,
        compute_metrics=func)
    
    return trainer

def get_latest_checkpoint_path(
        training_args: TrainingArguments) -> os.PathLike:
    
    checkpoint_dirs = sorted(pathlib.Path(training_args.output_dir).iterdir(), key=os.path.getmtime)
    
    return os.path.abspath(checkpoint_dirs[-1])
