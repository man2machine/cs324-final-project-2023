# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 02:33:59 2023

@author: Shahir, Hashem, Bruce
"""

import os
from functools import partial
from typing import Union, Callable, Any

import numpy as np

from datasets import Metric
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding, DataCollator, EvalPrediction
from transformers.utils import PaddingStrategy
from transformers import TrainingArguments, Trainer, PreTrainedModel, IntervalStrategy, DataCollator

from cs324_project.datasets import GlueDatasetTask, GlueTaskDatasetInfo
from cs324_project.utils import HF_AUTH_TOKEN, get_timestamp_str, get_rel_pkg_path


def _compute_metrics_sc_func(
        task: GlueDatasetTask,
        metric: Metric,
        eval_pred: EvalPrediction) -> Union[dict, None]:
    preds, labels = eval_pred
    if task != GlueDatasetTask.STSB:
        preds = np.argmax(preds, axis=1)
    else:
        preds = preds[:, 0]
    return metric.compute(predictions=preds, references=labels)


def _get_compute_metrics_sc_func(
        task: GlueDatasetTask,
        metric: Metric) -> Callable[[EvalPrediction], Union[dict, None]]:

    return partial(_compute_metrics_sc_func, task, metric)


def get_sc_data_collator(
        tokenizer: PreTrainedTokenizerBase) -> DataCollator:

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=PaddingStrategy.LONGEST)

    return data_collator


def get_training_args_sc(
        task: GlueDatasetTask,
        batch_size: int = 16,
        num_epochs: int = 1,
        verbose: bool = True) -> TrainingArguments:

    if task == GlueDatasetTask.STSB:
        metric_name = 'pearson'
    elif task == GlueDatasetTask.COLA:
        metric_name = 'matthews_correlation'
    else:
        metric_name = 'accuracy'
    output_dir = os.path.join(get_rel_pkg_path(
        "models/sc/"), "Model {}".format(get_timestamp_str()))
    if verbose:
        print("Creating training arguments, model output dir:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

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
        metric_for_best_model=metric_name,
        remove_unused_columns=False,
        hub_token=HF_AUTH_TOKEN)

    return args


def get_trainer_sc(
        dataset_info: GlueTaskDatasetInfo,
        model: PreTrainedModel,
        training_args: TrainingArguments) -> Trainer:

    data_collator = get_sc_data_collator(dataset_info.tokenizer)
    func = _get_compute_metrics_sc_func(dataset_info.task, dataset_info.metric)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_info.datasets_encoded_sc.train,
        eval_dataset=dataset_info.datasets_encoded_sc.val,
        tokenizer=dataset_info.tokenizer,
        data_collator=data_collator,
        compute_metrics=func)

    return trainer
