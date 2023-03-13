# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:28 2023

@author: Shahir, Hashem, Bruce
"""

import numpy as np

from datasets import Metric
from transformers import TrainingArguments, Trainer, EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase

from cs324_project.datasets import GlueDatasetTask, GlueTaskDatasetInfo
from cs324_project.utils import get_timestamp_str


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
    output_dir = "Model {}".format(get_timestamp_str())

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
        push_to_hub=True)
    
    return args

def compute_metrics(
        task: GlueDatasetTask,
        metric: Metric,
        eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    if task != GlueDatasetTask.STSB:
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def get_trainer(
        dataset_info: GlueTaskDatasetInfo,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_args: TrainingArguments) -> Trainer:
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_info.train,
        eval_dataset=dataset_info.val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)
    
    return trainer
