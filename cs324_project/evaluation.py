# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:52 2023

@author: Shahir, Hashem, Bruce
"""

import os
import json
from dataclasses import dataclass
import collections
from enum import Enum
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cs324_project.datasets import GlueDatasetTask
from cs324_project.training import get_latest_checkpoint_path
from cs324_project.utils import get_rel_pkg_path


class KeyLabelPair(NamedTuple):
    key: str
    label: str


class MethodType(KeyLabelPair, Enum):
    RANDOM = KeyLabelPair('random', "Random")
    WHOLE_WORD = KeyLabelPair('whole_word', "Whole Word")
    TYPHOON = KeyLabelPair('typhoon', "Typhoon")


class MetricType(KeyLabelPair, Enum):
    LOSS_TRAIN = KeyLabelPair('loss', "Training Loss")
    LOSS_VAL = KeyLabelPair('eval_loss', "Validation Loss")
    F1_VAL = KeyLabelPair('eval_f1', "Validation F1-Score")
    ACC_VAL = KeyLabelPair('eval_accuracy', "Validation Accuracy")
    MATTHEWS_VAL = KeyLabelPair('eval_matthews_correlation', "Validation Matthews Correlation")
    PEARSON_VAL = KeyLabelPair('eval_pearson', "Validation Pearson Correlation")
    SPEARMAN_VAL = KeyLabelPair('eval_spearmanr', "Validation Spearman Rank Correlation")


class TrainingPhase(Enum):
    MASKED_LANGUAGE_MODEL = KeyLabelPair('mlm', "Masked Language Modeling")
    SEQUENCE_CLASSIFICATION = KeyLabelPair('sc', "Sequence Classification")


try:
    __IPYTHON__
    table_display = display
except NameError:
    __IPYTHON__ = False
    table_display = print


@dataclass
class TrainerStateParsed:
    best_metric_name: MetricType
    metrics_per_epoch: dict[MetricType, list[float]]
    best_epoch: int
    metrics_at_best_epoch: dict[MetricType, float]


def parse_trainer_state(
        checkpoints_dirname: os.PathLike,
        best_metric: MetricType,
        eval_metrics: list[MetricType]) -> TrainerStateParsed:

    checkpoints_dirname = get_latest_checkpoint_path(checkpoints_dirname)

    with open(os.path.join(checkpoints_dirname, "trainer_state.json"), 'r') as f:
        data = json.load(f)

    metrics_per_epoch = collections.defaultdict(list)

    for entry in data['log_history']:
        train_entry = MetricType.LOSS_TRAIN.value[0] in entry

        epoch = entry['epoch']
        assert epoch == int(epoch)
        epoch = int(epoch)

        metrics_to_process = [MetricType.LOSS_TRAIN] if train_entry else eval_metrics
        for metric_name in metrics_to_process:
            key = metric_name.value.key
            metrics_per_epoch[metric_name].append(entry[key])
            assert len(metrics_per_epoch[metric_name]) == epoch

    metrics_per_epoch = dict(metrics_per_epoch)

    sign = -1 if best_metric in [MetricType.LOSS_TRAIN, MetricType.LOSS_VAL] else 1
    best_epoch = np.argmax(sign * np.array(metrics_per_epoch[best_metric])).item()
    assert metrics_per_epoch[best_metric][best_epoch] == data['best_metric']
    metrics_at_best_epoch = {k: v[best_epoch] for k, v in metrics_per_epoch.items()}

    result = TrainerStateParsed(
        best_metric_name=best_metric,
        metrics_per_epoch=metrics_per_epoch,
        best_epoch=best_epoch,
        metrics_at_best_epoch=metrics_at_best_epoch)

    return result


def get_best_metric_name(
        task: GlueDatasetTask,
        training_phase: TrainingPhase) -> MetricType:

    if training_phase == training_phase.MASKED_LANGUAGE_MODEL:
        return MetricType.LOSS_VAL
    else:
        if task == GlueDatasetTask.STSB:
            return MetricType.PEARSON_VAL
        elif task == GlueDatasetTask.COLA:
            return MetricType.MATTHEWS_VAL
        else:
            return MetricType.ACC_VAL


def get_eval_metric_names(
        task: GlueDatasetTask,
        training_phase: TrainingPhase) -> list[MetricType]:

    if training_phase == training_phase.MASKED_LANGUAGE_MODEL:
        return [MetricType.LOSS_VAL]
    else:
        if task in (GlueDatasetTask.MRPC, GlueDatasetTask.QQP):
            return [MetricType.LOSS_VAL, MetricType.ACC_VAL, MetricType.F1_VAL]
        elif task == GlueDatasetTask.STSB:
            return [MetricType.LOSS_VAL, MetricType.PEARSON_VAL, MetricType.SPEARMAN_VAL]
        elif task == GlueDatasetTask.COLA:
            return [MetricType.LOSS_VAL, MetricType.MATTHEWS_VAL]
        else:
            return [MetricType.LOSS_VAL, MetricType.ACC_VAL]


def parse_task_results(
        task: GlueDatasetTask) -> dict[TrainingPhase, dict[MethodType, TrainerStateParsed]]:

    data = {}
    for training_phase in TrainingPhase:
        data[training_phase] = {}
        for method in MethodType:
            checkpoints_dirname = get_rel_pkg_path(
                os.path.join("models_save/", training_phase.value.key, task.value, method.value.key))
            print("Processing", checkpoints_dirname)
            best_metric_name = get_best_metric_name(task, training_phase)
            eval_metric_names = get_eval_metric_names(task, training_phase)
            trainer_state = parse_trainer_state(
                checkpoints_dirname=checkpoints_dirname,
                best_metric=best_metric_name,
                eval_metrics=eval_metric_names)
            data[training_phase][method] = trainer_state

    return data


def plot_task_results(
        task: GlueDatasetTask,
        results: dict[TrainingPhase, dict[MethodType, TrainerStateParsed]],
        save: bool = False):
    
    if save:
        save_dir = os.path.join("plots/", task.value)
        os.makedirs(save_dir, exist_ok=True)
    
    task_label = task.value.upper()
    for training_phase in TrainingPhase:
        data = results[training_phase]
        metrics = list(data[MethodType.RANDOM].metrics_per_epoch.keys())
        for metric in metrics:
            fig = plt.figure(figsize=(4, 3))
            ax = fig.add_subplot()

            for method in MethodType:
                y = data[method].metrics_per_epoch[metric]
                label = f"{method.value.label} Masking"
                ax.plot(np.arange(len(y)), y, label=label)

            title = f"{training_phase.value.label} {task_label} {metric.value.label}"
            ax.set_title(title, pad=20)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.value.label)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
            if not __IPYTHON__:
                fig.show()
            
            if save:
                fname = f"{training_phase.value.key}_{metric.value.key}.png"
                fig.savefig(os.path.join(save_dir, fname), bbox_inches='tight')        

        eval_metric_names = get_eval_metric_names(task, training_phase)

        method_labels = [method.value.label for method in MethodType]
        metric_labels = [metric.value.label for metric in eval_metric_names]
        rows = []
        for metric in eval_metric_names:
            rows.append([data[method].metrics_at_best_epoch[metric] for method in MethodType])
        df = pd.DataFrame(rows, metric_labels, method_labels)
        print(f"{training_phase.value.label} {task_label} Best Metrics for each Method")
        table_display(df)
        
        if save:
            fname = f"{training_phase.value.key}_best.csv"
            df.to_csv(os.path.join(save_dir, fname))    
