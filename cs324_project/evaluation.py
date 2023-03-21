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

import numpy as np


class MetricName(Enum):
    LOSS_TRAIN = ('loss', 'loss_train', "Training Loss")
    LOSS_VAL = ('eval_loss', 'loss_val', "Validation Loss")
    F1_VAL = ('eval_f1', 'f1_val', "Validation F1-Score")
    ACC_VAL = ('eval_accuracy', 'acc_val', "Validation Accuracy")
    MATTHEWS_VAL = ('eval_matthews_correlation', 'matthews_corr_val', "Validation Matthews Correlation")
    PEARSON_VAL = ('eval_pearson', 'pearson_val', "Validation Pearson Correlation")
    

@dataclass
class TrainerStateParsed:
    best_metric_name: str
    metrics_per_epoch: dict[MetricName, list[float]]
    best_epoch: int
    metrics_at_best_epoch: dict[MetricName, float]


def parse_trainer_state(
        fname: os.PathLike,
        best_metric_name: MetricName,
        eval_metric_names: list[MetricName]) -> TrainerStateParsed:
    
    with open(fname, 'w') as f:
        data = json.load(f)
    
    metrics_per_epoch = collections.defaultdict(list)
    
    for entry in data['log_history']:
        train_entry = MetricName.LOSS_TRAIN.value[0] in entry
        
        epoch = train_entry['epoch']
        assert epoch == int(epoch)
        epoch = int(epoch)
        
        metrics_to_process = [MetricName.LOSS_TRAIN] if train_entry else eval_metric_names
        
        for metric_name in metrics_to_process:
            old_key, new_key, display_name = metric_name.value
            metrics_per_epoch[metric_name].append(entry[old_key])
            assert len(metrics_per_epoch[metric_name]) == epoch
    
    metrics_per_epoch = dict(metrics_per_epoch)
    
    best_epoch = np.argmax(metrics_per_epoch[best_metric_name]).item()
    metrics_at_best_epoch = {k: v[best_epoch] for k, v in metrics_per_epoch.items()}
    
    result = TrainerStateParsed(
        best_metric_name=best_metric_name,
        metrics_per_epoch=metrics_per_epoch,
        best_epoch=best_epoch,
        metrics_at_best_epoch=metrics_at_best_epoch)
    
    return result

