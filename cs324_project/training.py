# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:28 2023

@author: Shahir, Hashem, Bruce
"""

import os
import pathlib
import shutil
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

def get_latest_checkpoint_path(
        output_dir: os.PathLike) -> os.PathLike:

    checkpoint_dirs = sorted(pathlib.Path(output_dir).iterdir(), key=os.path.getmtime)

    return str(checkpoint_dirs[-1])


def delete_unneeded_checkpoints(
        *,
        trainer: Optional[Trainer] = None,
        args: Optional[TrainingArguments] = None,
        state: Optional[TrainerState] = None):
    
    if trainer is None:
        assert args is not None
        assert state is not None
    else:
        assert args is None
        assert state is None
        args = trainer.args
        state = trainer.state
    
    best_checkpoint = state.best_model_checkpoint
    latest_checkpoint = get_latest_checkpoint_path(args.output_dir)
    
    for output_dir in pathlib.Path(args.output_dir).iterdir():
        if os.path.isdir(output_dir) and str(output_dir) not in [best_checkpoint, latest_checkpoint]:
            shutil.rmtree(output_dir)


class DataSaverTrainerCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs) -> None:
        
        delete_unneeded_checkpoints(args=args, state=state)

def make_optimizer(
        params_to_update: list[torch.Tensor],
        lr: float = 1e-4,
        weight_decay: float = 1e-9,
        clip_grad_norm: bool = False) -> optim.Optimizer:

    optimizer = optim.AdamW(
        params_to_update,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=True)

    if clip_grad_norm:
        nn.utils.clip_grad_norm_(params_to_update, 3.0)

    return optimizer


def get_lr(
        optimizer: optim.Optimizer) -> None:

    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_optimizer_lr(
        optimizer: optim.Optimizer,
        lr: float) -> None:

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_scheduler(
        optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, threshold=0.025, patience=10, cooldown=5, min_lr=1e-6, verbose=True)

    return scheduler


def shrink_and_preturb(
        base_net: nn.Module,
        new_net: nn.Module,
        shrink: float = 0.5,
        perturb: float = 0.1) -> nn.Module:

    params1 = base_net.parameters()
    params2 = new_net.parameters()
    with torch.set_grad_enabled(False):
        for p1, p2 in zip(params1, params2):
            p1.mul_(shrink).add_(p2, alpha=perturb)

    return base_net
