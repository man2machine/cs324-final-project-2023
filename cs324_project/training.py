# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:28 2023

@author: Shahir, Hashem, Bruce
"""

import os
import pathlib
import shutil
from typing import Optional

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
