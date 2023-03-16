# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:20 2023

@author: Shahir, Hashem, Bruce
"""

import os
from enum import Enum
from typing import Union

from transformers import (
    PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer,
    AutoModelForSequenceClassification, AutoModelForMaskedLM)

from cs324_project.datasets import GlueTaskDatasetInfo
from cs324_project.utils import HF_AUTH_TOKEN

class ModelCheckpointName(str, Enum):
    DISTILBERT_HUGGINGFACE = "distilbert-base-uncased"
    BERT_TINY_GOOGLE = "prajjwal1/bert-tiny"
    TINYBERT_HUAWEI = "huawei-noah/TinyBERT_General_4L_312D"

def load_tokenizer(
        model_name_or_path: Union[ModelCheckpointName, os.PathLike]) -> PreTrainedTokenizerBase:
    
    if isinstance(model_name_or_path, ModelCheckpointName):
        model_name_or_path = model_name_or_path.value
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        use_auth_token=HF_AUTH_TOKEN)
    
    return tokenizer

def load_classification_model(
        model_name_or_path: Union[ModelCheckpointName, os.PathLike],
        dataset_info: GlueTaskDatasetInfo) -> PreTrainedModel:
    
    if isinstance(model_name_or_path, ModelCheckpointName):
        model_name_or_path = model_name_or_path.value
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=dataset_info.num_classes,
        use_auth_token=HF_AUTH_TOKEN)
    
    return model

def load_pretraining_model(
        model_name_or_path: Union[ModelCheckpointName, os.PathLike],
        dataset_info: GlueTaskDatasetInfo) -> PreTrainedModel:
    
    if isinstance(model_name_or_path, ModelCheckpointName):
        model_name_or_path = model_name_or_path.value
    
    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        num_labels=dataset_info.num_classes,
        use_auth_token=HF_AUTH_TOKEN)
    
    return model
