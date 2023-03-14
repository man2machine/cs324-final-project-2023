# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:20 2023

@author: Shahir, Hashem, Bruce
"""

from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification

from cs324_project.datasets import GlueTaskDatasetInfo
from cs324_project.utils import HF_AUTH_TOKEN

class ModelCheckpointName:
    DISTILBERT = "distilbert-base-uncased"
    BERT_TINY_GOOGLE = "prajjwal1/bert-tiny"
    TINYBERT_HUAWEI = "huawei-noah/TinyBERT_General_4L_312D"

def load_tokenizer(
        model_checkpoint_name: ModelCheckpointName) -> PreTrainedTokenizerBase:
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint_name,
        use_fast=True,
        use_auth_token=HF_AUTH_TOKEN)
    
    return tokenizer

def load_model(
        model_checkpoint_name: ModelCheckpointName,
        dataset_info: GlueTaskDatasetInfo) -> PreTrainedModel:
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint_name,
        num_labels=dataset_info.num_classes,
        use_auth_token=HF_AUTH_TOKEN)
    
    return model
