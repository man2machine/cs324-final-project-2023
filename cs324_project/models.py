# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:21:20 2023

@author: Shahir, Hashem, Bruce
"""

from enum import Enum

from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel

class ModelCheckpointName(Enum):
    DISTILBERT = "distilbert-base-uncased"
    BERT_TINY_GOOGLE = "prajjwal1/bert-tiny"
    TINYBERT_HUAWEI = "huawei-noah/TinyBERT_General_4L_312D"
    
def load_model_info(
        model_checkpoint: ModelCheckpointName) -> PreTrainedModel:
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
