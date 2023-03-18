# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 04:43:25 2023

@author: Shahir
"""

import collections
from functools import partial
from typing import Any

import numpy as np

from transformers import (
    default_data_collator, PreTrainedTokenizerBase, DataCollatorForLanguageModeling, DataCollator,
    DataCollatorWithPadding, PaddingStrategy)


def _whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        base_data_collator: DataCollator,
        prob: float,
        examples: list[dict[str, Any]]) -> dict[str, Any]:
    
    # Taken from https://huggingface.co/course/chapter7/3?fw=pt
    masked_examples = []
    for example in examples:
        word_ids = example.pop('word_ids')
        example['labels'] = example['input_ids'].copy()

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, prob, (len(mapping),))
        input_ids = example['input_ids']
        labels = example['labels']
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        example['labels'] = new_labels
        masked_examples.append(example)

    return base_data_collator(masked_examples)


def get_random_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=prob)

    return data_collator


def get_whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:
    
    default_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=PaddingStrategy.LONGEST)
    data_collator = partial(_whole_word_masking_data_collator, tokenizer, default_collator, prob)

    return data_collator
