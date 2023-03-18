# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 04:43:25 2023

@author: Shahir
"""

import collections
from functools import partial

import numpy as np

from transformers import (
    default_data_collator, BatchEncoding, PreTrainedTokenizerBase, DataCollatorForLanguageModeling, DataCollator)


def _whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float,
        examples_encoded: BatchEncoding) -> DataCollator:
    
    # Taken from https://huggingface.co/course/chapter7/3?fw=pt
    for example in examples_encoded:
        print(example)
        example = example.copy()
        word_ids = example.pop('word_ids')

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

    return default_data_collator(examples_encoded)


def get_random_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=prob)

    return data_collator


def get_whole_word_masking_data_collator(
        tokenizer: PreTrainedTokenizerBase,
        prob: float = 0.2) -> DataCollator:

    data_collator = partial(_whole_word_masking_data_collator, tokenizer, prob)

    return data_collator
