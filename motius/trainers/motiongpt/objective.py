"""Released MotionGPT instruction/pretraining objective construction."""
import random,numpy as np
import torch
from torch import Tensor
from typing import List


def forward_encdec(self, predict, texts: List[str], motion_tokens: Tensor, lengths: List[int], tasks: dict):
    motion_strings = self.motion_token_to_string(motion_tokens, lengths)
    condition = random.choice(['supervised', 'supervised', 'supervised'])
    if condition == 'text':
        inputs = texts
        outputs = texts
    elif condition == 'motion':
        inputs = motion_strings
        outputs = motion_strings
    else:
        inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)
    source_encoding = self.tokenizer(inputs, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
    source_attention_mask = source_encoding.attention_mask.to(motion_tokens.device)
    source_input_ids = source_encoding.input_ids.to(motion_tokens.device)
    if condition in ['text', 'motion']:
        batch_size, expandend_input_length = source_input_ids.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        target_mask = ~mask_indices
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        target_sentinel = self.create_sentinel_ids(target_mask.astype(np.int8))
        labels_input_ids = self.filter_input_ids(source_input_ids, target_sentinel)
        source_input_ids = self.filter_input_ids(source_input_ids, input_ids_sentinel)
    else:
        target_inputs = self.tokenizer(outputs, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
        labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = target_inputs.attention_mask.to(motion_tokens.device)
    labels_input_ids[labels_input_ids == 0] = -100
    outputs = predict(input_ids=source_input_ids, attention_mask=source_attention_mask if condition == 'supervised' else None, labels=labels_input_ids, decoder_attention_mask=lables_attention_mask if condition == 'supervised' else None)
    return outputs


def forward_dec(self, predict, texts: List[str], motion_tokens: Tensor, lengths: List[int], tasks: dict):
    self.tokenizer.padding_side = 'right'
    motion_strings = self.motion_token_to_string(motion_tokens, lengths)
    condition = random.choice(['text', 'motion', 'supervised', 'supervised', 'supervised'])
    if condition == 'text':
        labels = texts
    elif condition == 'motion':
        labels = motion_strings
    else:
        inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)
        labels = []
        for i in range(len(inputs)):
            labels.append(inputs[i] + ' \n ' + outputs[i] + self.tokenizer.eos_token)
    inputs = self.tokenizer(labels, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True, return_tensors='pt')
    labels_input_ids = inputs.input_ids.to(motion_tokens.device)
    lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
    outputs = predict(input_ids=labels_input_ids, attention_mask=lables_attention_mask, labels=inputs['input_ids'])
    return outputs
