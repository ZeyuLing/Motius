from transformers.generation import GenerationMixin
import math
import os
import torch
from torch import nn
from einops import rearrange
from transformers import GPT2LMHeadModel, GPT2Model, AutoTokenizer
from typing import Optional, Tuple, List, Union, Callable
from transformers.utils import logging
import inspect
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.cache_utils import StaticCache, Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import is_torchdynamo_compiling
import random
import numpy as np

class MotionGPT3LanguageSampler(GenerationMixin):

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, type_ids: torch.Tensor=None, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, token_type_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, forward_mod: np.ndarray[int]=None, mot_labels=None):
        return self.model(type_ids=type_ids, input_ids=input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values, token_type_ids=token_type_ids, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, forward_mod=forward_mod, mot_labels=mot_labels)

    def generate(self, inputs: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None, synced_gpus: Optional[bool]=None, mode: Union[str, List[str]]='text', **kwargs):
        tokenizer = kwargs.pop('tokenizer', None)
        self.current_mode = mode
        self.current_mod_id = self.mod2id[mode]
        modality_info = self.modality_infos[self.current_mod_id]
        som_id, eom_id = (modality_info.som_id, modality_info.eom_id)
        pad_id = modality_info.pad_id
        token_id_start = modality_info.token_id_start
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        if model_input_name == 'inputs_embeds':
            inputs_tensor = inputs_tensor[0]
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        kwargs_has_attention_mask = model_kwargs.get('attention_mask', None) is not None
        generation_config.bos_token_id = som_id
        generation_config.eos_token_id = eom_id
        generation_config.pad_token_id = pad_id
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        if not self.config.is_encoder_decoder and model_input_name == 'inputs_embeds':
            generation_config.use_cache = True
        accepts_attention_mask = 'attention_mask' in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = 'encoder_outputs' not in model_kwargs
        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config, model_kwargs)
        elif kwargs_has_attention_mask:
            if model_input_name == 'input_ids' and len(model_kwargs['attention_mask'].shape) > 2:
                raise ValueError('`attention_mask` passed to `generate` must be 2D.')
        input_ids = inputs_tensor if model_input_name == 'input_ids' else model_kwargs.pop('input_ids')
        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get('min_length') is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(generation_config=generation_config, has_default_max_length=has_default_max_length, has_default_min_length=has_default_min_length, model_input_name=model_input_name, inputs_tensor=inputs_tensor, input_ids_length=input_ids_length)
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        model_kwargs.setdefault('past_key_values', None)
        logits_processor = LogitsProcessorList()
        prepared_logits_processor_mod = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor, device=inputs_tensor.device, model_kwargs=model_kwargs)
        stopping_criteria = StoppingCriteriaList()
        prepared_stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer)
        model_kwargs['use_cache'] = generation_config.use_cache
        input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
        max_length = generation_config.max_length
        output_attentions = generation_config.output_attentions
        do_sample = generation_config.do_sample
        eos_token_id = eom_id
        pad_token_id = pad_id
        has_eos_stopping_criteria = True
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_logits = generation_config.output_logits
        output_hidden_states = generation_config.output_hidden_states
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        decoder_hidden_states = [[]] * self.modality_num
        raw_logits = () if return_dict_in_generate and output_logits else None
        if model_kwargs.get('type_ids') is None:
            model_kwargs['type_ids'] = torch.zeros_like(input_ids)
        all_type_ids = model_kwargs['type_ids'] if input_ids.shape[-1] > 0 else input_ids.clone().detach()
        som_generated = torch.tensor([False] * batch_size, device=device)
        eom_generated = torch.tensor([False] * batch_size, device=device)
        current_mod_from = cur_len
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if this_peer_finished:
                break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({'output_attentions': output_attentions} if output_attentions else {})
            model_inputs.update({'output_hidden_states': output_hidden_states} if output_hidden_states else {})
            model_inputs.update({'forward_mod': self.current_mod_id})
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[self.current_mod_id][:, -1, :].clone().float().to(device)
            next_token_scores = prepared_logits_processor_mod(input_ids[:, current_mod_from:], next_token_logits)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            next_type_ids = torch.ones(batch_size, device=device, dtype=torch.long) * self.current_mod_id
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                next_type_ids = next_type_ids * unfinished_sequences + self.text_id * (1 - unfinished_sequences)
            all_type_ids = torch.cat([all_type_ids, next_type_ids[:, None]], -1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~prepared_stopping_criteria(input_ids, ())
            this_peer_finished = unfinished_sequences.max() == 0
            if 'type_ids' in model_kwargs:
                past_types = model_kwargs.pop('type_ids')
                if model_kwargs.get('use_cache'):
                    model_kwargs['type_ids'] = next_type_ids[:, None]
                else:
                    model_kwargs['type_ids'] = torch.cat((past_types, next_type_ids[:, None]), -1)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
            cur_len += 1
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_hidden_states:
                    for i in range(self.modality_num):
                        decoder_hidden_states[i].append(outputs.hidden_states[-1][i][:, -1:, :])
            del outputs
        for i, modality_info in enumerate(self.modality_infos):
            input_ids[all_type_ids == i] += modality_info.token_id_start
        input_ids = input_ids[:, input_ids_length:]
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(sequences=input_ids, logits=raw_logits, decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
            else:
                return GenerateDecoderOnlyOutput(sequences=input_ids, hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
        else:
            return input_ids

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
        """Prepare cached generation inputs for the modality-wise embedding list."""
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                past_length = int(past_key_values.get_seq_length() or 0)
            elif len(past_key_values):
                past_length = int(past_key_values[0][0].shape[-2])
        model_inputs = {'past_key_values': past_key_values, 'attention_mask': attention_mask}
        if inputs_embeds is not None and past_length == 0:
            model_inputs['input_ids'] = None
            model_inputs['inputs_embeds'] = inputs_embeds
        else:
            model_inputs['input_ids'] = input_ids[:, -1:].contiguous()
            model_inputs['inputs_embeds'] = None
        if attention_mask is not None and kwargs.get('position_ids') is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs['position_ids'] = position_ids
        for key, value in kwargs.items():
            if key in {'type_ids', 'position_ids', 'token_type_ids'}:
                if value is not None and past_length > 0:
                    value = value[:, -1:].contiguous()
            model_inputs[key] = value
        model_inputs.pop('labels', None)
        return model_inputs

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        if 'inputs_embeds' in model_kwargs and (not self.config.is_encoder_decoder):
            cache_position = torch.ones_like(model_kwargs['inputs_embeds'][0][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        elif 'decoder_inputs_embeds' in model_kwargs and self.config.is_encoder_decoder:
            cache_position = torch.ones_like(model_kwargs['decoder_inputs_embeds'][0][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        past_length = 0
        if model_kwargs.get('past_key_values') is not None:
            cache = model_kwargs['past_key_values']
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, 'get_seq_length') and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]
        model_kwargs['cache_position'] = cache_position
        return model_kwargs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder: bool=False, num_new_tokens: int=1):
        model_kwargs['past_key_values'] = outputs.past_key_values
        if getattr(outputs, 'state', None) is not None:
            model_kwargs['state'] = outputs.state
        if not is_encoder_decoder:
            if 'attention_mask' in model_kwargs:
                attention_mask = model_kwargs['attention_mask']
                model_kwargs['attention_mask'] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        elif 'decoder_attention_mask' in model_kwargs:
            decoder_attention_mask = model_kwargs['decoder_attention_mask']
            model_kwargs['decoder_attention_mask'] = torch.cat([decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))], dim=-1)
        if model_kwargs.get('use_cache', True):
            model_kwargs['cache_position'] = model_kwargs['cache_position'][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop('cache_position')
            new_positions = torch.arange(past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype).to(past_positions.device)
            model_kwargs['cache_position'] = torch.cat((past_positions, new_positions))
        if 'position_ids' in model_kwargs:
            past_positions = model_kwargs.pop('position_ids')
            last_pos_ids = past_positions[:, -1]
            if model_kwargs.get('use_cache'):
                model_kwargs['position_ids'] = last_pos_ids[:, None] + num_new_tokens
            else:
                new_positions = (torch.arange(num_new_tokens) + last_pos_ids + 1).to(past_positions.device)
                model_kwargs['position_ids'] = torch.cat((past_positions, new_positions), -1)
        return model_kwargs
