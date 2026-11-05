import math
import os
import torch
from torch import nn
from einops import rearrange
from transformers import GPT2LMHeadModel, GPT2Model, AutoTokenizer
from typing import Optional, Tuple, List, Union, Callable
from transformers.utils import logging
logger = logging.get_logger(__name__)
import inspect
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.cache_utils import StaticCache, Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import is_torchdynamo_compiling
from .transformer import MoTGPT2Model
from .modalities import get_modalities_infos
from .modality_layers import get_embeds_from_ids
import random
import numpy as np

class MoTGPT2LMHeadModel(GPT2LMHeadModel):

    def __init__(self, config, modality_num=2, motion_codebook_size=512 + 4, mot_factor=1.0, attention_mode='all'):
        super().__init__(config)
        self.modality_num = modality_num
        self.forward_mod = 0
        self.config.d_model = config.n_embd
        self.config.motion_vocab_size = motion_codebook_size
        self.config.mot_loss = 'skip'
        self.modality_infos = None
        self.last_pos_ids = None
        config = self.config
        self.mot_factor = mot_factor
        self.mot_embed_dim = int(config.hidden_size // config.num_attention_heads * self.mot_factor) * config.num_attention_heads
        config.mot_embed_dim = self.mot_embed_dim
        config.embed_dims = [config.hidden_size, self.mot_embed_dim]
        config.mot_factor = self.mot_factor
        all_layers_idx = list(range(config.num_hidden_layers))
        config.text_cross_model_attention = all_layers_idx[-1:]
        if attention_mode == 'all':
            config.cross_model_attention = all_layers_idx
        elif attention_mode == 'first':
            config.cross_model_attention = all_layers_idx[:1]
        elif attention_mode == 'last':
            config.cross_model_attention = all_layers_idx[-1:]
        elif attention_mode == 'Ahalf':
            config.cross_model_attention = all_layers_idx[:config.num_hidden_layers // 2]
        elif attention_mode == 'halfB':
            config.cross_model_attention = all_layers_idx[-config.num_hidden_layers // 2:]
        elif attention_mode == 'firstthird':
            config.cross_model_attention = all_layers_idx[:config.num_hidden_layers // 3]
        elif attention_mode == 'midthird':
            config.cross_model_attention = all_layers_idx[config.num_hidden_layers // 3:int(config.num_hidden_layers / 3 * 2)]
        elif attention_mode == 'lastthird':
            config.cross_model_attention = all_layers_idx[int(config.num_hidden_layers / 3 * 2):]
        elif attention_mode == 'odd':
            config.cross_model_attention = all_layers_idx[::2]
        elif attention_mode == 'even':
            config.cross_model_attention = all_layers_idx[1::2]
        elif attention_mode.startswith('last'):
            lnum = int(attention_mode.split('last')[-1])
            config.cross_model_attention = all_layers_idx[-lnum:]
        elif attention_mode.startswith('first'):
            lnum = int(attention_mode.split('first')[-1])
            config.cross_model_attention = all_layers_idx[:lnum]
        elif attention_mode == 'None':
            config.cross_model_attention = []
        else:
            assert False, f'not recognized attention_mode {attention_mode}'
        self.transformer = MoTGPT2Model(config, modality_num=modality_num)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def set_modality_info(self, tokenizer):
        modality_infos = get_modalities_infos(config=self.config, tokenizer=tokenizer)
        self.mod2id = {m.modality_name: i for i, m in enumerate(modality_infos)}
        self.text_id = self.mod2id['text']
        modality_infos[self.text_id].pre_processor = self.transformer.wte
        modality_infos[self.text_id].post_processor = self.lm_head
        self.modality_infos = modality_infos
        self.pad_ids = [m.pad_id for m in modality_infos]
        pre_processors = [m.pre_processor for m in self.modality_infos]
        post_processors = [m.post_processor for m in self.modality_infos]
        self.pre_processors = nn.ModuleList(pre_processors)
        self.post_processors = nn.ModuleList(post_processors)

    def update_typeids(self, type_ids):
        self.valid_pos = type_ids
        self.transformer.update_typeids(type_ids)

    def get_embeddings_from_ids(self, input_ids, type_ids):
        inputs_embeds = []
        for i in range(self.modality_num):
            mod_valid_pos = type_ids == i
            mot_input_id = input_ids.masked_fill(~mod_valid_pos, self.pad_ids[i])
            mod_inputs_embeds = self.pre_processors[i](mot_input_id)
            inputs_embeds.append(mod_inputs_embeds)
        return inputs_embeds

    def forward(self, type_ids: torch.Tensor=None, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, token_type_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, forward_mod: np.ndarray[int]=None, mot_labels=None):
        self.type_ids = type_ids
        valid_pos = torch.stack([type_ids == i for i in range(self.modality_num)], dim=0).to(type_ids.device)
        self.update_typeids(valid_pos)
        self.transformer.position_ids = position_ids
        bs = self.valid_pos[0].shape[0]
        if forward_mod is None:
            forward_mod = self.forward_mod
        if inputs_embeds is None:
            inputs_embeds = get_embeds_from_ids(input_ids, self.valid_pos, self.pad_ids, self.pre_processors)
            input_ids = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states_mods = transformer_outputs[0]
        lm_logits = [post_processor(hid) for post_processor, hid in zip(self.post_processors, hidden_states_mods)]
        total_loss = None
        out_logits = lm_logits
        if not return_dict:
            output = (out_logits,) + transformer_outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=total_loss, logits=out_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions, cross_attentions=transformer_outputs.cross_attentions)
