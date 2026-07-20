import math
import os
import torch
from torch import nn
from einops import rearrange
from transformers import (
    GPT2LMHeadModel, 
    GPT2Model,
    AutoTokenizer
)
from typing import Optional, Tuple, List, Union, Callable

from transformers.utils import logging
logger = logging.get_logger(__name__)

import inspect
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from transformers.cache_utils import StaticCache, Cache
from transformers.generation.configuration_utils import GenerationConfig
# from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import is_torchdynamo_compiling

from .my_modeling_mot_gpt2_sepattn import MoTGPT2Model
from .modality_utils_sepattn import get_modalities_infos
from .mot_module import get_embeds_from_ids


import random
import numpy as np
def seed_setting(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MoTGPT2LMHeadModel(GPT2LMHeadModel):

    def __init__(self, config, modality_num=2, motion_codebook_size=512+4, 
                 mot_factor=1.0, attention_mode='all'):
        super().__init__(config)
        self.modality_num = modality_num
        self.forward_mod = 0  # 'text'
        self.config.d_model = config.n_embd
        self.config.motion_vocab_size = motion_codebook_size
        self.config.mot_loss = 'skip'
        # self.config.text_vocab_size = 50257
        self.modality_infos = None
        self.last_pos_ids = None
        config = self.config
        
        self.mot_factor = mot_factor
        # self.mot_embed_dim = int(config.hidden_size*self.mot_factor)
        self.mot_embed_dim = int(config.hidden_size//config.num_attention_heads*self.mot_factor)*config.num_attention_heads
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
            config.cross_model_attention = all_layers_idx[:config.num_hidden_layers//2]
        elif attention_mode == 'halfB':
            config.cross_model_attention = all_layers_idx[-config.num_hidden_layers//2:]
        elif attention_mode == 'firstthird':
            config.cross_model_attention = all_layers_idx[:config.num_hidden_layers//3]
        elif attention_mode == 'midthird':
            config.cross_model_attention = all_layers_idx[config.num_hidden_layers//3:int(config.num_hidden_layers/3*2)]
        elif attention_mode == 'lastthird':
            config.cross_model_attention = all_layers_idx[int(config.num_hidden_layers/3*2):]
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
        # self.modlist = self.modality_infos.keys()
        self.mod2id = {m.modality_name:i for i,m in enumerate(modality_infos)}
        self.text_id = self.mod2id['text']
        modality_infos[self.text_id].pre_processor = self.transformer.wte
        modality_infos[self.text_id].post_processor = self.lm_head
        self.modality_infos = modality_infos
        self.pad_ids = [m.pad_id for m in modality_infos]

        pre_processors = ([m.pre_processor for m in self.modality_infos])
        post_processors = ([m.post_processor for m in self.modality_infos])
        self.pre_processors = nn.ModuleList(pre_processors)
        self.post_processors = nn.ModuleList(post_processors)

    def update_typeids(self, type_ids):
        self.valid_pos = type_ids
        # self.type_ids = type_ids
        # for n in self.layers:
        #     n.update_typeids(type_ids)
        self.transformer.update_typeids(type_ids)
        # self.post_processors.update_typeids(type_ids)
        
    def init_mod_token_embeddings(self, mod_id, added_num_tokens=None):
        if added_num_tokens is None:
            added_num_tokens = self.modality_infos[mod_id].mod_voc_size
        old_embeddings = self.pre_processors[0]
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = self.pre_processors[mod_id]
        self._init_added_embeddings_weights_with_mean(
                    old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
                )
        
        old_lm_head = self.post_processors[1]
        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size() 
        new_lm_head = self.post_processors[mod_id]
        self._init_added_lm_head_weights_with_mean(
                    old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed=False
                )
    
    def get_embeddings_from_ids(self, input_ids, type_ids):
        inputs_embeds = []
        for i in range(self.modality_num):
            mod_valid_pos = (type_ids == i)
            mot_input_id = input_ids.masked_fill(~mod_valid_pos,self.pad_ids[i])
            mod_inputs_embeds = self.pre_processors[i](mot_input_id)
            inputs_embeds.append(mod_inputs_embeds)
        return inputs_embeds

    def forward(
        self, 
        type_ids: torch.Tensor = None,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_mod: np.ndarray[int] = None,
        mot_labels=None,
    ):
        self.type_ids = type_ids
        valid_pos = torch.stack([type_ids==i for i in range(self.modality_num)], dim=0).to(type_ids.device)
        self.update_typeids(valid_pos)
        
        self.transformer.position_ids = position_ids
        
        bs = self.valid_pos[0].shape[0]
        if forward_mod is None:
            # forward_mod = np.array([self.forward_mod]*bs)
            forward_mod = self.forward_mod

        if inputs_embeds is None:
            inputs_embeds = get_embeds_from_ids(input_ids, self.valid_pos, self.pad_ids, self.pre_processors)
            input_ids = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states_mods = transformer_outputs[0]
        lm_logits = [post_processor(hid) for post_processor, hid in zip(self.post_processors, hidden_states_mods)]

        total_loss = None
        if labels is not None:
            total_loss = 0.
            if not isinstance(labels, list):
                mlabels = [labels]*self.modality_num
            assert len(mlabels) == len(lm_logits), 'labels not match modalities'
            for i, out_logit in enumerate(lm_logits):
                # if out_logit is None: continue
                if self.valid_pos[i].sum() == 0: continue
                mlabel = mlabels[i].to(out_logit.device)
                loss_fct = self.modality_infos[i].loss_fct
                loss_mod = loss_fct(out_logit, mlabel, self.valid_pos[i])
                
                total_loss = total_loss + (loss_mod).nan_to_num_(0)
        
        out_logits = lm_logits
        
        if not return_dict:
            output = (out_logits, ) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=total_loss,
            # logits=pred,
            logits=out_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    def generate(self, 
                 inputs: Optional[torch.Tensor] = None, 
                 generation_config: Optional[GenerationConfig] = None,
                #  logits_processor: Optional[LogitsProcessorList] = None, 
                #  stopping_criteria: Optional[StoppingCriteriaList] = None,
                 prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                 synced_gpus: Optional[bool] = None,
                #  assistant_model: Optional["PreTrainedModel"] = None,
                #  streamer: Optional["BaseStreamer"] = None,
                #  negative_prompt_ids: Optional[torch.Tensor] = None,
                #  negative_prompt_attention_mask: Optional[torch.Tensor] = None,
                 mode:Union[str, List[str]] = 'text', # 默认生成文本
                #  mode:Union[int, List[int]] = 0, # 默认生成文本
                 **kwargs,
                #  type_ids=type_ids1, position_ids=position_ids, do_sample=True, use_cache=False, max_new_tokens=40
                 ):
        tokenizer = kwargs.pop("tokenizer", None)
        
        self.current_mode = mode
        self.current_mod_id = self.mod2id[mode]
        modality_info = self.modality_infos[self.current_mod_id]
        som_id, eom_id = modality_info.som_id, modality_info.eom_id
        pad_id = modality_info.pad_id
        token_id_start = modality_info.token_id_start
        # pre_processor, post_processor = modality_info.pre_processor, modality_info.post_processor
        # mod_vocab = modality_info.mod_voc_size

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        if model_input_name == 'inputs_embeds':
            inputs_tensor = inputs_tensor[0]
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        generation_config.bos_token_id = som_id
        generation_config.eos_token_id = eom_id
        generation_config.pad_token_id = pad_id
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # 4. Define other model kwargs
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")


        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        # if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        #     model_kwargs["num_logits_to_keep"] = 1
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # This custom GPT2 implementation uses the legacy tuple cache returned
        # by its own attention blocks, not Transformers' DynamicCache.
        model_kwargs.setdefault("past_key_values", None)

        # # 9. prepare logits processors and stopping criteria
        # # logits_processor
        logits_processor = LogitsProcessorList()
        # # <transformers.generation.logits_process.TopKLogitsWarper
        # # generation_config.top_k: 50
        prepared_logits_processor_mod = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            # negative_prompt_ids=negative_prompt_ids,
            # negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # stopping_criteria
        stopping_criteria = StoppingCriteriaList()
        # <transformers.generation.stopping_criteria.MaxLengthCriteria
        # <transformers.generation.stopping_criteria.EosTokenCriteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
        )

        # # 8. determine generation mode
        # generation_mode = generation_config.get_generation_mode(assistant_model)

        model_kwargs["use_cache"] = generation_config.use_cache

        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        max_length = generation_config.max_length
        output_attentions = generation_config.output_attentions
        do_sample = generation_config.do_sample
        # eos_token_id = generation_config._eos_token_tensor
        # pad_token_id = generation_config._pad_token_tensor
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

        # # self.last_pos_ids
        # position2last = model_kwargs['position_ids'] if 'position_ids' in model_kwargs else model_kwargs['cache_position']
        # if kwargs.get('type_ids') is not None:
        #     type_ids = kwargs['type_ids']
        #     for i in self.mod2id.values():
        #         if (type_ids==i).sum() == 0: continue
        #         self.last_pos_ids[:,i] = (position2last*(type_ids==i)).max(-1).values.long()
        # else:
        #     self.last_pos_ids[:,0] = position2last.clone().max(-1).values.long()
            
        decoder_hidden_states = [[]]*self.modality_num
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        # for i in range(max_length):
        
        if model_kwargs.get('type_ids') is None:
            model_kwargs['type_ids'] = torch.zeros_like(input_ids)
        all_type_ids = model_kwargs['type_ids'] if input_ids.shape[-1] > 0 else input_ids.clone().detach()
        som_generated = torch.tensor([False]*batch_size, device=device)
        # som_generated = torch.tensor([som_id in in_id for in_id in input_ids*(all_type_ids==1)], device=device)
        eom_generated = torch.tensor([False]*batch_size, device=device)
        current_mod_from = cur_len
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # print(i, input_ids.shape)
            if this_peer_finished: break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # print(model_inputs.keys())
            # print(i, model_inputs['type_ids'])
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            model_inputs.update({'forward_mod': self.current_mod_id})

            outputs = self(**model_inputs, return_dict=True)
            # if text_num:
            next_token_logits = outputs.logits[self.current_mod_id][:, -1, :].clone().float().to(device)
            # print(input_ids.max(), current_mod_from, input_ids[:,current_mod_from:].shape, next_token_logits.shape)
            next_token_scores = prepared_logits_processor_mod(input_ids[:,current_mod_from:], next_token_logits)
            # print(self.current_mod_id, som_id, eom_id, som_generated.shape, next_token_scores.shape)
            # next_token_scores[:, som_id][som_generated] = torch.tensor(float("-inf"), device=input_ids.device)
            # next_token_scores[:, eom_id][~som_generated] = torch.tensor(float("-inf"), device=input_ids.device)
            # # probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # # token_next = torch.multinomial(probs, num_samples=1)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            next_type_ids = torch.ones(batch_size, device=device, dtype=torch.long)*self.current_mod_id

            # finished sentencces should have their next token be a padding token
            if has_eos_stopping_criteria:  # False
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                next_type_ids = next_type_ids * unfinished_sequences + self.text_id* (1 - unfinished_sequences)

            all_type_ids = torch.cat([all_type_ids, next_type_ids[:,None]], -1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # input_ids = torch.cat([input_ids, token_next], -1)
            # new_embeds = pre_processor(next_tokens[:, None])
            # if 'inputs_embeds' in model_kwargs and model_kwargs['use_cache']:
            #     model_kwargs['inputs_embeds'] = new_embeds
            #     # model_kwargs['input_ids'] = None
            # elif self.current_mod_id != 0:
            #     if model_kwargs['use_cache']:
            #         model_kwargs['inputs_embeds'] = new_embeds
            #     else:
            #         inputs_embeds = self.transformer.wte(model_kwargs[['input_ids']])
            #         model_kwargs['inputs_embeds'] = torch.cat((inputs_embeds,new_embeds), dim=1)
                    
            # unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)
            # som_generated = som_generated | (next_tokens==som_id)
            # eom_generated = eom_generated | (som_generated & (next_tokens==eom_id))
            # unfinished_sequences = unfinished_sequences & ~eom_generated
            unfinished_sequences = unfinished_sequences & ~prepared_stopping_criteria(input_ids, ())
            this_peer_finished = unfinished_sequences.max() == 0
            
            if "type_ids" in  model_kwargs:
                past_types = model_kwargs.pop("type_ids")
                if model_kwargs.get('use_cache'):
                    model_kwargs["type_ids"] = next_type_ids[:, None]
                else:
                    model_kwargs["type_ids"] = torch.cat((past_types, next_type_ids[:, None]), -1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            cur_len += 1
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_hidden_states:
                    for i in range(self.modality_num):
                        decoder_hidden_states[i].append(outputs.hidden_states[-1][i][:,-1:,:])

            del outputs

        # all_type_ids == torch.cat([all_type_ids,torch.full((batch_size,1), self.current_mod_id)])
        for i, modality_info in enumerate(self.modality_infos):
            input_ids[all_type_ids==i] += modality_info.token_id_start
        input_ids = input_ids[:,input_ids_length:]
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    # scores=scores,
                    logits=raw_logits,
                    # encoder_attentions=encoder_attentions,
                    # encoder_hidden_states=encoder_hidden_states,
                    # decoder_attentions=decoder_attentions,
                    # cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    # scores=scores,
                    # logits=raw_logits,
                    # attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        """Prepare cached generation inputs for the modality-wise embedding list."""

        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = int(past_key_values.get_seq_length() or 0)
            elif len(past_key_values):
                past_length = int(past_key_values[0][0].shape[-2])

        model_inputs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
        if inputs_embeds is not None and past_length == 0:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids[:, -1:].contiguous()
            model_inputs["inputs_embeds"] = None

        if attention_mask is not None and kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids
        for key, value in kwargs.items():
            if key in {"type_ids", "position_ids", "token_type_ids"}:
                if value is not None and past_length > 0:
                    value = value[:, -1:].contiguous()
            model_inputs[key] = value
        model_inputs.pop("labels", None)
        return model_inputs

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs
        
    def _update_model_kwargs_for_generation(
        self,
        outputs,  #: ModelOutput,
        model_kwargs,  #: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        model_kwargs["past_key_values"] = outputs.past_key_values
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        # if 'type_ids' not in model_kwargs and hasattr(self, 'type_ids') and self.type_ids is not None:
        #     model_kwargs['type_ids'] = self.type_ids
        # if "type_ids" in  model_kwargs:
        #     past_types = model_kwargs.pop("type_ids")
        #     if model_kwargs.get('use_cache'):
        #         model_kwargs["type_ids"] = torch.full((*(past_types.shape[:-1]),1), self.current_mod_id, 
        #                                               dtype=torch.long, device=self.device)
        #     else:
        #         new_types = torch.full((*(past_types.shape[:-1]),num_new_tokens), self.current_mod_id, 
        #                                device=self.device, dtype=torch.long)
        #         model_kwargs["type_ids"] = torch.cat((past_types, new_types), -1)

        # # self.last_pos_ids
        # # mid = self.mod2id[self.current_mode]
        # mid = self.current_mod_id
        # # last_pos_ids = self.last_pos_ids[...,mid:mid+1].clone().detach()
        # last_pos_ids = self.last_pos_ids[...,mid]#.clone().detach()
        
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        if "position_ids" in  model_kwargs:
            # past_positions = self.transformer.position_ids
            past_positions = model_kwargs.pop("position_ids")
            last_pos_ids = past_positions[:,-1]
            if model_kwargs.get('use_cache'):
                model_kwargs["position_ids"] = last_pos_ids[:,None]+num_new_tokens
            else:
                new_positions = (torch.arange(num_new_tokens)+last_pos_ids+1).to(past_positions.device)
                model_kwargs["position_ids"] = torch.cat((past_positions, new_positions), -1)

        # self.last_pos_ids[...,mid] = self.last_pos_ids[...,mid] + num_new_tokens
        return model_kwargs


if __name__ == "__main__":

    model_config = "deps/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPT2LMHeadModel.from_pretrained(model_config).eval()
    model.transformer._attn_implementation = "sdpa"

    model2 = MoTGPT2LMHeadModel(model.config).eval()
    model2.generation_config = model.generation_config

    state_dict = model.state_dict()
    new_state_dict = dict()
    for name, param in state_dict.items():
        name = name.replace("attn.c_attn", "c_attn.fn.0")
        name = name.replace("attn.c_proj", "c_proj.fn.0")
        name = name.replace("mlp", "mlp.fn.0")
        name = name.replace("ln_f", "ln_f.fn.0")
        name = name.replace("ln_1", "ln_1.fn.0")
        name = name.replace("ln_2", "ln_2.fn.0")
        
        new_state_dict[name] = param

        # new_state_dict[name.replace('fn.0','fn.1')] = param

    # torch.save(new_state_dict, 'deps/mot-gpt2-medium/model_state_dict.pth')
    msg = model2.load_state_dict(new_state_dict, strict=False)

    tokenizer.add_tokens(
                [f'<motion_id_{i}>' for i in range(512)])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_motion>', '<end_of_motion>', '<masked_motion>', '<pad_motion>']})
    model2.config.mot_lm_dim = model2.config.motion_vocab_size
    model2.set_modality_info(tokenizer)

    # print('missing keys:', msg.missing_keys)
    print('====' * 10)
    print('unexpected_keys keys:', msg.unexpected_keys)

    inputs = tokenizer(["How are you today"], return_tensors="pt")
    # [2437,  389,  345, 1909]
    # type_ids = torch.zeros_like(inputs.input_ids)
    # position_ids = torch.arange(inputs.input_ids.shape[-1]).unsqueeze(0).long()

    _pre = True
    if _pre:
        # inputs1 = tokenizer([" A A B How are you today "], return_tensors="pt")
        inputs1 = tokenizer([" A A AHow are you today"], return_tensors="pt")
        # [317,  317, 317, 2437,  389,  345, 1909])
    else:
        inputs1 = tokenizer(["How are you today A A A"], return_tensors="pt")
    
    # inputs2 = tokenizer(["translate: Hello"], return_tensors="pt")
    # type_ids = torch.zeros_like(inputs2.input_ids)
    # position_ids = torch.arange(inputs2.input_ids.shape[-1]).unsqueeze(0).long()


    generation_config = dict(
        max_new_tokens=20,
        do_sample=True,
        # use_cache=False,
        # top_k=1,
        # top_p=0.0001,
        num_return_sequences=1,
        # use_cache=False,
        pad_token_id=50256,
        # no_repeat_ngram_size=4,
    )

    input_ids = inputs1.input_ids
    inputs_embeds = model2.transformer.wte(inputs1.input_ids)
    type_ids = torch.zeros_like(inputs1.input_ids)
    position_ids = torch.arange(inputs1.input_ids.shape[-1]).unsqueeze(0).long()

    type_ids1 = type_ids.clone().detach()
    attention_mask1=inputs1.attention_mask.clone().detach()
    position_ids1 = position_ids.clone().detach()
    if _pre:
        type_ids1[...,:3] = 1
        attention_mask1[...,:3]=0
        position_ids1[...,3:] =  position_ids1[...,3:]- 3
    else:
        type_ids1[...,-4:-1] = 1
        attention_mask1[...,-4:-1]=0
        position_ids1[...,-4:-1] =  position_ids1[...,-4:-1]- position_ids1[...,-3]
        position_ids1[...,-1] = position_ids1[...,-5] +1


    torch.manual_seed(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs1 = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, 
                            **generation_config)
    print('inputs', tokenizer.batch_decode(outputs1, skip_special_tokens=True))

    torch.manual_seed(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs1 = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=inputs1.attention_mask, 
                            **generation_config)
    print('inputs1', tokenizer.batch_decode(outputs1, skip_special_tokens=True))

    seed_setting(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs1 = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask1, 
                            #   do_sample=False, use_cache=True,  )
                            **generation_config)
    print('attention_mask1', tokenizer.batch_decode(outputs1, skip_special_tokens=True))

    torch.manual_seed(0)
    # input_ids=input_ids
    outputs2 = model2.generate(input_ids=input_ids, attention_mask=inputs1.attention_mask,
                            type_ids=type_ids, # position_ids=position_ids, 
                            **generation_config)
    print('mine inputs1', tokenizer.batch_decode(outputs2, skip_special_tokens=True))

    torch.manual_seed(0)
    seed_setting(0)
    outputs2 = model2.generate(input_ids=input_ids, attention_mask=attention_mask1, 
                            type_ids=type_ids, # position_ids=position_ids, 
                            # do_sample=False, use_cache=True, )
                            **generation_config)
    print('mine attention_mask1', tokenizer.batch_decode(outputs2, skip_special_tokens=True))
    

        # attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #     attention_mask=attention_mask,
        #     input_shape=(batch_size, input_shape[-1]),
        #     inputs_embeds=inputs_embeds,
        #     past_key_values_length=past_length,
        # )
