from transformers.utils import logging
logger=logging.get_logger(__name__)
import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable
from transformers.models.gpt2.modeling_gpt2 import Conv1D, GPT2MLP
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers import GPT2LMHeadModel, GPT2Model, AutoTokenizer
from .modality_layers import MoTDiffFuncMod, MoTLayerNorm, apply_residual

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def sdpa_attention_forward(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Optional[torch.Tensor], dropout: float=0.0, is_causal: Optional[bool]=None, **kwargs) -> Tuple[torch.Tensor, None]:
    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key.shape[-2]]
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=causal_mask, dropout_p=dropout, is_causal=is_causal)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, None)
from transformers.activations import ACT2FN

class MoTGPT2MLP(nn.Module):

    def __init__(self, intermediate_size, hidden_size, config, out_size=None):
        super().__init__()
        embed_dim = hidden_size
        if out_size is None:
            out_size = embed_dim
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(out_size, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MoTGPT2Block(nn.Module):

    def __init__(self, config, layer_idx=None, modality_num=2, shared_attn=True):
        super().__init__()
        self.config = config
        self.embed_dims = config.embed_dims
        self.split_sizes = self.embed_dims
        self.mot_factor = config.mot_factor
        self.model_parallel = False
        self.device_map = None
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.type_ids = None
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = False
        self.scale_attn_by_inverse_layer_idx = False
        max_positions = config.max_position_embeddings
        self.register_buffer('bias', torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions), persistent=False)
        self.modality_num = modality_num
        modality_channels = [self.embed_dim] * modality_num
        self.modality_channels = modality_channels
        self.shared_attn = layer_idx in config.cross_model_attention
        self.text_shared_attn = layer_idx in config.text_cross_model_attention
        self.attn_out_dims = [self.embed_dim * 3, self.embed_dims[1] * 3]
        if self.shared_attn:
            attn_out_dim = self.attn_out_dims[0]
            c_attns = [Conv1D(self.attn_out_dims[0], embed_dim) for embed_dim in self.embed_dims]
            c_projs = [Conv1D(embed_dim, self.embed_dim) for embed_dim in self.embed_dims]
        else:
            attn_out_dim = None
            c_attns = [Conv1D(3 * embed_dim, embed_dim) for embed_dim in self.embed_dims]
            c_projs = [Conv1D(embed_dim, embed_dim) for embed_dim in self.embed_dims]
        self.c_attn = MoTDiffFuncMod(c_attns, modality_num, out_dims=self.attn_out_dims, out_dim=attn_out_dim)
        self.c_proj = MoTDiffFuncMod(c_projs, modality_num, out_dims=self.embed_dims)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.ln_1 = MoTLayerNorm([nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon) for embed_dim in self.embed_dims], self.modality_num)
        self.ln_2 = MoTLayerNorm([nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon) for embed_dim in self.embed_dims], self.modality_num)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        inner_dims = [inner_dim, int(inner_dim * self.mot_factor)]
        out_dim = None
        mlps = [MoTGPT2MLP(indim, hidden_size=self.embed_dims[i], config=config, out_size=out_dim) for i, indim in enumerate(inner_dims)]
        self.mlp = MoTDiffFuncMod(mlps, modality_num, out_dims=self.embed_dims)

    def update_typeids(self, type_ids):
        self.valid_pos = type_ids
        self.ln_1.update_typeids(type_ids)
        self.ln_2.update_typeids(type_ids)

    def apply_qkv_proj(self, hidden_states: torch.Tensor, module: torch.Tensor):
        if self.type_ids is None:
            self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        type_ids = self.type_ids
        hidden_states_ = torch.zeros_like(hidden_states).repeat(1, 1, 3)
        for type_id, _ in enumerate(self.modality_channels):
            if torch.any(type_ids == type_id).cpu().tolist():
                hidden_states_[type_ids == type_id] = module[type_id](hidden_states[type_ids == type_id])
        return hidden_states_

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if not self.is_cross_attention:
            query_length, key_length = (query.size(-2), key.size(-2))
            causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def forward_attn_shared(self, hidden_states: torch.Tensor, layer_past: List[torch.Tensor]=None, use_cache: bool=False, attention_mask: Optional[torch.FloatTensor]=None, **kwargs):
        bsz, q_len, _ = hidden_states[0].size()
        all_states = torch.zeros((*self.valid_pos[0].shape, self.c_attn.out_dim), device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        for i, mod_val_pos in enumerate(self.valid_pos):
            all_states[mod_val_pos] = self.c_attn.fn[i](hidden_states[i][mod_val_pos])
        query_states, key_states, value_states = all_states.split(self.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        if use_cache is True:
            pask_key_values = (key_states, value_states)
        else:
            pask_key_values = None
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask=attention_mask, head_mask=None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], -1)
        final_attn_output = []
        for i, mod_val_pos in enumerate(self.valid_pos):
            hid = torch.zeros_like(hidden_states[i])
            hid[mod_val_pos] = self.c_proj.fn[i](attn_output[mod_val_pos])
            hid = self.resid_dropout(hid)
            final_attn_output.append(hid)
        return (final_attn_output, pask_key_values)

    def forward_attn_alone(self, hidden_state: torch.Tensor, mod_id: int, layer_past: List[torch.Tensor]=None, use_cache: bool=False, attention_mask: Optional[torch.FloatTensor]=None, **kwargs):
        bsz, q_len, _ = hidden_state.size()
        all_states = self.c_attn.fn[mod_id](hidden_state)
        query_states, key_states, value_states = all_states.split(self.split_sizes[mod_id], dim=2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        if use_cache is True:
            pask_key_values = (key_states, value_states)
        else:
            pask_key_values = None
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask=attention_mask, head_mask=None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], -1)
        attn_output = self.c_proj.fn[mod_id](attn_output)
        attn_output = self.resid_dropout(attn_output)
        return (attn_output, pask_key_values)

    def forward_attn(self, hidden_states: torch.Tensor, layer_past: List[torch.Tensor]=None, use_cache: bool=False, attention_mask: Optional[torch.FloatTensor]=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        if use_cache is True:
            pask_key_values = (key_states, value_states)
        else:
            pask_key_values = None
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask=attention_mask, head_mask=None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], -1)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return (attn_output, pask_key_values)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], layer_past: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, **kwargs) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        residual = hidden_states
        layernorm_output = self.ln_1(hidden_states)
        if self.shared_attn:
            attn_output, pask_key_values = self.forward_attn_shared(layernorm_output, layer_past=layer_past, use_cache=use_cache, attention_mask=attention_mask)
        else:
            attn_output = []
            pask_key_values = []
            for mod_id in range(self.modality_num):
                lpast = None
                if layer_past is not None:
                    lpast = layer_past[mod_id]
                attn_, pask_kv = self.forward_attn_alone(layernorm_output[mod_id], mod_id=mod_id, layer_past=lpast, use_cache=use_cache, attention_mask=attention_mask)
                attn_output.append(attn_)
                pask_key_values.append(pask_kv)
        hidden_states = apply_residual(attn_output, residual)
        residual = hidden_states
        layernorm_output = self.ln_2(hidden_states)
        feed_forward_hidden_states = [self.mlp.fn[i](hid) for i, hid in enumerate(layernorm_output)]
        hidden_states = apply_residual(feed_forward_hidden_states, residual)
        if use_cache:
            outputs = (hidden_states, pask_key_values)
        else:
            outputs = (hidden_states,)
        return outputs

class MoTGPT2Model(GPT2Model):

    def __init__(self, config, modality_num=2):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.config = config
        self.embed_dims = config.embed_dims
        self.mot_embed_dim = config.mot_embed_dim
        self.modality_num = modality_num
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MoTGPT2Block(config, layer_idx=i, modality_num=modality_num) for i in range(config.num_hidden_layers)])
        self.ln_f = MoTLayerNorm([nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon) for embed_dim in self.embed_dims], self.modality_num)

    def update_typeids(self, type_ids):
        self.valid_pos = type_ids
        for n in self.h:
            n.update_typeids(type_ids)
        self.ln_f.update_typeids(type_ids)

    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        if self.type_ids is None:
            self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        type_ids = self.type_ids
        assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
        hidden_states_ = torch.zeros_like(hidden_states)
        for type_id in range(self.modality_num):
            if torch.any(type_ids == type_id).cpu().tolist():
                hidden_states_[type_ids == type_id] = module[type_id](hidden_states[type_ids == type_id])
        return hidden_states_

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if inputs_embeds is None:
            raise ValueError(f'You have to specify inputs_embeds')
        input_shape = inputs_embeds[0].size()[:-1]
        batch_size = inputs_embeds[0].shape[0]
        device = inputs_embeds[0].device
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        elif 0 in self.config.cross_model_attention:
            past_length = past_key_values[0][0].size(-2)
        else:
            past_length = past_key_values[0][0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = [in_emb + position_embeds for in_emb in inputs_embeds]
        _use_sdpa = self._attn_implementation == 'sdpa' and output_attentions is False and (head_mask is None)
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == 'flash_attention_2':
            attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask=attention_mask, input_shape=(batch_size, input_shape[-1]), inputs_embeds=inputs_embeds[0], past_key_values_length=past_length)
        elif attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states[0].dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states[0].dtype).min
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1])
            elif not self._attn_implementation == 'flash_attention_2':
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = [hid + token_type_embeds for hid in hidden_states]
        hidden_states = [self.drop(hid) for hid in hidden_states]
        output_shapes = [(-1,) + input_shape[1:] + (hid.size(-1),) for hid in hidden_states]
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = (self.h[i], past_key_values[i])
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple((past_state.to(hidden_states.device) for past_state in layer_past))
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, None, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and 'cuda:' + str(k) != self.last_device:
                        hidden_states = hidden_states.to('cuda:' + str(k + 1))
        hidden_states = self.ln_f(hidden_states)
        hidden_states = [hid.view(outshape) for hid, outshape in zip(hidden_states, output_shapes)]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

    def _convert_head_mask_to_5d(self: torch.nn.Module, head_mask: torch.Tensor, num_hidden_layers: int) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if head_mask.dim() != 5:
            raise ValueError(f'head_mask.dim != 5, instead {head_mask.dim()}')
        return head_mask.to(dtype=self.dtype)

    def get_head_mask(self: torch.nn.Module, head_mask: Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask
