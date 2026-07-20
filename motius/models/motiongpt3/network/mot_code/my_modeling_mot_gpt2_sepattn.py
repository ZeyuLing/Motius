
import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable


from transformers.models.gpt2.modeling_gpt2 import Conv1D, GPT2MLP#, eager_attention_forward
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    # CausalLMOutputWithCrossAttentions,
    # QuestionAnsweringModelOutput,
    # SequenceClassifierOutputWithPast,
    # TokenClassifierOutput,
)
from transformers import (
    GPT2LMHeadModel, 
    GPT2Model,
    AutoTokenizer
)

from .mot_module import MoTDiffFuncMod, MoTLayerNorm, apply_residual

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


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    # scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        # scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

    
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

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.type_ids = None  # torch.LongTensor=None

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = False
        self.scale_attn_by_inverse_layer_idx = False
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        self.modality_num = modality_num
        modality_channels = [self.embed_dim] * modality_num
        self.modality_channels = modality_channels
        # self.shared_attn = shared_attn
        self.shared_attn = layer_idx in config.cross_model_attention
        self.text_shared_attn = layer_idx in config.text_cross_model_attention
        print(layer_idx, self.shared_attn)
        
        # TODO: support GQA in the future
        # c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_out_dims = [self.embed_dim*3, self.embed_dims[1]*3]
        if self.shared_attn:
            attn_out_dim = self.attn_out_dims[0]
            c_attns = [Conv1D(self.attn_out_dims[0], embed_dim) for embed_dim in self.embed_dims]
            c_projs = [Conv1D(embed_dim, self.embed_dim) for embed_dim in self.embed_dims]
        else:
            attn_out_dim = None
            c_attns = [Conv1D(3*embed_dim, embed_dim) for embed_dim in self.embed_dims]
            c_projs = [Conv1D(embed_dim, embed_dim) for embed_dim in self.embed_dims]
        self.c_attn = MoTDiffFuncMod(c_attns, modality_num, out_dims=self.attn_out_dims, out_dim=attn_out_dim)
        self.c_proj = MoTDiffFuncMod(c_projs, modality_num, out_dims=self.embed_dims)
        # self.c_attn = MoTMod(c_attn, modality_num, out_dim=self.attn_out_dim)
        # self.c_proj = MoTMod(c_proj, modality_num)

        # TODO: add dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # TODO: add gate ffn layers
        self.ln_1 = MoTLayerNorm([nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon) for embed_dim in self.embed_dims], self.modality_num)
        self.ln_2 = MoTLayerNorm([nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon) for embed_dim in self.embed_dims], self.modality_num)
        # self.ln_1 = MoTLayerNorm(nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon), self.modality_num)
        # self.ln_2 = MoTLayerNorm(nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon), self.modality_num)
        # # self.ln_1 = nn.ModuleList([nn.LayerNorm(mch, eps=config.layer_norm_epsilon) for mch in modality_channels])
        # # self.ln_2 = nn.ModuleList([nn.LayerNorm(mch, eps=config.layer_norm_epsilon) for mch in modality_channels])
        
        # mlp use config for: hidden_size, activation_function, resid_pdrop
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        inner_dims = [inner_dim, int(inner_dim*self.mot_factor)]
        # self.mlp = nn.ModuleList(
        #     [
        #         GPT2MLP(inner_dim, config) for mch in modality_channels
        #     ]
        # )
        # hidden_size, activation_function, resid_pdrop
        # if layer_idx == config.num_hidden_layers-1:
        #     mlps = [MoTGPT2MLP(indim, hidden_size=self.embed_dim, config=config) 
        #                     for i,indim in enumerate(inner_dims)]
        # else:
        # out_dim = None if layer_idx == config.num_hidden_layers-1 else self.embed_dim
        out_dim=None
        mlps = [MoTGPT2MLP(indim, hidden_size=self.embed_dims[i], config=config, out_size=out_dim) 
                        for i,indim in enumerate(inner_dims)]
        self.mlp = MoTDiffFuncMod(mlps, modality_num, out_dims=self.embed_dims)

    def update_typeids(self, type_ids):
        self.valid_pos = type_ids
        # self.type_ids = type_ids
        # self.mlp.update_typeids(type_ids)
        # self.c_attn.update_typeids(type_ids)
        # self.c_proj.update_typeids(type_ids)
        self.ln_1.update_typeids(type_ids)
        self.ln_2.update_typeids(type_ids)

    # def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
    #     if self.type_ids is None:
    #         self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
    #     type_ids = self.type_ids
        
    #     # print(type_ids)
    #     assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
    #     # exit()
    #     # FIXME: support different channel size
    #     hidden_states_ = torch.zeros_like(hidden_states)
    #     for type_id, _ in enumerate(self.modality_channels):
    #         if torch.any(type_ids == type_id).cpu().tolist():
    #             hidden_states_[type_ids == type_id] = module[type_id](hidden_states[type_ids == type_id])
            
    #     return hidden_states_
    
    def apply_qkv_proj(self, hidden_states: torch.Tensor, module: torch.Tensor):
        if self.type_ids is None:
            self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        type_ids = self.type_ids
        
        # FIXME: support different channel size
        hidden_states_ = torch.zeros_like(hidden_states).repeat(1, 1, 3)
        for type_id, _ in enumerate(self.modality_channels):
            if torch.any(type_ids == type_id).cpu().tolist():
                hidden_states_[type_ids == type_id] = module[type_id](hidden_states[type_ids == type_id])
            
        return hidden_states_

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # print(attention_mask.shape)
        # print(attn_weights.shape, query.shape, key.shape, value.shape)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
    def forward_attn_shared(
        self, 
        hidden_states: torch.Tensor,
        # type_ids: torch.Tensor=None,
        layer_past: List[torch.Tensor]=None,    # FIXME: do not support kv-cache here
        use_cache: bool=False,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states[0].size()
        all_states = torch.zeros((*(self.valid_pos[0].shape), self.c_attn.out_dim), device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        for i, mod_val_pos in enumerate(self.valid_pos):
            # print(f'mod {i}: hidden: {hidden_states[i].shape}')
            all_states[mod_val_pos] = self.c_attn.fn[i](hidden_states[i][mod_val_pos])
        query_states, key_states, value_states = all_states.split(self.split_size, dim=2)
        
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        
        # for kv cache support
        # print('layer_past',layer_past is not None)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            pask_key_values = (key_states, value_states)
        else:
            pask_key_values = None
        
        # print(query_states.shape, key_states.shape, value_states.shape)
        # attn_1 eager
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, 
                                            attention_mask=attention_mask, head_mask=None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], -1)
        # attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        
        
        final_attn_output = []
        for i, mod_val_pos in enumerate(self.valid_pos):
            hid = torch.zeros_like(hidden_states[i])
            hid[mod_val_pos] = self.c_proj.fn[i](attn_output[mod_val_pos])
            hid = self.resid_dropout(hid)
            final_attn_output.append(hid)

        return (final_attn_output, pask_key_values)
    
    def forward_attn_alone(self, 
            hidden_state: torch.Tensor,
            mod_id: int,
            # type_ids: torch.Tensor=None,
            layer_past: List[torch.Tensor]=None,    # FIXME: do not support kv-cache here
            use_cache: bool=False,
            attention_mask: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        bsz, q_len, _ = hidden_state.size()
        # mod_val_pos = self.valid_pos[mod_id]
        # all_states = torch.zeros((*(mod_val_pos.shape), self.c_attn.out_dim), device=hidden_state.device, dtype=hidden_state.dtype)
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
        
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, 
                                            attention_mask=attention_mask, head_mask=None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], -1)
        
        attn_output = self.c_proj.fn[mod_id](attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return (attn_output, pask_key_values)
    
    def forward_attn(self, 
            hidden_states: torch.Tensor,
            # type_ids: torch.Tensor=None,
            layer_past: List[torch.Tensor]=None,    # FIXME: do not support kv-cache here
            use_cache: bool=False,
            attention_mask: Optional[torch.FloatTensor] = None,
            **kwargs
        ):
        bsz, q_len, _ = hidden_states.size()
        # query_states, key_states, value_states = self.apply_qkv_proj(hidden_states, self.c_attn).split(self.split_size, dim=2)
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        # bs x src x nh x dim
        # # query = self._split_heads(query, self.num_heads, self.head_dim)
        #     new_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        #     tensor = tensor.view(new_shape)
        #     return tensor.permute(0, 2, 1, 3)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        
        # for kv cache support
        # print('layer_past',layer_past is not None)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            pask_key_values = (key_states, value_states)
        else:
            pask_key_values = None
        
        # print(query_states.shape, key_states.shape, value_states.shape)
        # attn_1 eager
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, 
                                               attention_mask=attention_mask, head_mask=None)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], -1)
        # TODO: kv-repeat for GQA
        
        # TODO: rope position ids for Qwen, LlaMA

        # # attn_2 sdpa
        # # FIXME: hard code
        # attention_interface: Callable = sdpa_attention_forward
        # # torch.nn.functional.scaled_dot_product_attention
        # bsz, q_len, _ = hidden_states.size()
        # is_cross_attention = False
        # attention_mask = None
        # is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False
        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=self.attn_dropout.p if self.training else 0.0,
        #     is_causal=is_causal,     # FIXME: hard code, use causal attention
        #     **kwargs,
        # )
        # attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()

        # # attn_3 sdpa
        # query_states = query_states.contiguous()
        # key_states = key_states.contiguous()
        # value_states = value_states.contiguous()
        # causal_mask = attention_mask
        # if attention_mask is not None:
        #     causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        # causal_mask = None
        # is_causal = True if causal_mask is None and q_len > 1 else False
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=None,
        #     dropout_p=self.attn_dropout.p if self.training else 0.0,
        #     is_causal=True,
        # )
        # attn_output = attn_output.transpose(1, 2).contiguous()  # for torch.nn.functional.scaled_dot_product_attention only
        # attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()

        # # _merge_heads(self, tensor, num_heads, self.head_dim):
        #     tensor = tensor.permute(0, 2, 1, 3).contiguous()
        #     new_shape = tensor.size()[:-2] + (num_heads * self.head_dim,)
        #     return tensor.view(new_shape)
        
        # attn_output = self.apply_module(attn_output, self.c_proj)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return (attn_output, pask_key_values)

    # def apply_residual(self, hidden_states, residual):
    #     hidden_states = [residual[i] + hid for i, hid in enumerate(hidden_states)]
    #     return hidden_states

    def forward(self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            # type_ids: Optional[torch.LongTensor]=None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            **kwargs,
        ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        # hidden_states: bs x length x ch
        # type_ids: [0, 0, 1, 1, 0, 0, 2, 2], bs x length
        # if type_ids is None:
        #     pass
        
        residual = hidden_states
        layernorm_output = self.ln_1(hidden_states)
        # layernorm_output = [
        #         self.ln_1.fn[i](hid)
        #         for i, hid in enumerate(hidden_states)
        #     ]
        if self.shared_attn:
            attn_output, pask_key_values = self.forward_attn_shared(
                layernorm_output, 
                layer_past = layer_past, 
                use_cache=use_cache,
                attention_mask=attention_mask,
            #     head_mask=head_mask,
            #     output_attentions=output_attentions,
            )
            # if not self.text_shared_attn:
            #     lpast = None
            #     mod_id = 0  # self.text_id
            #     if layer_past is not None:
            #         lpast = layer_past[mod_id]
            #     attn_, pask_kv = self.forward_attn_alone(
            #         layernorm_output[mod_id],
            #         mod_id=mod_id,
            #         layer_past = lpast, 
            #         use_cache=use_cache,
            #         attention_mask=attention_mask,
            #     #     head_mask=head_mask,
            #     #     output_attentions=output_attentions,
            #     )
            #     attn_output[0] = attn_
            #     pask_key_values[0] = pask_kv
        else:
            attn_output = []
            pask_key_values = []
            for mod_id in range(self.modality_num):
                lpast = None
                if layer_past is not None:
                    lpast = layer_past[mod_id]
                attn_, pask_kv = self.forward_attn_alone(
                    layernorm_output[mod_id],
                    mod_id=mod_id,
                    layer_past = lpast, 
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                #     head_mask=head_mask,
                #     output_attentions=output_attentions,
                )
                attn_output.append(attn_)
                pask_key_values.append(pask_kv)
            
        hidden_states = apply_residual(attn_output, residual)

        residual = hidden_states
        layernorm_output = self.ln_2(hidden_states)
        # layernorm_output = [
        #         self.ln_2.fn[i](hid)
        #         for i, hid in enumerate(hidden_states)
        #     ]
        # feed_forward_hidden_states = self.mlp(feed_forward_hidden_states = )
        feed_forward_hidden_states = [self.mlp.fn[i](hid) for i, hid in enumerate(layernorm_output)]
        # residual connection
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
        
        # self.mot_factor = 1.0
        # self.mot_embed_dim = self.embed_dim*self.mot_factor
        # self.embed_dims = [config.hidden_size, self.mot_embed_dim]
        
        # self.embed_dims = config.embed_dims
        # self.mot_factor = config.mot_factor
        # self.mot_embed_dim = config.mot_embed_dim
        
        self.modality_num = modality_num
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        # self.wpe_mot = nn.Embedding(config.max_position_embeddings, self.mot_embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        # self.drop = MoTDrop(config.embd_pdrop)
        self.h = nn.ModuleList([MoTGPT2Block(config, layer_idx=i, modality_num=modality_num) for i in range(config.num_hidden_layers)])
        # exit()
        # FIXME: should set to MoT mode too
        self.ln_f = MoTLayerNorm([nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon) for embed_dim in self.embed_dims], self.modality_num)
        # self.ln_module = nn.ModuleList([
        #     nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon) for i in range(self.modality_num)])

    # def ln_f(self, hid):
    #     self.apply_module(hid, self.ln_module)
    
    def update_typeids(self, type_ids):
        self.valid_pos = type_ids
        # self.type_ids = type_ids
        for n in self.h:
            n.update_typeids(type_ids)
        self.ln_f.update_typeids(type_ids)
        # self.wte.update_typeids(type_ids)
        
    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        if self.type_ids is None:
            self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        type_ids = self.type_ids
        
        assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
        # FIXME: support different channel size
        hidden_states_ = torch.zeros_like(hidden_states)
        for type_id in range(self.modality_num):
            if torch.any(type_ids == type_id).cpu().tolist():
                hidden_states_[type_ids == type_id] = module[type_id](hidden_states[type_ids == type_id])
            
        return hidden_states_

    def forward(
        self,
        # *args,
        # **kwargs,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            raise ValueError(f"You have to specify inputs_embeds")
        # for n in self.h:
        #     n.type_ids = self.type_ids
        # print(kwargs.keys())
        # return super().forward(*args, **kwargs)
        
        input_shape = inputs_embeds[0].size()[:-1]
        batch_size = inputs_embeds[0].shape[0]
        device = inputs_embeds[0].device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            # print('cross_model_attention', self.config.cross_model_attention)
            # print(len(past_key_values)) 24
            # print(len(past_key_values[0])) 2
            # print(len(past_key_values[0][0])) 2
            if 0 in self.config.cross_model_attention:
                past_length = past_key_values[0][0].size(-2)  # past_key_values[0][0][0]: [1, 16, 7, 64])
            else:
                past_length = past_key_values[0][0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # # if inputs_embeds is None:
        # #     inputs_embeds = self.wte(input_ids)
        # position_embeds = self.wpe(position_ids)
        # hidden_states = [emb + position_embeds for emb in inputs_embeds]
        # hidden_states = inputs_embeds + position_embeds

        position_embeds = self.wpe(position_ids)
        # position_embeds_mot = [position_embeds, self.wpe_mot(position_ids)]
        # hidden_states = [in_emb + pos_emb for in_emb, pos_emb in zip(inputs_embeds, position_embeds_mot)]
        hidden_states = [in_emb + position_embeds for in_emb in inputs_embeds]

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds[0],
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=hidden_states[0].dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states[0].dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = [hid + token_type_embeds for hid in hidden_states]

        hidden_states = [self.drop(hid) for hid in hidden_states]
        # hidden_states = self.drop(hidden_states)
    
        output_shapes = [(-1,) + input_shape[1:] + (hid.size(-1),) for hid in hidden_states]
        # output_shape = (-1,) + input_shape[1:] + (hidden_states[0].size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        # hidden_states = [self.ln_f.fn[i](hid) for i, hid in enumerate(hidden_states)]
        hidden_states = [hid.view(outshape) for hid,outshape in zip(hidden_states, output_shapes)]

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
