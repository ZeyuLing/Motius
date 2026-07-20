
import copy
import torch
from torch import nn
import torch.nn.functional as F
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def apply_residual(hidden_states, residual):
    hidden_states = [residual[i] + hid for i, hid in enumerate(hidden_states)]
    return hidden_states

def get_embeds_from_ids(input_ids, valid_pos, pad_ids, pre_processors):
    inputs_embeds = []
    for i, mod_valid_pos in enumerate(valid_pos):
        mot_input_id = input_ids.clone()
        mot_input_id = mot_input_id.masked_fill_(~mod_valid_pos, pad_ids[i])
        mod_inputs_embeds = pre_processors[i](mot_input_id)
        inputs_embeds.append(mod_inputs_embeds)
    return inputs_embeds

class MoTBase(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def update_typeids(self, type_ids):
        if type_ids is not None:
            self.valid_pos = type_ids
            # self.type_ids = type_ids

    def forward(self, hidden_states):
        hidden_states = self.apply_module(hidden_states, self.fn)
        # hidden_states = self.norm_fn[0](hidden_states)
        return hidden_states
    
    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        # if self.type_ids is None:
        #     self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        # type_ids = self.type_ids
        # assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
        if self.valid_pos is None:
            type_ids = self.type_ids if self.type_ids is not None else torch.zeros_like(hidden_states[..., 0])
            self.valid_pos = [type_ids==i for i in range(self.modality_num)]
        valid_pos = self.valid_pos
        # self.type_ids = None
        # assert mod_valid_pos[0].shape == hidden_states.shape[:2], f'valid_pos shape not match: {mod_valid_pos[0].shape}, {hidden_states.shape}'

        # FIXME: support different channel size
        # # hidden_states_ = None
        # if self.out_dim is None:
        #     hidden_states_ = torch.zeros_like(hidden_states).to(hidden_states)
        # else:
        #     hidden_states_ = torch.zeros((*mod_valid_pos[0].shape, self.out_dim), 
        #                                  device=hidden_states.device,
        #                                  dtype=hidden_states.dtype)
        
        out_shape = (*(valid_pos[0].shape), hidden_states.shape[-1]
                     ) if self.out_dim is None else (
                         *(valid_pos[0].shape), self.out_dim)
        hidden_states_ = torch.zeros(out_shape).to(hidden_states)
        # for type_id in range(self.modality_num):
        #     out = module[type_id](hidden_states[type_ids==type_id])
        #     hidden_states_[type_ids==type_id] = out
        for type_id, mod_valid_pos in enumerate(valid_pos):
            # if mod_valid_pos.any():
            out = module[type_id](hidden_states[mod_valid_pos])
            # print(hidden_states.shape, hidden_states[mod_valid_pos].shape, mod_valid_pos.shape)
            # print(out.shape, hidden_states_.shape)
            hidden_states_[mod_valid_pos] = out
        return hidden_states_


class MoTLayerNorm(MoTBase):
    def __init__(self, norm_fn, modality_num=2, out_dim=None):
        super().__init__()
        if isinstance(norm_fn,list):
            self.fn = nn.ModuleList(norm_fn)
        else:
            self.fn = _get_clones(norm_fn, modality_num)
        self.modality_num = modality_num
        # self.out_dim = out_dim
        # self.out_dims = [out_dim]*modality_num
        self.type_ids = None
        self.valid_pos = None

    def forward(self, hidden_states):
        out_hiddens = []
        # for i, hid in enumerate(hidden_states):
        for type_id, mod_valid_pos in enumerate(self.valid_pos):
            hid = hidden_states[type_id]
            fake_hidden = torch.zeros_like(hid).to(hid)
            if mod_valid_pos.sum() > 0:
                fake_hidden[mod_valid_pos] = self.fn[type_id](hid[mod_valid_pos])
            out_hiddens.append(fake_hidden)
            
        return out_hiddens
    
class MoTDrop(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def forward(self, hiddens):
        # final_attn_output = []
        # for i, mod_val_pos in enumerate(valid_pos):
        #     hid = torch.zeros_like(hidden_states[i])
        #     hid[mod_val_pos] = self.c_proj.fn[i](attn_output[mod_val_pos])
        #     hid = self.resid_dropout(hid)
        #     final_attn_output.append(hid)
        return [F.dropout(hid, self.p, self.training, self.inplace) for hid in hiddens]
    

class MoTMod(MoTBase):
    def __init__(self, norm_fn, modality_num=2, out_dim=None):
        super().__init__()
        self.fn = _get_clones(norm_fn, modality_num)
        self.modality_num = modality_num
        if isinstance(norm_fn, (nn.Linear, nn.Embedding)):
            out_dim = norm_fn.out_features
        self.out_dim = out_dim
        # self.out_dims = [out_dim]*modality_num
        self.type_ids = None
        self.valid_pos = None

    # def update_typeids(self, type_ids):
    #     self.type_ids = type_ids

    # def forward(self, hidden_states):
    #     hidden_states = self.apply_module(hidden_states, self.fn)
    #     # hidden_states = self.norm_fn[0](hidden_states)
    #     return hidden_states
    
    # def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
    #     if self.type_ids is None:
    #         self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
    #     type_ids = self.type_ids
    #     # self.type_ids = None

    #     assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
    #     # FIXME: support different channel size
    #     # hidden_states_ = None
    #     if self.out_dim is None:
    #         hidden_states_ = torch.zeros_like(hidden_states)
    #     else:
    #         hidden_states_ = torch.zeros((*hidden_states.shape[:-1], self.out_dim))
    #     for type_id in range(self.modality_num):
    #         if torch.any(type_ids == type_id).cpu().tolist():
    #             out = module[type_id](hidden_states[type_ids == type_id])
    #             # if hidden_states_ is None:
    #             #     hidden_states_ = torch.zeros((*hidden_states.shape[:-1], out.shape[-1]))
    #             hidden_states_[type_ids == type_id] = out
            
    #     return hidden_states_

class MoTEmbed(MoTBase):
    def __init__(self, fn_list, modality_num=2, out_dims=None):
        super().__init__()
        self.fn = nn.ModuleList(fn_list)
        self.modality_num = modality_num
        if out_dims is None:
            out_dims = []
            for fn in fn_list:
                if isinstance(fn, (nn.Linear)):
                    out_dims.append(fn.out_features)
                elif isinstance(fn, (nn.Embedding)):
                    out_dims.append(fn.embedding_dim)
                elif hasattr(fn, 'output_dim'):
                    out_dims.append(fn.output_dim)
                else:
                    raise ValueError('out_dims is None with not Linear or Embedding term in fn_list')
                
        assert (len(out_dims) == modality_num) and (len(fn_list) == modality_num)
        self.out_dims = out_dims
        self.out_dim = max(out_dims)
        self.type_ids = None
        self.valid_pos = None
        
    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        # if self.type_ids is None:
        #     self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        # type_ids = self.type_ids
        # assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
        if self.valid_pos is None:
            type_ids = self.type_ids if self.type_ids is not None else torch.zeros_like(hidden_states[..., 0])
            self.valid_pos = [type_ids==i for i in range(self.modality_num)]
        valid_pos = self.valid_pos
        # self.type_ids = None
        assert valid_pos[0].shape == hidden_states.shape[:2], f'valid_pos shape not match: {mod_valid_pos[0].shape}, {hidden_states.shape}'

        # FIXME: support different channel size
        # hidden_states_ = None
        out_shape = (*(valid_pos[0].shape), hidden_states.shape[-1]
                     ) if self.out_dim is None else (
                         *(valid_pos[0].shape), self.out_dim)
        hidden_states_ = torch.zeros(out_shape).to(module[0].weight)

        # for type_id in range(self.modality_num):
        #     out = module[type_id](hidden_states[type_ids==type_id])
        #     hidden_states_[type_ids==type_id] = out
        for type_id, mod_valid_pos in enumerate(valid_pos):
            if mod_valid_pos.sum():
                out = module[type_id](hidden_states[mod_valid_pos])
                hidden_states_[mod_valid_pos] = out
        return hidden_states_

class MoTDiffFuncMod(MoTBase):
    def __init__(self, fn_list, modality_num=2, out_dims=None,out_dim=None):
        super().__init__()
        assert modality_num == len(fn_list), 'mismatch in modality_num and fn_list'
        self.fn = nn.ModuleList(fn_list)
        self.modality_num = modality_num
        # if out_dims is None:
        #     out_dims = []
        #     for fn in fn_list:
        #         if isinstance(fn, (nn.Linear)):
        #             out_dims.append(fn.out_features)
        #         elif isinstance(fn, (nn.Embedding)):
        #             out_dims.append(fn.embedding_dim)
        #         elif hasattr(fn, 'output_dim'):
        #             out_dims.append(fn.output_dim)
        #         else:
        #             raise ValueError('out_dims is None with not Linear or Embedding term in fn_list')

        # assert (len(out_dims) == modality_num) and (len(fn_list) == modality_num)
        self.out_dims = out_dims
        if out_dim is None:
            out_dim = max(out_dims)
        self.out_dim = out_dim
        self.type_ids = None
        self.valid_pos = None

    def update_typeids(self, type_ids):
        if type_ids is not None:
            if isinstance(type_ids, torch.Tensor):
                type_ids = [type_ids==i for i in range(self.modality_num)]
            self.valid_pos = type_ids
            # self.type_ids = type_ids

    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        # if self.type_ids is None:
        #     self.type_ids = torch.zeros_like(hidden_states[..., 0]).long()
        # type_ids = self.type_ids
        # assert type_ids.shape == hidden_states.shape[:-1], f'type_ids shape not match: {type_ids.shape}, {hidden_states.shape}'
        # dtype = hidden_states.dtype
        # device = hidden_states.device
        # min_dtype = torch.finfo(dtype).min
        # self.type_ids = None
        if self.valid_pos is None:
            type_ids = self.type_ids if self.type_ids is not None else torch.zeros_like(hidden_states[..., 0])
            self.valid_pos = [type_ids==i for i in range(self.modality_num)]
        valid_pos = self.valid_pos
        # self.type_ids = None
        # assert valid_pos[0].shape == hidden_states.shape[:-1], f'valid_pos shape not match: {valid_pos[0].shape}, {hidden_states.shape}'

        # FIXME: support different channel size
        hidden_states_ = []
        # hidden_states_ = torch.full((*hidden_states.shape[:-1], self.out_dim), min_dtype, 
        #                             device=device, dtype=dtype, requires_grad=True)
        for type_id, mod_valid_pos in enumerate(valid_pos):
            if mod_valid_pos.any():
                out = module[type_id](hidden_states[mod_valid_pos])
                # hidden_states_[type_ids == type_id][...,:odim] = out
                hidden_states_.append(out)
            else:
                hidden_states_.append(None)
            
        return hidden_states_
    
    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()


# from typing import Any, Optional, Tuple, TypedDict
# from transformers.modeling_outputs import ModelOutput
# @dataclass
# class Seq2ModOutput(ModelOutput):

#     loss: Optional[torch.FloatTensor] = None
#     # logits: torch.FloatTensor = None
#     preds: torch.FloatTensor = None
#     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
#     decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
#     cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
#     encoder_last_hidden_state: Optional[torch.FloatTensor] = None
#     encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
#     encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
