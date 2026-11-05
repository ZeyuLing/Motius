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

    def forward(self, hidden_states):
        hidden_states = self.apply_module(hidden_states, self.fn)
        return hidden_states

    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        if self.valid_pos is None:
            type_ids = self.type_ids if self.type_ids is not None else torch.zeros_like(hidden_states[..., 0])
            self.valid_pos = [type_ids == i for i in range(self.modality_num)]
        valid_pos = self.valid_pos
        out_shape = (*valid_pos[0].shape, hidden_states.shape[-1]) if self.out_dim is None else (*valid_pos[0].shape, self.out_dim)
        hidden_states_ = torch.zeros(out_shape).to(hidden_states)
        for type_id, mod_valid_pos in enumerate(valid_pos):
            out = module[type_id](hidden_states[mod_valid_pos])
            hidden_states_[mod_valid_pos] = out
        return hidden_states_

class MoTLayerNorm(MoTBase):

    def __init__(self, norm_fn, modality_num=2, out_dim=None):
        super().__init__()
        if isinstance(norm_fn, list):
            self.fn = nn.ModuleList(norm_fn)
        else:
            self.fn = _get_clones(norm_fn, modality_num)
        self.modality_num = modality_num
        self.type_ids = None
        self.valid_pos = None

    def forward(self, hidden_states):
        out_hiddens = []
        for type_id, mod_valid_pos in enumerate(self.valid_pos):
            hid = hidden_states[type_id]
            fake_hidden = torch.zeros_like(hid).to(hid)
            if mod_valid_pos.sum() > 0:
                fake_hidden[mod_valid_pos] = self.fn[type_id](hid[mod_valid_pos])
            out_hiddens.append(fake_hidden)
        return out_hiddens

class MoTDiffFuncMod(MoTBase):

    def __init__(self, fn_list, modality_num=2, out_dims=None, out_dim=None):
        super().__init__()
        assert modality_num == len(fn_list), 'mismatch in modality_num and fn_list'
        self.fn = nn.ModuleList(fn_list)
        self.modality_num = modality_num
        self.out_dims = out_dims
        if out_dim is None:
            out_dim = max(out_dims)
        self.out_dim = out_dim
        self.type_ids = None
        self.valid_pos = None

    def update_typeids(self, type_ids):
        if type_ids is not None:
            if isinstance(type_ids, torch.Tensor):
                type_ids = [type_ids == i for i in range(self.modality_num)]
            self.valid_pos = type_ids

    def apply_module(self, hidden_states: torch.Tensor, module: torch.Tensor):
        if self.valid_pos is None:
            type_ids = self.type_ids if self.type_ids is not None else torch.zeros_like(hidden_states[..., 0])
            self.valid_pos = [type_ids == i for i in range(self.modality_num)]
        valid_pos = self.valid_pos
        hidden_states_ = []
        for type_id, mod_valid_pos in enumerate(valid_pos):
            if mod_valid_pos.any():
                out = module[type_id](hidden_states[mod_valid_pos])
                hidden_states_.append(out)
            else:
                hidden_states_.append(None)
        return hidden_states_
