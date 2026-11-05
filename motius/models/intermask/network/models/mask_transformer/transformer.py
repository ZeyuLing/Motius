import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
from clip.model import LayerNorm as CLIPLayerNorm
from clip.model import Transformer as CLIPTransformer
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from .transformer_modules import *
from torch.distributions.categorical import Categorical
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class SpaTempPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SpaTempPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = PositionalEncoding2D(d_model)

    def forward(self, x):
        seqlen, bs, input_feats = x.shape
        x1, sep, x2  = x.split([seqlen//2, 1, seqlen//2])

        def add_positional_encoding(x):
            x = x.permute(1,0,2) # [seqen, bs, input_feats] -> [bs, seqen, input_feats]
            x = x.reshape(x.shape[0], 5, x.shape[1]//5, x.shape[2])

            x = x + self.positional_encoding(x)

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            x = x.permute(1,0,2)

            return x

        x1 = add_positional_encoding(x1)
        x2 = add_positional_encoding(x2)

        x = torch.cat([x1, sep, x2], dim=0)
        return self.dropout(x)

class OutputProcess_adaLN(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu

        self.LayerNorm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_mod = AdaLNModulation(latent_dim, nchunks=2)

        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)

        shift, scale = self.adaLN_mod(cond)
        hidden_states = modulate(self.LayerNorm(hidden_states), shift, scale)

        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.nbp = 5
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)

        self.position_enc = SpaTempPositionalEncoding(self.latent_dim, self.dropout)

        self.Transformer = InterMTransformer(d_model=self.latent_dim,
                                        nhead=num_heads,
                                        dim_feedforward=ff_size,
                                        dropout=dropout,
                                        num_layers=num_layers,
                                        nbp=self.nbp)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 3  # three dummy tokens, one for masking, one for padding, one for separating
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1
        self.sep_id = opt.num_tokens + 2

        self.output_process = OutputProcess_adaLN(out_feats=opt.num_tokens, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.initialize_weights()

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.load_and_freeze_clip(
                clip_version,
                load_pretrained=kargs.get('load_clip_weights', True),
            )


    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def initialize_weights(self):
        def __init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    module.bias.data.zero_()
                if module.weight is not None:
                    module.weight.data.fill_(1.0)

        self.apply(__init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.Transformer.blocks:
            nn.init.constant_(block.adaLN_mod_combined.model[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod_combined.model[-1].bias, 0)
            nn.init.constant_(block.adaLN_mod_split.model[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod_split.model[-1].bias, 0)


        # nn.init.normal_(self.position_enc.pe, mean=0.0, std=0.02)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].weight, 0)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].bias, 0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_')]

    def load_and_freeze_clip(self, clip_version, load_pretrained=True):

        ##From InterGen
        if load_pretrained:
            clip_model, _ = clip.load(clip_version, device="cpu", jit=False)
            self.clip_token_embedding = clip_model.token_embedding
            self.clip_transformer = clip_model.transformer
            self.clip_positional_embedding = clip_model.positional_embedding
            self.clip_ln_final = clip_model.ln_final
            self.clip_dtype = clip_model.dtype
        else:
            attention_mask = torch.empty(77, 77).fill_(float("-inf")).triu_(1)
            self.clip_token_embedding = nn.Embedding(49408, 768)
            self.clip_transformer = CLIPTransformer(
                width=768,
                layers=12,
                heads=12,
                attn_mask=attention_mask,
            )
            self.clip_positional_embedding = nn.Parameter(torch.empty(77, 768), requires_grad=False)
            self.clip_ln_final = CLIPLayerNorm(768)
            self.clip_dtype = torch.float32

        for p in self.clip_transformer.parameters():
            p.requires_grad = False
        for p in self.clip_token_embedding.parameters():
            p.requires_grad = False
        for p in self.clip_ln_final.parameters():
            p.requires_grad = False

        clipTransLayer = nn.TransformerEncoderLayer(d_model=768,
                                                    nhead=8,
                                                    dim_feedforward=2048,
                                                    dropout=0.1,
                                                    activation="gelu",
                                                    batch_first=True)
        self.clipTrans = nn.TransformerEncoder(clipTransLayer, num_layers=2)
        self.clipln = nn.LayerNorm(768)

    def encode_text(self, tokens):
        device = next(self.parameters()).device

        # From InterGen
        with torch.no_grad():
            text = tokens.to(device)
            x = self.clip_token_embedding(text).type(self.clip_dtype)
            pe_tokens = x + self.clip_positional_embedding.type(self.clip_dtype)
            x = pe_tokens.permute(1,0,2)
            x = self.clip_transformer(x)
            x = x.permute(1,0,2)
            clip_out = self.clip_ln_final(x).type(self.clip_dtype)

        out = self.clipTrans(clip_out)
        out = self.clipln(out)
        feat_clip_text = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''
        bs, n_tokens = motion_ids.shape

        cond = self.mask_cond(cond, force_mask=force_mask)

        motion_ids = torch.cat((motion_ids[:, :n_tokens//2],
                                self.sep_id * torch.ones((bs, 1), device=motion_ids.device, dtype=torch.long),
                                motion_ids[:, n_tokens//2:]), dim=-1)

        x = self.token_emb(motion_ids)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)
        cond = self.cond_emb(cond) #(1, b, latent_dim)

        x = self.position_enc(x)

        padding_mask = torch.cat([padding_mask[:, :n_tokens//2],
                                torch.zeros_like(padding_mask[:, 0:1]),
                                padding_mask[:, n_tokens//2:]], dim=1) #(b, seqlen+1)

        output = self.Transformer(x, cond, src_key_padding_mask=padding_mask) #(seqlen, b, e)
        logits = self.output_process(output, cond) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits








    def forward(self, motion_ids, cond, padding_mask, force_mask=False):
        return self.trans_forward(motion_ids, cond, padding_mask, force_mask)
