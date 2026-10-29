import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import clip
import math
from einops import rearrange, repeat
from functools import partial
from pathlib import Path

def _load_openai_clip_model(clip_version, device):
    clip_path = Path(str(clip_version)) if clip_version is not None else None
    if clip_path is not None and clip_path.is_file():
        if str(clip_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(str(clip_path), device='cpu')
        else:
            state = torch.load(str(clip_path), map_location='cpu')
        clip_model = clip.model.build_model(dict(state))
        if str(device) == 'cpu' and any((torch.is_floating_point(v) and v.dtype == torch.float32 for v in state.values())):
            clip_model = clip_model.float()
            clip_model.load_state_dict(state, strict=True)
        return clip_model
    clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
    if str(device) != 'cpu':
        clip.model.convert_weights(clip_model)
    return clip_model

class InputProcess(nn.Module):

    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe.clone(memory_format=torch.contiguous_format))

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, dropout=0.1, height=200, width=50):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 4 != 0:
            raise ValueError('Cannot use sin/cos positional encoding with odd dimension (got dim={:d})'.format(d_model))
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.permute(1, 2, 0)
        self.register_buffer('pe', pe.clone(memory_format=torch.contiguous_format))

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :x.shape[1], None, :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):

    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)
        output = output.permute(1, 2, 0)
        return output

class OutputProcess(nn.Module):

    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)
        output = output.permute(1, 2, 0)
        return output

class MaskTransformer2D(nn.Module):

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1, clip_version=None, opt=None, **kargs):
        super(MaskTransformer2D, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        '\n        Preparing Networks\n        '
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position2d_enc = PositionalEncoding2D(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        if opt.attnj:
            self.attnj = True
            seqTransEncoderLayer2 = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
            self.seqTransEncoder2 = nn.TransformerEncoder(seqTransEncoderLayer2, num_layers=num_layers)
        else:
            self.attnj = False
        if getattr(opt, 'attnt', False):
            self.attnt = True
            seqTransEncoderLayer3 = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
            self.seqTransEncoder3 = nn.TransformerEncoder(seqTransEncoderLayer3, num_layers=num_layers)
        else:
            self.attnt = False
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError('Unsupported condition mode!!!')
        _num_tokens = opt.num_tokens2d + 2
        self.mask_id = opt.num_tokens2d
        self.pad_id = opt.num_tokens2d + 1
        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens2d, latent_dim=self.latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)
        self.apply(self.__init_weights)
        '\n        Preparing frozen weights\n        '
        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    def load_and_freeze_token_emb(self, codebook):
        """
        :param codebook: (c, d)
        :return:
        """
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0))
        self.token_emb.requires_grad_(False)
        print('Token embedding initialized!')

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model = _load_openai_clip_model(clip_version, self.opt.device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, tokens):
        device = next(self.parameters()).device
        text = tokens.to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        """
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(motion_ids)
        bs, T, J, dim = x.shape
        x = self.input_process(x.reshape(bs, T * J, dim).permute(1, 0, 2))
        cond = self.cond_emb(cond).unsqueeze(0)
        x = torch.cat([cond.repeat(J, 1, 1)[None, ...], x.reshape(T, J, bs, x.shape[-1])], dim=0)
        x = self.position2d_enc(x)
        xseq = x.reshape((T + 1) * J, bs, x.shape[-1])
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        padding_mask = rearrange(padding_mask, 'b t j -> b (t j)')
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)
        if not self.attnj:
            output = output.reshape(T + 1, J, bs, -1)[1:].reshape(T * J, bs, -1)
        else:
            x2 = output.reshape(T + 1, J, bs, -1)
            x2 = self.position2d_enc(x2)
            x2 = rearrange(x2, 't j b d -> j (b t) d')
            padding_mask2 = padding_mask.reshape(bs, T + 1, J)
            padding_mask2[:, 0] = True
            padding_mask2 = rearrange(padding_mask2, 'b t j -> (b t) j')
            nonpad_ids = torch.where(~padding_mask2[:, 0])[0]
            output2 = self.seqTransEncoder2(x2[:, nonpad_ids])
            x2[:, nonpad_ids] = output2
            output = rearrange(x2.reshape(J, bs, T + 1, -1), 'j b t d -> t j b d')[1:].reshape(T * J, bs, -1)
        if self.attnt:
            padding_mask3 = padding_mask.reshape(bs, T + 1, J)
            padding_mask3 = rearrange(padding_mask3, 'b t j -> (b j) t')
            x3 = output.reshape(T, J, bs, -1)
            xseq3 = torch.cat([cond.repeat(J, 1, 1)[None, ...], x3], dim=0)
            xseq3 = self.position2d_enc(xseq3)
            xseq3 = rearrange(xseq3, 't j b d -> t (b j) d')
            output3 = self.seqTransEncoder3(xseq3, src_key_padding_mask=padding_mask3)
            output = output + rearrange(output3.reshape(T + 1, bs, J, -1)[1:], 't b j d -> (t j) b d')
        logits = self.output_process(output)
        return logits

    def forward(self, motion_ids, cond, padding_mask, force_mask=False):
        return self.trans_forward(motion_ids, cond, padding_mask, force_mask=force_mask)

class ResidualTransformer2D(nn.Module):

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1, num_heads=4, dropout=0.1, clip_dim=512, shared_codebook=False, share_weight=False, clip_version=None, opt=None, **kargs):
        super(ResidualTransformer2D, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.cond_mode = cond_mode
        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        self.cond_drop_prob = cond_drop_prob
        '\n        Preparing Networks\n        '
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position2d_enc = PositionalEncoding2D(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        if opt.attnj:
            self.attnj = True
            seqTransEncoderLayer2 = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
            self.seqTransEncoder2 = nn.TransformerEncoder(seqTransEncoderLayer2, num_layers=num_layers)
        else:
            self.attnj = False
        if opt.attnt:
            self.attnt = True
            seqTransEncoderLayer3 = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
            self.seqTransEncoder3 = nn.TransformerEncoder(seqTransEncoderLayer3, num_layers=num_layers)
        else:
            self.attnt = False
        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError('Unsupported condition mode!!!')
        _num_tokens = opt.num_tokens2d + 2
        self.mask_id = opt.num_tokens2d
        self.pad_id = opt.num_tokens2d + 1
        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)
        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers - 1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                self.output_proj_weight = output_proj.expand(opt.num_quantizers - 1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers - 1, _num_tokens)
        elif share_weight:
            self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
            self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
            self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
            self.output_proj_bias = None
            self.registered = False
        else:
            output_proj_weight = torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim))
            self.output_proj_weight = nn.Parameter(output_proj_weight)
            self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, _num_tokens)))
            token_embed_weight = torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim))
            self.token_embed_weight = nn.Parameter(token_embed_weight)
        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight
        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model = _load_openai_clip_model(clip_version, self.opt.device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, tokens):
        device = next(self.parameters()).device
        text = tokens.to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)

    def output_project(self, logits, qids):
        """
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)

        :return:
            -logits (bs, ntoken, seqlen)
        """
        output_proj_weight = self.output_proj_weight[qids]
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]
        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def trans_forward(self, motion_codes, qids, cond, padding_mask, T, J, force_mask=False):
        """
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(motion_codes.permute(1, 0, 2))
        N, B, D = x.shape
        q_onehot = self.encode_quant(qids).float().to(x.device)
        q_emb = self.quant_emb(q_onehot).unsqueeze(0)
        cond = self.cond_emb(cond).unsqueeze(0)
        x = x.reshape(T, J, B, D)
        xseq = torch.cat([cond.repeat(J, 1, 1)[None, ...], q_emb.repeat(J, 1, 1)[None, ...], x], dim=0)
        xseq = self.position2d_enc(xseq)
        xseq = xseq.reshape((T + 2) * J, B, D)
        padding_mask = padding_mask.reshape(B, T, J)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)
        padding_mask = rearrange(padding_mask, 'b t j -> b (t j)')
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)
        output = output.reshape(T + 2, J, B, -1)[2:].reshape(T * J, B, D)
        if self.attnj:
            x2 = output.reshape(T, J, B, D)
            x2 = self.position2d_enc(x2)
            x2 = rearrange(x2, 't j b d -> j (b t) d')
            padding_mask2 = padding_mask.reshape(B, T + 2, J)[:, 2:]
            padding_mask2 = rearrange(padding_mask2, 'b t j -> (b t) j')
            nonpad_ids = torch.where(~padding_mask2[:, 0])[0]
            output2 = torch.zeros_like(x2)
            output2[:, nonpad_ids] = self.seqTransEncoder2(x2[:, nonpad_ids])
            output = output + rearrange(output2.reshape(T, B, J, -1), 't b j d -> (t j) b d')
        if self.attnt:
            x3 = output.reshape(T, J, B, D)
            xseq3 = torch.cat([cond.repeat(J, 1, 1)[None, ...], q_emb.repeat(J, 1, 1)[None, ...], x3], dim=0)
            xseq3 = self.position2d_enc(xseq3)
            xseq3 = rearrange(xseq3, 't j b d -> t (b j) d')
            padding_mask3 = padding_mask.reshape(B, T + 2, J)
            padding_mask3 = rearrange(padding_mask3, 'b t j -> (b j) t')
            output3 = self.seqTransEncoder3(xseq3, src_key_padding_mask=padding_mask3)
            output = output + rearrange(output3.reshape(T + 2, B, J, -1)[2:], 't b j d -> (t j) b d')
        logits = self.output_process(output)
        return logits

    def forward(self, motion_codes, qids, cond, padding_mask, T, J, force_mask=False):
        self.process_embed_proj_weight()
        logits = self.trans_forward(motion_codes, qids, cond, padding_mask, T, J, force_mask=force_mask)
        return self.output_project(logits, qids - 1)
