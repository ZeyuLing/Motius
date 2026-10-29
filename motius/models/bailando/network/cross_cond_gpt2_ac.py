"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
# from .gpt3p import condGPT3Part
# logger = logging.getLogger(__name__)


class CrossCondGPT2AC(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        # self.down_half_gpt = CrossCondGPTHead(config.down_half_gpt)
        # if hasattr(config, 'critic_net'):
        self.critic_net = CrossCondGPTHead(config.critic_net)
        self.block_size = config.block_size
    #     # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    # def get_block_size(self):
    #     return self.block_size

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    def get_block_size(self):
        return self.block_size

    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False

            # def sample(self, xs, cond):
    #     x_up, x_down = xs
    #     return (self.up_half_gpt.sample(x_up, cond), self.down_half_gpt.sample(x_down, cond))




    def forward(self, idxs, cond, head="policy"):
        idx_up, idx_down = idxs
        features = self.gpt_base(idx_up, idx_down, cond)
        network = self.critic_net if head == "critic" else self.gpt_head
        return network(features)


class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # self.mask = se
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T // 3
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:t,:t].repeat(1, 1, 3, 3) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondGPTBase(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # # input embedding stem
        # self.requires_head = config.requires_head
        # self.requires_tail = config.requires_tail

        # if config.requires_head:
        # dance_
        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd  )
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size*3, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size


        self.apply(self._init_weights)


        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, idx_up, idx_down, cond):
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # if self.requires_head:
        token_embeddings_up = self.tok_emb_up(idx_up) # each index maps to a (learnable) vector
        token_embeddings_down = self.tok_emb_down(idx_down) # each index maps to a (learnable) vector
        token_embeddings = torch.cat([self.cond_emb(cond), token_embeddings_up, token_embeddings_down ], dim=1)

        position_embeddings = torch.cat([self.pos_emb[:, :t, :], self.pos_emb[:, self.block_size:self.block_size+t, :], self.pos_emb[:, self.block_size*2:self.block_size*2+t, :]], dim=1) # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)
        # x = self.ln_f(x)

        return x

class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            # module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):

        x = self.blocks(x)
        x = self.ln_f(x)
        N, T, C = x.size()
        t = T//3
        logits_up = self.head_up(x[:, t:t*2, :])
        logits_down = self.head_down(x[:, t*2:t*3, :]) # down half

        return logits_up, logits_down
