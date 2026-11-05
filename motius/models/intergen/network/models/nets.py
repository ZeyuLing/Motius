"""Paired motion denoiser; checkpoint names follow InterGen."""
import torch
from torch import nn
from .utils import PositionalEncoding, TimestepEmbedder, zero_module
from .blocks import TransformerBlock
from .layers import FinalLayer

class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))



    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:]

        if mask is not None:
            mask = mask[...,0]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)

        return output



class InterDiffusion(nn.Module):
    """Checkpoint-compatible denoiser container (no diffusion process)."""
    def __init__(self, cfg):
        super().__init__()
        self.net = InterDenoiser(cfg.INPUT_DIM, cfg.LATENT_DIM, ff_size=cfg.FF_SIZE,
                                num_layers=cfg.NUM_LAYERS, num_heads=cfg.NUM_HEADS,
                                dropout=cfg.DROPOUT, activation=cfg.ACTIVATION,
                                cfg_weight=cfg.CFG_WEIGHT)

    def forward(self, x, timesteps, mask=None, cond=None):
        return self.net(x, timesteps, mask=mask, cond=cond)
