from typing import Union
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .positions import build_position_encoding
from .attention import SkipTransformerEncoder, SkipTransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class AutoMldVae(nn.Module):

    def __init__(self, nfeats: int, latent_dim: tuple=[1, 256], h_dim: int=512, ff_size: int=1024, num_layers: int=9, num_heads: int=4, dropout: float=0.1, arch: str='all_encoder', normalize_before: bool=False, activation: str='gelu', position_embedding: str='learned', **kwargs) -> None:
        super().__init__()
        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = 'mld'
        self.query_pos_encoder = build_position_encoding(self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(self.h_dim, position_embedding=position_embedding)
        encoder_layer = TransformerEncoderLayer(self.h_dim, num_heads, ff_size, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)
        if self.arch == 'all_encoder':
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        elif self.arch == 'encoder_decoder':
            decoder_layer = TransformerDecoderLayer(self.h_dim, num_heads, ff_size, dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            raise ValueError('Not support architecture!')
        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.h_dim))
        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)
        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))

    def encode(self, future_motion, history_motion, scale_latent: bool=False) -> Union[Tensor, Distribution]:
        device = future_motion.device
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]
        x = torch.cat((history_motion, future_motion), dim=1)
        x = self.skel_embedding(x)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq)[:dist.shape[0]]
        dist = self.encoder_latent_proj(dist)
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if scale_latent:
            latent = latent / self.latent_std
        return (latent, dist)

    def decode(self, z: Tensor, history_motion, nfuture, scale_latent: bool=False):
        bs = history_motion.shape[0]
        if scale_latent:
            z = z * self.latent_std
        z = self.decoder_latent_proj(z)
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)
        if self.arch == 'all_encoder':
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-nfuture:]
        elif self.arch == 'encoder_decoder':
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(tgt=xseq, memory=z)
            output = output[-nfuture:]
        output = self.final_layer(output)
        feats = output.permute(1, 0, 2)
        return feats

    def forward(self, future_motion=None, history_motion=None, z=None, nfuture=None, scale_latent=False):
        if z is not None:
            return self.decode(z, history_motion, nfuture, scale_latent)
        latent, distribution = self.encode(future_motion, history_motion, scale_latent)
        reconstruction = self.decode(latent, history_motion, future_motion.shape[1], scale_latent)
        return reconstruction, latent, distribution
