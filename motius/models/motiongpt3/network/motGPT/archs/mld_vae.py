from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

# from motius.models.motiongpt3.network.motGPT.archs.tools.embeddings import TimestepEmbedding, Timesteps
from motius.models.motiongpt3.network.motGPT.archs.operator import PositionalEncoding
from motius.models.motiongpt3.network.motGPT.archs.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from motius.models.motiongpt3.network.motGPT.models.utils.position_encoding import build_position_encoding
from motius.models.motiongpt3.network.motGPT.utils.temos_utils import lengths_to_mask
"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""


class MldVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 datatype='humanml',
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = ablation['MLP_DIST']
        self.pe_type = ablation['PE_TYPE']
        if 'motionx' in datatype.lower():
            # motionx_vae:
            self.mean_std_inv = 0.7281  # 0.7281 for 4/9
            self.mean_mean = 0.0636  # -0.1463 for 4/9
        else:
            # humanml3d_vae:
            self.mean_std_inv = 0.8457  # 0.6769 for 4/5; 0.8457 for 4/9
            self.mean_std_inv_2 = self.mean_std_inv**2
            # 0.71521
            self.mean_mean = -0.1379  # -0.1379 for 4/5; -0.1463 for 4/9

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))  # 2,256

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def encode_dist(self, features: Tensor, lengths: Optional[List[int]] = None):
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device,max_len=nframes)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)  # cat([bs,2], [bs,max_seq_len])

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)
        if xseq.shape[0]>500:
            print(xseq.shape, dist.shape, x.shape)

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]
        return dist
    
    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode_dist2z(self, dist):
        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...]  # 1,bs,256
            logvar = dist[self.latent_size:, ...]  # 1,bs,256

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        # print('in mldvae encode', features.shape, latent.shape) # ([bs, 40, 263]) torch.Size([1, bs, 256])
        # print('[MLDVAE] latent is None', latent is None, dist is None)
        return latent, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        dist = self.encode_dist(features, lengths)  # 1,bs,256
        # content distribution
        # self.latent_dim => 2*self.latent_dim
        latent, dist = self.encode_dist2z(dist)
        return latent, dist


    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        # if self.pe_type == "actor":
        #     queries = self.query_pos_decoder(queries)
        #     output = self.decoder(tgt=queries,
        #                             memory=z,
        #                             tgt_key_padding_mask=~mask).squeeze(0)
        # elif self.pe_type == "mld":
        queries = self.query_pos_decoder(queries)
        # mem_pos = self.mem_pos_decoder(z)
        output = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=~mask,
            # query_pos=query_pos,
            # pos=mem_pos,
        ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
