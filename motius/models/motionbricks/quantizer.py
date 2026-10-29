import numpy as np
import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize
class QuantizeEMAResetMultiHead(nn.Module):

    def __init__(self, nb_code: int, code_dim: int, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.num_heads = args.num_heads
        assert self.code_dim % self.num_heads == 0, 'code dim cannot be divided by the number of heads.'
        self.nb_code_per_head = int(round(2 ** (np.log2(self.nb_code) / self.num_heads)))
        assert self.nb_code_per_head ** self.num_heads == self.nb_code, 'the specified number of code is not compatible with the number of heads.'
        self.vq = VectorQuantize(dim=self.code_dim, codebook_dim=self.code_dim // self.num_heads, heads=self.num_heads, separate_codebook_per_head=True, codebook_size=self.nb_code_per_head, accept_image_fmap=False, threshold_ema_dead_code=1, decay=self.mu, kmeans_init=bool(args.kmeans_init))
        self.init = True
        self._calculate_per_head_perplexity = getattr(args, 'calculate_per_head_perplexity', False)

    def forward(self, x):
        features = x.permute(0, 2, 1).contiguous()
        quantized, indices, _ = self.vq(features)
        return dict(raw_quantized=quantized, encoder_features=features, code_vectors=quantized.detach(), indices=indices)

    @staticmethod
    def straight_through(result):
        features=result['encoder_features']
        batch,tokens=features.shape[:2]
        quantized=features+(result['raw_quantized']-features).detach()
        return quantized.view(batch,tokens,-1).permute(0,2,1).contiguous()

    def dequantize(self, indices):
        batch, tokens = indices.shape[:2]
        return self.vq.get_codes_from_indices(indices).view(batch, tokens, -1)
