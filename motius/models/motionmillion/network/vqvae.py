"""HumanVQVAE / VQVAE_251 (FSQ tokenizer) for MotionMillion / "Go to Zero".

Refactored to take explicit keyword arguments instead of an ``args`` namespace.
Only the released Go-to-Zero configuration is supported: non-causal conv
encoder/decoder + FSQ quantizer. The 272-dim motion representation is assumed
(``dataname != 'kit'``).
"""
import torch.nn as nn

from .encdec import Decoder, Encoder
from .fsq import FSQ

# FSQ level tables keyed by the effective codebook size (nb_code).
_FSQ_LEVELS = {
    256: [8, 6, 5],
    512: [8, 8, 8],
    1024: [8, 5, 5, 5],
    2048: [8, 8, 6, 5],
    4096: [7, 5, 5, 5, 5],
    16384: [8, 8, 8, 6, 5],
    65536: [8, 8, 8, 5, 5, 5],
}


class VQVAE_251(nn.Module):
    def __init__(
        self,
        nb_code=65536,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        kernel_size=3,
        use_patcher=False,
        patch_size=1,
        patch_method="haar",
        input_dim=272,
        quantizer="FSQ",
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer

        self.encoder = Encoder(
            input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
            activation=activation, norm=norm, kernel_size=kernel_size,
            use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method,
        )
        self.decoder = Decoder(
            input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
            activation=activation, norm=norm, kernel_size=kernel_size,
            use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method,
        )

        if quantizer != "FSQ":
            raise ValueError(f"Only FSQ is supported; got quantizer={quantizer!r}")
        if nb_code not in _FSQ_LEVELS:
            raise ValueError(f"Unsupported nb_code={nb_code}")
        self.quantizer = FSQ(levels=_FSQ_LEVELS[nb_code], dim=code_dim)

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        _, code_idx, _, _, _, _ = self.quantizer(x_encoder)
        return code_idx.view(N, -1)

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        return self.postprocess(x_decoder), loss, perplexity, activate, indices

    def forward_decoder(self, x):
        """Token indices ``(N, T)`` -> reconstructed motion ``(N, T', input_dim)``."""
        x_d = self.quantizer.dequantize(x)  # (N, T, code_dim)
        x_d = x_d.permute(0, 2, 1).contiguous()  # (N, code_dim, T)
        x_decoder = self.decoder(x_d)
        return self.postprocess(x_decoder)


class HumanVQVAE(nn.Module):
    def __init__(
        self,
        nb_code=65536,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        kernel_size=3,
        use_patcher=False,
        patch_size=1,
        patch_method="haar",
        input_dim=272,
        nb_joints=22,
        quantizer="FSQ",
    ):
        super().__init__()
        self.nb_joints = nb_joints
        self.vqvae = VQVAE_251(
            nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth,
            dilation_growth_rate, activation=activation, norm=norm, kernel_size=kernel_size,
            use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method,
            input_dim=input_dim, quantizer=quantizer,
        )

    def encode(self, x):
        return self.vqvae.encode(x)

    def forward(self, x):
        return self.vqvae(x)

    def forward_decoder(self, x):
        return self.vqvae.forward_decoder(x)
