import torch
import clip

from torch import nn
from clip.model import LayerNorm, Transformer
from . import *


class InterGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        # The released InterGen checkpoint includes the complete frozen CLIP
        # text tower. Construct its architecture locally so loading InterGen
        # never downloads the unused CLIP visual tower.
        attention_mask = torch.empty(77, 77).fill_(float("-inf")).triu_(1)
        self.token_embedding = nn.Embedding(49408, 768)
        self.clip_transformer = Transformer(
            width=768,
            layers=12,
            heads=12,
            attn_mask=attention_mask,
        )
        self.positional_embedding = nn.Parameter(torch.empty(77, 768))
        self.ln_final = LayerNorm(768)
        self.dtype = torch.float32

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        self.clip_ln = nn.LayerNorm(768)

    def compute_loss(self, batch):
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch["cond"] = cond

        return batch
