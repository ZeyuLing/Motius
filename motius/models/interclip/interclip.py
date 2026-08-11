"""InterGen / InterMask InterCLIP evaluator network.

This is a small, self-contained port of the official InterGen evaluator model
(`datasets/evaluator_models.py` plus the `MotionEncoder` bits from
`models/nets.py`). It intentionally avoids importing any third-party checkout;
only the OpenAI CLIP package and the evaluator checkpoint are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import clip
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class InterCLIPConfig:
    NAME: str = "InterCLIP"
    NUM_LAYERS: int = 8
    NUM_HEADS: int = 8
    DROPOUT: float = 0.1
    # Official InterCLIP receives native 262 per person, then drops the last
    # four foot-contact channels before the motion encoder, so the checkpointed
    # input projection is `(262 - 4) * 2 = 516`.
    INPUT_DIM: int = 258
    LATENT_DIM: int = 1024
    FF_SIZE: int = 2048
    ACTIVATION: str = "gelu"
    MOTION_REP: str = "global"
    FINETUNE: bool = False


def _set_requires_grad(module: nn.Module, requires_grad: bool = False) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class InterCLIPMotionEncoder(nn.Module):
    def __init__(self, cfg: InterCLIPConfig):
        super().__init__()
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))
        self.embed_motion = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, cfg.DROPOUT, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.FF_SIZE,
            dropout=cfg.DROPOUT,
            activation=cfg.ACTIVATION,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_LAYERS)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, mask = batch["motions"], batch["mask"]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[1], 2, -1)[..., :-4].reshape(batch_size, x.shape[1], -1)
        x_emb = self.embed_motion(x)
        q = self.query_token[torch.zeros(batch_size, dtype=torch.long, device=x.device)][:, None]
        emb = torch.cat([q, x_emb], dim=1)

        seq_mask = mask > 0.5
        token_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        batch["motion_emb"] = self.out(h[:, 0])
        return batch


class InterCLIP(nn.Module):
    def __init__(self, cfg: InterCLIPConfig = InterCLIPConfig()):
        super().__init__()
        self.cfg = cfg
        self.motion_encoder = InterCLIPMotionEncoder(cfg)
        # InterCLIP checkpoints contain both text embedding tensors. Construct
        # their modules directly instead of downloading an unused CLIP visual
        # encoder and transformer during every evaluator load.
        self.token_embedding = nn.Embedding(49408, 768)
        self.positional_embedding = nn.Parameter(torch.empty(77, 768))
        self.dtype = torch.float32
        self.latent_scale = nn.Parameter(torch.Tensor([1]))
        _set_requires_grad(self.token_embedding, False)

        text_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=cfg.FF_SIZE,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.textTransEncoder = nn.TransformerEncoder(text_layer, num_layers=8)
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 512)

    def generate_src_mask(self, timesteps: int, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(lengths.shape[0], timesteps, device=lengths.device)
        for i, length in enumerate(lengths):
            mask[i, int(length):] = 0
        return mask

    def encode_motion(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(
            batch["motions"].device
        )
        batch.update(self.motion_encoder(batch))
        batch["motion_emb"] = batch["motion_emb"] / batch["motion_emb"].norm(dim=-1, keepdim=True) * self.latent_scale
        return batch

    def encode_text(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        raw_text: List[str] = list(batch["text"])
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)
            pe_tokens = x + self.positional_embedding.type(self.dtype)
        out = self.textTransEncoder(pe_tokens)
        out = self.text_ln(out)
        out = out[torch.arange(x.shape[0], device=device), text.argmax(dim=-1)]
        out = self.out(out)
        batch["text_emb"] = out / out.norm(dim=-1, keepdim=True) * self.latent_scale
        return batch


def load_interclip_checkpoint(path: str, device: str = "cpu", input_dim: int = 258) -> InterCLIP:
    cfg = InterCLIPConfig(INPUT_DIM=input_dim)
    model = InterCLIP(cfg)
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
