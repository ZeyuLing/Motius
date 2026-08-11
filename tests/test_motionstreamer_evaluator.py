from __future__ import annotations

import torch

from motius.models.motionstreamer.evaluator import (
    ActorAgnosticEncoder,
    PositionalEncoding,
)
from tools.eval_t2m_motionstreamer import _selected_caption


def test_positional_encoding_preserves_shape() -> None:
    encoder = PositionalEncoding(16, dropout=0.0)
    values = torch.zeros(7, 3, 16)
    assert encoder(values).shape == values.shape


def test_motionstreamer_actor_encoder_shapes() -> None:
    encoder = ActorAgnosticEncoder(
        nfeats=12,
        latent_dim=16,
        num_layers=1,
        num_heads=4,
        ff_size=32,
        dropout=0.0,
        max_len=8,
    ).eval()
    distribution = encoder(torch.randn(2, 8, 12), torch.tensor([8, 4]))
    assert distribution.loc.shape == (2, 16)
    assert distribution.scale.shape == (2, 16)
    assert torch.isfinite(distribution.loc).all()


def test_selected_caption_treats_nan_tag_as_full_clip(tmp_path) -> None:
    caption = tmp_path / "sample.txt"
    caption.write_text("a person waves.#tokens#nan#0.0\n", encoding="utf-8")
    assert _selected_caption(caption) == "a person waves."
