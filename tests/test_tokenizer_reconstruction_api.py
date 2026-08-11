from __future__ import annotations

import numpy as np
import pytest
import torch

from motius.pipelines.momask import MoMaskPipeline
from motius.pipelines.motiongpt import MotionGPTPipeline
from motius.pipelines.t2mgpt import T2MGPTPipeline


class _IdentityAutoencoder:
    def __init__(self, *, channel_first: bool = False) -> None:
        self.channel_first = channel_first

    def __call__(self, motion: torch.Tensor):
        output = motion.permute(0, 2, 1) if self.channel_first else motion
        return output, None, None


class _Bundle:
    def __init__(self, autoencoder_name: str) -> None:
        self.device = torch.device("cpu")
        self.mean = torch.linspace(-1.0, 1.0, 263)
        self.std = torch.linspace(0.5, 1.5, 263)
        setattr(
            self,
            autoencoder_name,
            _IdentityAutoencoder(channel_first=autoencoder_name == "vq_model"),
        )

    def eval(self):
        return self

    def to_device(self, device):
        self.device = torch.device(device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        return self

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std + self.mean


@pytest.mark.parametrize(
    ("pipeline_cls", "autoencoder_name"),
    [
        (T2MGPTPipeline, "vqvae"),
        (MoMaskPipeline, "vq_model"),
        (MotionGPTPipeline, "vae"),
    ],
)
def test_tokenizer_reconstruction_round_trip_and_order(pipeline_cls, autoencoder_name):
    pipeline = pipeline_cls(_Bundle(autoencoder_name))
    motions = [
        np.random.default_rng(1).normal(size=(9, 263)).astype(np.float32),
        np.random.default_rng(2).normal(size=(12, 263)).astype(np.float32),
        np.random.default_rng(3).normal(size=(8, 263)).astype(np.float32),
    ]

    reconstructed = pipeline.infer_motion_reconstruction(motions, batch_size=1)

    assert [len(value) for value in reconstructed] == [8, 12, 8]
    for source, result in zip(motions, reconstructed):
        np.testing.assert_allclose(result, source[: len(result)], atol=2e-7, rtol=2e-7)
    assert pipeline.infer_reconstruction.__func__ is pipeline.infer_motion_reconstruction.__func__


@pytest.mark.parametrize(
    ("pipeline_cls", "autoencoder_name"),
    [
        (T2MGPTPipeline, "vqvae"),
        (MoMaskPipeline, "vq_model"),
        (MotionGPTPipeline, "vae"),
    ],
)
def test_tokenizer_reconstruction_rejects_invalid_motion(pipeline_cls, autoencoder_name):
    pipeline = pipeline_cls(_Bundle(autoencoder_name))

    with pytest.raises(ValueError, match=r"expected \(T,263\)"):
        pipeline.infer_motion_reconstruction([np.zeros((8, 135), np.float32)])
    with pytest.raises(ValueError, match="at least 4 frames"):
        pipeline.infer_motion_reconstruction([np.zeros((3, 263), np.float32)])
