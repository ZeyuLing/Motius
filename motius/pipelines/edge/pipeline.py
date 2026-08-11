"""Motius-native inference for EDGE music-to-dance generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from motius.models.edge.audio import (
    extract_edge_jukebox_features,
    validate_edge_music_features,
)
from motius.models.edge.network import (
    edge_motion_to_aistpp_joints,
    edge_motion_to_motion135,
)
from motius.models.edge.network.sampler import edge_ddim_sample, stitch_edge_windows
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@dataclass(frozen=True)
class EDGEGenerationOutput:
    """Generated EDGE representation and AIST++-compatible joints."""

    joints: np.ndarray
    edge_motion: np.ndarray
    motion_135: np.ndarray
    contacts: np.ndarray
    music_features: np.ndarray
    fps: float


@PIPELINES.register_module()
class EDGEPipeline(BasePipeline):
    """Generate 30 fps SMPL-24 dance motion from Jukebox music features."""

    BUNDLE_CLS = "motius.models.edge.EDGEBundle"

    def __init__(self, bundle, device: str | torch.device | None = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device: str | torch.device):
        self.bundle.to(torch.device(device))
        self.bundle.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.inference_mode()
    def infer_music_to_dance(
        self,
        audio: str | Path | None = None,
        *,
        music_features: np.ndarray | torch.Tensor | None = None,
        max_seconds: float | None = None,
        max_frames: int | None = None,
        seed: int = 0,
        guidance_weight: float | None = None,
        sampling_steps: int | None = None,
        eta: float | None = None,
        jukebox_fp16: bool = False,
        jukebox_cache_dir: str | Path | None = None,
    ) -> EDGEGenerationOutput:
        """Run the official overlapping-window EDGE inference protocol."""

        if (audio is None) == (music_features is None):
            raise ValueError("Provide exactly one of audio or music_features")
        if music_features is None:
            features = extract_edge_jukebox_features(
                audio,
                max_seconds=max_seconds,
                fp16=jukebox_fp16,
                cache_dir=jukebox_cache_dir,
            )
        else:
            if torch.is_tensor(music_features):
                music_features = music_features.detach().cpu().numpy()
            features = validate_edge_music_features(music_features)
        condition = torch.from_numpy(features).to(self.device)
        sampling = self.bundle.config["sampling"]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        normalized = edge_ddim_sample(
            self.bundle.network,
            condition,
            representation_dim=int(self.bundle.config["representation_dim"]),
            guidance_weight=float(
                sampling["guidance_weight"]
                if guidance_weight is None
                else guidance_weight
            ),
            sampling_steps=int(
                sampling["sampling_steps"] if sampling_steps is None else sampling_steps
            ),
            eta=float(sampling["eta"] if eta is None else eta),
            generator=generator,
        )
        windows = self.bundle.denormalize(normalized)
        motion = stitch_edge_windows(windows)
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            motion = motion[: int(max_frames)]
        joints = edge_motion_to_aistpp_joints(motion)[0]
        motion_135 = edge_motion_to_motion135(motion)
        motion_np = motion.float().cpu().numpy().astype(np.float32)
        return EDGEGenerationOutput(
            joints=joints.float().cpu().numpy().astype(np.float32),
            edge_motion=motion_np,
            motion_135=motion_135.float().cpu().numpy().astype(np.float32),
            contacts=motion_np[:, :4],
            music_features=features,
            fps=self.bundle.fps,
        )

    def __call__(self, audio=None, **kwargs):
        return self.infer_music_to_dance(audio, **kwargs)


__all__ = ["EDGEGenerationOutput", "EDGEPipeline"]
