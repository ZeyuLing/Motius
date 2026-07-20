"""Music-to-dance inference with the released Bailando architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from motius.models.bailando.audio import (
    BAILANDO_AUDIO_FEATURE_DIM,
    BAILANDO_BEAT_CHANNEL,
    extract_bailando_audio_features,
)
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@dataclass(frozen=True)
class DanceGenerationOutput:
    """Generated AIST++ SMPL-24 joints plus reproducibility metadata."""

    joints: np.ndarray
    model_motion: np.ndarray
    codes_up: np.ndarray
    codes_down: np.ndarray
    music_features: np.ndarray
    music_beats: np.ndarray
    fps: float


def _world_joints_to_model_motion(joints: torch.Tensor) -> torch.Tensor:
    if joints.ndim == 3:
        joints = joints.unsqueeze(0)
    if joints.ndim != 4 or joints.shape[-2:] != (24, 3):
        raise ValueError(
            "initial_motion must be AIST++ SMPL-24 joints with shape "
            f"(T,24,3) or (B,T,24,3), got {tuple(joints.shape)}"
        )
    root = joints[:, :, :1]
    relative = joints - root
    relative[:, :, 0] = root[:, :, 0]
    return relative.flatten(2)


def _decoded_motion_to_world_joints(motion: torch.Tensor) -> torch.Tensor:
    if motion.ndim != 3 or motion.shape[-1] != 72:
        raise ValueError(f"Expected decoded motion (B,T,72), got {tuple(motion.shape)}")
    velocity = motion[:, :, :3]
    root = torch.zeros_like(velocity)
    if motion.shape[1] > 1:
        root[:, 1:] = torch.cumsum(velocity[:, :-1], dim=1)
    relative = motion.reshape(motion.shape[0], motion.shape[1], 24, 3).clone()
    joints = relative + root[:, :, None]
    joints[:, :, 0] = root
    return joints


@PIPELINES.register_module()
class BailandoPipeline(BasePipeline):
    """Generate 60 fps SMPL-24 joint motion from audio or official features."""

    BUNDLE_CLS = "motius.models.bailando.BailandoBundle"

    def __init__(self, bundle, device: str | None = None, **kwargs):
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

    def _prepare_features(
        self,
        *,
        audio: str | Path | np.ndarray | None,
        sample_rate: int | None,
        music_features: np.ndarray | torch.Tensor | None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        if (audio is None) == (music_features is None):
            raise ValueError("Provide exactly one of audio or music_features")
        if music_features is None:
            music_features = extract_bailando_audio_features(
                audio, sample_rate=sample_rate
            )
        feature_array = (
            music_features.detach().cpu().numpy()
            if torch.is_tensor(music_features)
            else np.asarray(music_features)
        ).astype(np.float32)
        if feature_array.ndim == 2:
            feature_array = feature_array[None]
        if feature_array.ndim != 3 or feature_array.shape[-1] != BAILANDO_AUDIO_FEATURE_DIM:
            raise ValueError(
                "music_features must have shape (T,438) or (B,T,438), "
                f"got {feature_array.shape}"
            )
        if feature_array.shape[1] < 2:
            raise ValueError("Bailando requires at least two music feature frames")
        return torch.from_numpy(feature_array).to(self.device), feature_array

    def _initial_codes(
        self,
        batch_size: int,
        *,
        initial_motion: np.ndarray | torch.Tensor | None,
        initial_codes: Sequence[int] | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_motion is not None and initial_codes is not None:
            raise ValueError("Provide initial_motion or initial_codes, not both")
        if initial_motion is not None:
            motion = torch.as_tensor(
                initial_motion, dtype=torch.float32, device=self.device
            )
            model_motion = _world_joints_to_model_motion(motion)
            if model_motion.shape[0] == 1 and batch_size > 1:
                model_motion = model_motion.expand(batch_size, -1, -1).clone()
            if model_motion.shape[0] != batch_size:
                raise ValueError("initial_motion batch size does not match music")
            encoded = self.bundle.vqvae.encode(model_motion.clone())
            return encoded[0][0][:, :1], encoded[1][0][:, :1]

        values = initial_codes or self.bundle.default_initial_codes
        if len(values) != 2:
            raise ValueError("initial_codes must contain upper and lower code")
        tensors = []
        for value in values:
            if torch.is_tensor(value):
                code = value.to(self.device, dtype=torch.long)
                if code.ndim == 0:
                    code = code.reshape(1, 1)
                elif code.ndim == 1:
                    code = code[:, None]
            else:
                code = torch.full(
                    (batch_size, 1), int(value), device=self.device, dtype=torch.long
                )
            if code.shape[0] == 1 and batch_size > 1:
                code = code.expand(batch_size, -1)
            if code.shape != (batch_size, 1):
                raise ValueError(
                    f"Each initial code must resolve to {(batch_size, 1)}, got {tuple(code.shape)}"
                )
            tensors.append(code)
        return tensors[0], tensors[1]

    @torch.inference_mode()
    def infer_music_to_dance(
        self,
        audio: str | Path | np.ndarray | None = None,
        *,
        sample_rate: int | None = None,
        music_features: np.ndarray | torch.Tensor | None = None,
        initial_motion: np.ndarray | torch.Tensor | None = None,
        initial_codes: Sequence[int] | tuple[torch.Tensor, torch.Tensor] | None = None,
        max_frames: int | None = None,
    ) -> DanceGenerationOutput:
        """Generate dance from raw audio or precomputed official features.

        For the official AIST++ protocol, pass the corresponding GT motion as
        ``initial_motion``; only its first VQ token seeds generation. For music
        without a reference motion, the released demo seed ``(423, 12)`` is
        used by default.
        """

        features, feature_array = self._prepare_features(
            audio=audio,
            sample_rate=sample_rate,
            music_features=music_features,
        )
        if max_frames is not None:
            if max_frames < self.bundle.code_downsample:
                raise ValueError("max_frames must be at least one VQ stride")
            feature_frames = max(
                2,
                int(np.ceil(max_frames / self.bundle.code_downsample)),
            )
            features = features[:, :feature_frames]
            feature_array = feature_array[:, :feature_frames]

        seed_up, seed_down = self._initial_codes(
            features.shape[0],
            initial_motion=initial_motion,
            initial_codes=initial_codes,
        )
        generated = self.bundle.gpt.sample(
            (seed_up, seed_down), cond=features[:, 1:]
        )
        codes_up, codes_down = generated[0][0], generated[1][0]
        decoded = self.bundle.vqvae.decode(generated)
        joints = _decoded_motion_to_world_joints(decoded)
        if max_frames is not None:
            decoded = decoded[:, :max_frames]
            joints = joints[:, :max_frames]

        return DanceGenerationOutput(
            joints=joints.cpu().numpy().astype(np.float32),
            model_motion=decoded.cpu().numpy().astype(np.float32),
            codes_up=codes_up.cpu().numpy(),
            codes_down=codes_down.cpu().numpy(),
            music_features=feature_array,
            music_beats=feature_array[:, :, BAILANDO_BEAT_CHANNEL] > 0.5,
            fps=self.bundle.fps,
        )

    def __call__(self, audio=None, **kwargs):
        return self.infer_music_to_dance(audio, **kwargs)


__all__ = ["BailandoPipeline", "DanceGenerationOutput"]
