"""Text-to-motion and music-to-dance inference with TM2D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from motius.models.bailando.audio import (
    BAILANDO_BEAT_CHANNEL,
    extract_bailando_audio_features,
)
from motius.motion.representation.humanml import recover_from_ric
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@dataclass(frozen=True)
class TM2DGenerationOutput:
    """Generated joints and the native 60 fps TM2D representation."""

    joints: np.ndarray
    native_joints: np.ndarray
    model_motion: np.ndarray
    motion_tokens: np.ndarray
    fps: float
    native_fps: float
    music_features: np.ndarray | None = None
    music_beats: np.ndarray | None = None


def _resample_exact(joints: np.ndarray, output_frames: int) -> np.ndarray:
    if len(joints) == output_frames:
        return joints.copy()
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    if len(joints) == 1:
        return np.repeat(joints, output_frames, axis=0)
    source = np.linspace(0.0, 1.0, len(joints), dtype=np.float64)
    target = np.linspace(0.0, 1.0, output_frames, dtype=np.float64)
    flat = joints.reshape(len(joints), -1)
    result = np.empty((output_frames, flat.shape[1]), dtype=np.float64)
    for channel in range(flat.shape[1]):
        result[:, channel] = np.interp(target, source, flat[:, channel])
    return result.reshape(output_frames, *joints.shape[1:]).astype(np.float32)


@PIPELINES.register_module()
class TM2DPipeline(BasePipeline):
    """Run both single-modality tasks retained by the joint TM2D checkpoint."""

    BUNDLE_CLS = "motius.models.tm2d.TM2DBundle"

    def __init__(self, bundle, device: str | None = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device: str | torch.device):
        self.bundle.to(torch.device(device))
        self.bundle.eval()
        return self

    @property
    def device(self):
        return self.bundle.device

    def _generator(self, seed: int):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def _decode(self, tokens: torch.Tensor):
        motion = self.bundle.decode_tokens(tokens)
        joints = recover_from_ric(motion, joints_num=24)
        return motion, joints

    @torch.inference_mode()
    def infer_text_to_motion(
        self,
        caption: str,
        *,
        duration_seconds: float | None = None,
        num_frames: int | None = None,
        output_fps: float = 30.0,
        pretokenized: Sequence[str] | None = None,
        sample: bool = True,
        top_k: int | None = None,
        seed: int = 0,
    ) -> TM2DGenerationOutput:
        """Generate one motion using the TM2D text-only branch.

        Duration is required because the released model conditions on an
        explicit motion-length indicator. ``num_frames`` takes precedence and
        is interpreted at ``output_fps``.
        """

        if num_frames is None:
            if duration_seconds is None or duration_seconds <= 0:
                raise ValueError("Provide positive duration_seconds or num_frames")
            num_frames = max(1, int(round(duration_seconds * output_fps)))
        if num_frames < 1 or output_fps <= 0:
            raise ValueError("num_frames and output_fps must be positive")
        duration_seconds = num_frames / float(output_fps)
        token_count = max(
            1,
            int(round(duration_seconds * self.bundle.fps / self.bundle.code_stride)),
        )
        if token_count >= self.bundle.config["max_target_length"]:
            raise ValueError(
                f"Requested duration needs {token_count} tokens; TM2D supports fewer than "
                f"{self.bundle.config['max_target_length']}"
            )

        source_ids = self.bundle.tokenizer.encode(
            caption,
            token_count,
            pretokenized=pretokenized,
        )
        source = torch.from_numpy(source_ids).to(self.device)
        encoded, source_mask = self.bundle.text_transformer.encode(source)
        initial = torch.full(
            (1, 1),
            self.bundle.config["motion_start_id"],
            dtype=torch.long,
            device=self.device,
        )
        generated = self.bundle.text_transformer.generate_from_encoded(
            encoded,
            source_mask,
            initial,
            token_count,
            sample=sample,
            top_k=top_k,
            forbidden_ids=(
                self.bundle.config["motion_start_id"],
                self.bundle.config["motion_end_id"],
                self.bundle.config["motion_pad_id"],
            ),
            generator=self._generator(seed),
        )[:, 1:]
        motion, native_joints = self._decode(generated)
        native_array = native_joints[0].cpu().numpy().astype(np.float32)
        output_joints = _resample_exact(native_array, num_frames)
        return TM2DGenerationOutput(
            joints=output_joints,
            native_joints=native_array,
            model_motion=motion[0].cpu().numpy().astype(np.float32),
            motion_tokens=generated[0].cpu().numpy(),
            fps=float(output_fps),
            native_fps=self.bundle.fps,
        )

    def _prepare_music_features(
        self,
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
        values = (
            music_features.detach().cpu().numpy()
            if torch.is_tensor(music_features)
            else np.asarray(music_features)
        ).astype(np.float32)
        if values.ndim == 2:
            values = values[None]
        expected = self.bundle.config["audio_feature_dim"]
        if values.ndim != 3 or values.shape[-1] != expected:
            raise ValueError(f"music_features must have shape (T,{expected}) or (B,T,{expected})")
        if values.shape[0] != 1:
            raise ValueError("TM2D music inference currently accepts one clip per call")
        if values.shape[1] < 1:
            raise ValueError("music_features cannot be empty")
        return torch.from_numpy(values).to(self.device), values

    def _encode_music_chunks(self, features: torch.Tensor):
        chunk = self.bundle.config["audio_chunk_length"]
        overlap = self.bundle.config["audio_chunk_overlap"]
        stride = chunk - overlap
        encoded_parts, mask_parts = [], []
        for start in range(0, features.shape[1], stride):
            values = features[:, start : start + chunk]
            lengths = torch.full(
                (values.shape[0],), values.shape[1], dtype=torch.long, device=self.device
            )
            encoded, mask = self.bundle.audio_transformer.encode(values, lengths)
            if start:
                encoded, mask = encoded[:, overlap:], mask[:, :, overlap:]
            encoded_parts.append(encoded)
            mask_parts.append(mask)
        return torch.cat(encoded_parts, dim=1), torch.cat(mask_parts, dim=2)

    def _music_seed_token(
        self,
        initial_token: int | None,
        initial_motion,
        generator,
    ) -> int:
        if initial_token is not None and initial_motion is not None:
            raise ValueError("Provide initial_token or initial_motion, not both")
        if initial_motion is not None:
            reference_tokens = self.bundle.encode_motion(initial_motion).reshape(-1)
            index = torch.randint(
                len(reference_tokens), (1,), device=self.device, generator=generator
            )
            return int(reference_tokens[index].item())
        if initial_token is None:
            initial_token = self.bundle.config["default_music_seed_token"]
        if not 0 <= int(initial_token) < self.bundle.config["codebook_size"]:
            raise ValueError("initial_token must be a VQ codebook index")
        return int(initial_token)

    @torch.inference_mode()
    def infer_music_to_dance(
        self,
        audio: str | Path | np.ndarray | None = None,
        *,
        sample_rate: int | None = None,
        music_features: np.ndarray | torch.Tensor | None = None,
        initial_token: int | None = None,
        initial_motion: np.ndarray | torch.Tensor | None = None,
        sample: bool = True,
        top_k: int | None = None,
        seed: int = 0,
        max_frames: int | None = None,
    ) -> TM2DGenerationOutput:
        """Generate 60 fps dance from raw audio or official 438-D features.

        The authors' AIST++ protocol seeds generation with one VQ token sampled
        from the paired GT motion. Pass its unnormalized 287-D representation as
        ``initial_motion`` to reproduce that protocol. Standalone inference uses
        a fixed released-demo token so no reference motion is required.
        """

        features, feature_array = self._prepare_music_features(
            audio, sample_rate, music_features
        )
        if max_frames is not None:
            feature_frames = max(1, int(np.ceil(max_frames / self.bundle.code_stride)))
            features = features[:, :feature_frames]
            feature_array = feature_array[:, :feature_frames]
        encoded, source_mask = self._encode_music_chunks(features)
        generator = self._generator(seed)
        seed_token = self._music_seed_token(initial_token, initial_motion, generator)
        chunk = self.bundle.config["audio_chunk_length"]
        overlap = self.bundle.config["audio_chunk_overlap"]

        parts = []
        start = 0
        previous = torch.full(
            (1, 1), seed_token, dtype=torch.long, device=self.device
        )
        while start < features.shape[1]:
            end = min(start + chunk, features.shape[1])
            steps = end - start - 1
            generated = self.bundle.audio_transformer.generate_from_encoded(
                encoded[:, start:end],
                source_mask[:, :, start:end],
                previous,
                max(0, steps),
                sample=sample,
                top_k=top_k,
                forbidden_ids=(
                    self.bundle.config["motion_start_id"],
                    self.bundle.config["motion_end_id"],
                    self.bundle.config["motion_pad_id"],
                ),
                generator=generator,
            )
            parts.append(generated if start == 0 else generated[:, overlap:])
            if end == features.shape[1]:
                break
            previous = generated[:, -overlap:]
            start = end - overlap
        tokens = torch.cat(parts, dim=1)
        if tokens.shape[1] != features.shape[1]:
            raise RuntimeError("TM2D audio token count does not match feature duration")
        motion, native_joints = self._decode(tokens)
        if max_frames is not None:
            motion = motion[:, :max_frames]
            native_joints = native_joints[:, :max_frames]
        joints = native_joints[0].cpu().numpy().astype(np.float32)
        return TM2DGenerationOutput(
            joints=joints,
            native_joints=joints,
            model_motion=motion[0].cpu().numpy().astype(np.float32),
            motion_tokens=tokens[0].cpu().numpy(),
            fps=self.bundle.fps,
            native_fps=self.bundle.fps,
            music_features=feature_array[0],
            music_beats=feature_array[0, :, BAILANDO_BEAT_CHANNEL] > 0.5,
        )

    def __call__(self, *args, task: str = "text-to-motion", **kwargs):
        if task in {"text-to-motion", "t2m"}:
            return self.infer_text_to_motion(*args, **kwargs)
        if task in {"music-to-dance", "m2d"}:
            return self.infer_music_to_dance(*args, **kwargs)
        raise ValueError(f"Unsupported TM2D task: {task!r}")


__all__ = ["TM2DGenerationOutput", "TM2DPipeline"]
