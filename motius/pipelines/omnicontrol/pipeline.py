"""OmniControl text-to-motion and 3D joint-position control pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from motius.models.omnicontrol.network import ClassifierFreeSampleModel
from motius.motion.representation.humanml import recover_from_ric
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


OMNICONTROL_MIN_FRAMES = 24
OMNICONTROL_MAX_FRAMES = 196
_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@PIPELINES.register_module()
class OmniControlPipeline(BasePipeline):
    """Generate HML263 motion from text and arbitrary 3D position hints.

    OmniControl's native observation is a world-space joint position at a
    selected frame. Full-frame temporal conditions are represented by selecting
    all 22 joints on the requested frames. Local rotations are not part of the
    method's native control interface.
    """

    BUNDLE_CLS = "motius.models.omnicontrol.OmniControlBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to(torch.device(device))
        return self

    @property
    def device(self):
        return self.bundle.device

    @staticmethod
    def clamp_length(length: int) -> int:
        return max(OMNICONTROL_MIN_FRAMES, min(OMNICONTROL_MAX_FRAMES, int(length)))

    @staticmethod
    def _indices_for_mode(
        mode: str,
        length: int,
        prefix_ratio: float,
        boundary_ratio: float,
        keyframes: Optional[Sequence[int]],
    ) -> List[int]:
        if mode in {"first_frame", "start_1f"}:
            return [0]
        if mode in {"first_last", "both_1f"}:
            return [0, length - 1]
        if mode in {"prefix", "pre20"}:
            return list(range(max(1, int(round(length * prefix_ratio)))))
        if mode in {"boundary", "mid80"}:
            count = max(1, int(round(length * boundary_ratio)))
            return list(range(count)) + list(range(max(count, length - count), length))
        if mode in {"keyframes", "adaptive_keyframes"}:
            if keyframes is None:
                raise ValueError("keyframes mode requires keyframe_indices")
            return sorted({max(0, min(length - 1, int(index))) for index in keyframes})
        if mode in {"trajectory", "dense"}:
            return list(range(length))
        if mode in {"none", "t2m"}:
            return []
        raise ValueError(f"unsupported OmniControl control mode: {mode}")

    def _build_hints(
        self,
        motions: Sequence[np.ndarray],
        lengths: Sequence[int],
        n_frames: int,
        control_mode: str,
        joint_indices: Optional[Iterable[int]],
        axes: str,
        keyframe_indices: Optional[Sequence[Sequence[int]]],
        prefix_ratio: float,
        boundary_ratio: float,
    ):
        motion = torch.zeros((len(lengths), n_frames, 263), device=self.device)
        for batch_index, (value, length) in enumerate(zip(motions, lengths)):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.ndim != 2 or value.shape[1] != 263:
                raise ValueError(f"motion {batch_index} must have shape (T,263), got {tuple(value.shape)}")
            motion[batch_index, :length] = value[:length]
        joints = recover_from_ric(motion, 22)

        selected_joints = list(range(22)) if joint_indices is None else sorted({int(x) for x in joint_indices})
        if not selected_joints or min(selected_joints) < 0 or max(selected_joints) >= 22:
            raise ValueError(f"joint_indices must be within [0,21], got {selected_joints}")
        axis_indices = sorted({_AXIS_TO_INDEX[axis] for axis in axes.lower()})
        if not axis_indices:
            raise ValueError("axes must contain at least one of x, y, z")

        axis_mask = torch.zeros(
            (len(lengths), n_frames, 22, 3), dtype=torch.bool, device=self.device
        )
        for batch_index, length in enumerate(lengths):
            keys = None if keyframe_indices is None else keyframe_indices[batch_index]
            frames = self._indices_for_mode(
                control_mode, length, prefix_ratio, boundary_ratio, keys
            )
            if frames:
                frame_tensor = torch.as_tensor(frames, device=self.device)
                joint_tensor = torch.as_tensor(selected_joints, device=self.device)
                axis_tensor = torch.as_tensor(axis_indices, device=self.device)
                axis_mask[batch_index][
                    frame_tensor[:, None, None],
                    joint_tensor[None, :, None],
                    axis_tensor[None, None, :],
                ] = True

        raw_mean = self.bundle.raw_mean.view(1, 1, 22, 3)
        raw_std = self.bundle.raw_std.view(1, 1, 22, 3)
        normalized = (joints - raw_mean) / raw_std
        hint = torch.where(axis_mask, normalized, torch.zeros_like(normalized))
        return (
            hint.reshape(len(lengths), n_frames, 66),
            axis_mask,
            axis_mask.any(dim=-1).any(dim=-1),
        )

    @torch.no_grad()
    def infer_control(
        self,
        captions: Sequence[str],
        motions: Sequence[np.ndarray],
        lengths: Optional[Sequence[int]] = None,
        control_mode: str = "first_last",
        joint_indices: Optional[Iterable[int]] = None,
        axes: str = "xyz",
        keyframe_indices: Optional[Sequence[Sequence[int]]] = None,
        prefix_ratio: float = 0.2,
        boundary_ratio: float = 0.1,
        guidance_param: Optional[float] = None,
        seed: int = 0,
        progress: bool = False,
    ) -> List[np.ndarray]:
        if len(captions) != len(motions):
            raise ValueError("captions and motions must have equal batch size")
        if lengths is None:
            lengths = [len(motion) for motion in motions]
        if len(lengths) != len(motions):
            raise ValueError("lengths and motions must have equal batch size")
        lengths = [self.clamp_length(length) for length in lengths]
        n_frames = max(lengths)
        if keyframe_indices is not None and len(keyframe_indices) != len(lengths):
            raise ValueError("keyframe_indices must have one sequence per sample")

        hint, axis_mask, frame_mask = self._build_hints(
            motions,
            lengths,
            n_frames,
            control_mode,
            joint_indices,
            axes,
            keyframe_indices,
            prefix_ratio,
            boundary_ratio,
        )
        valid = torch.arange(n_frames, device=self.device)[None] < torch.tensor(
            lengths, device=self.device
        )[:, None]
        scale = self.bundle.guidance_param if guidance_param is None else float(guidance_param)
        inner_model = self.bundle.net
        text_embed = inner_model.encode_text(list(captions))
        model = ClassifierFreeSampleModel(inner_model) if scale != 1.0 else inner_model
        model_kwargs = {
            "y": {
                "mask": valid[:, None, None, :],
                "lengths": torch.tensor(lengths, device=self.device),
                "text": list(captions),
                "text_embed": text_embed,
                "hint": hint,
                "hint_axis_mask": axis_mask,
                "hint_frame_mask": frame_mask,
                "scale": torch.full((len(lengths),), scale, device=self.device),
            }
        }

        fork_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            noise = torch.randn(
                (len(lengths), 263, 1, n_frames), device=self.device
            )
            sample = self.bundle.diffusion.p_sample_loop(
                model,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=progress,
            )
        sample = sample[:, :, 0, :].permute(0, 2, 1).contiguous()
        arrays = self.bundle.denormalize(sample).cpu().numpy().astype(np.float32)
        return [arrays[index, :length] for index, length in enumerate(lengths)]

    def infer_t2m(self, captions, lengths, **kwargs):
        blank = [np.zeros((int(length), 263), dtype=np.float32) for length in lengths]
        return self.infer_control(
            captions,
            blank,
            lengths=lengths,
            control_mode="none",
            **kwargs,
        )

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)


__all__ = ["OmniControlPipeline"]
