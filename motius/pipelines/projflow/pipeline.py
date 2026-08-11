"""Native ProjFlow inference pipeline."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from motius.motion.representation.humanml import (
    hml263_to_joints,
    joints_to_hml263,
)
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


PROJFLOW_MAX_FRAMES = 196
_AXES = {"x": 0, "y": 1, "z": 2}


@PIPELINES.register_module()
class ProjFlowPipeline(BasePipeline):
    """Exact spatial motion control with the released ACMDM Flow prior."""

    BUNDLE_CLS = "motius.models.projflow.ProjFlowBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to(torch.device(device))
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @staticmethod
    def _validate_lengths(lengths: Sequence[int]) -> list[int]:
        values = [int(length) for length in lengths]
        if not values or min(values) < 1:
            raise ValueError("ProjFlow lengths must contain positive frame counts")
        if max(values) > PROJFLOW_MAX_FRAMES:
            raise ValueError(
                f"ProjFlow supports at most {PROJFLOW_MAX_FRAMES} frames, "
                f"got {max(values)}"
            )
        return values

    @staticmethod
    def _as_joints(motion: np.ndarray) -> np.ndarray:
        value = np.asarray(motion, dtype=np.float32)
        if value.ndim == 2 and value.shape[1] == 263:
            return hml263_to_joints(value).astype(np.float32)
        if value.ndim == 3 and value.shape[1:] == (22, 3):
            return value
        raise ValueError(
            "ProjFlow source motion must have shape (T,263) or (T,22,3), "
            f"got {value.shape}"
        )

    @staticmethod
    def _frames_for_mode(
        mode: str,
        length: int,
        *,
        prefix_ratio: float,
        boundary_ratio: float,
        keyframes: Optional[Sequence[int]],
    ) -> list[int]:
        key = str(mode).lower().removesuffix("_uncond")
        if key in {"first_frame", "start_1f"}:
            return [0]
        if key in {"first_last", "both_1f", "loop"}:
            return [0, length - 1]
        if key in {"prefix", "pre20"}:
            return list(range(max(1, int(round(length * prefix_ratio)))))
        if key in {"boundary", "mib10", "mid80"}:
            count = max(1, int(round(length * boundary_ratio)))
            return list(range(count)) + list(range(max(count, length - count), length))
        if key in {"keyframes", "adaptive_keyframes", "waypoints", "sparse"}:
            if keyframes is None:
                raise ValueError(f"{mode} requires keyframe_indices")
            return sorted({max(0, min(length - 1, int(index))) for index in keyframes})
        if key in {"trajectory", "dense", "joints", "body_part"}:
            return list(range(length))
        if key in {"none", "t2m"}:
            return []
        raise ValueError(f"Unsupported ProjFlow control mode: {mode}")

    def _prepare_inputs(
        self,
        motions: Sequence[np.ndarray],
        lengths: Optional[Sequence[int]],
    ) -> tuple[list[np.ndarray], list[int], int]:
        joints = [self._as_joints(motion) for motion in motions]
        if lengths is None:
            lengths = [len(motion) for motion in joints]
        if len(lengths) != len(joints):
            raise ValueError("lengths and motions must have equal batch size")
        values = self._validate_lengths(lengths)
        for index, (motion, length) in enumerate(zip(joints, values)):
            if len(motion) < length:
                raise ValueError(
                    f"source motion {index} has {len(motion)} frames, "
                    f"requested {length}"
                )
        return joints, values, max(values)

    def _build_control(
        self,
        joints: Sequence[np.ndarray],
        lengths: Sequence[int],
        frames: int,
        *,
        control_mode: str,
        joint_indices: Optional[Iterable[int]],
        axes: str,
        keyframe_indices: Optional[Sequence[Sequence[int]]],
        prefix_ratio: float,
        boundary_ratio: float,
        position_mask,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(joints)
        world = torch.zeros((batch, frames, 22, 3), device=self.device)
        for index, (motion, length) in enumerate(zip(joints, lengths)):
            world[index, :length] = torch.as_tensor(
                motion[:length], dtype=torch.float32, device=self.device
            )
        normalized = self.bundle.normalize_joints(world)

        if position_mask is not None:
            mask = torch.as_tensor(position_mask, dtype=torch.bool, device=self.device)
            expected = (batch, frames, 22, 3)
            if tuple(mask.shape) != expected:
                raise ValueError(f"position_mask must have shape {expected}, got {tuple(mask.shape)}")
        else:
            if joint_indices is None:
                mode = str(control_mode).lower().removesuffix("_uncond")
                selected = (
                    [0]
                    if mode in {"trajectory", "dense", "waypoints", "sparse"}
                    else list(range(22))
                )
            else:
                selected = sorted({int(value) for value in joint_indices})
            if not selected or min(selected) < 0 or max(selected) >= 22:
                raise ValueError(f"joint_indices must be within [0,21], got {selected}")
            axis_indices = sorted({_AXES[value] for value in str(axes).lower() if value in _AXES})
            if not axis_indices:
                raise ValueError("axes must contain at least one of x, y, z")
            if keyframe_indices is not None and len(keyframe_indices) != batch:
                raise ValueError("keyframe_indices must contain one sequence per sample")
            mask = torch.zeros((batch, frames, 22, 3), dtype=torch.bool, device=self.device)
            for batch_index, length in enumerate(lengths):
                keys = None if keyframe_indices is None else keyframe_indices[batch_index]
                selected_frames = self._frames_for_mode(
                    control_mode,
                    length,
                    prefix_ratio=prefix_ratio,
                    boundary_ratio=boundary_ratio,
                    keyframes=keys,
                )
                if selected_frames:
                    frame_tensor = torch.as_tensor(selected_frames, device=self.device)
                    joint_tensor = torch.as_tensor(selected, device=self.device)
                    axis_tensor = torch.as_tensor(axis_indices, device=self.device)
                    mask[batch_index][
                        frame_tensor[:, None, None],
                        joint_tensor[None, :, None],
                        axis_tensor[None, None, :],
                    ] = True
        valid = torch.arange(frames, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        mask &= valid[:, :, None, None]
        control = torch.where(mask, normalized, torch.zeros_like(normalized))
        return control.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2).float()

    @staticmethod
    def _as_hml263(joints: np.ndarray) -> np.ndarray:
        if len(joints) < 1:
            raise ValueError("Cannot encode an empty joint sequence as HML263")
        padded = np.concatenate([joints, joints[-1:]], axis=0)
        return joints_to_hml263(padded)

    @staticmethod
    def _format_outputs(joints: torch.Tensor, lengths: Sequence[int], return_format: str):
        arrays = joints.detach().cpu().numpy().astype(np.float32)
        outputs = [arrays[index, :length] for index, length in enumerate(lengths)]
        key = return_format.lower().replace("-", "").replace("_", "")
        if key in {"joints", "joints66", "smpl22joints"}:
            return outputs
        if key in {"hml263", "humanml263", "humanml3d263"}:
            return [ProjFlowPipeline._as_hml263(value) for value in outputs]
        raise ValueError(f"Unsupported ProjFlow return_format: {return_format}")

    @torch.no_grad()
    def infer_control(
        self,
        captions: Sequence[str],
        motions: Sequence[np.ndarray],
        *,
        lengths: Optional[Sequence[int]] = None,
        control_mode: str = "first_last",
        joint_indices: Optional[Iterable[int]] = None,
        axes: str = "xyz",
        keyframe_indices: Optional[Sequence[Sequence[int]]] = None,
        prefix_ratio: float = 0.2,
        boundary_ratio: float = 0.1,
        position_mask=None,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        seed: int = 0,
        return_format: str = "joints",
        use_projflow: bool = True,
    ):
        if len(captions) != len(motions):
            raise ValueError("captions and motions must have equal batch size")
        source_joints, lengths, frames = self._prepare_inputs(motions, lengths)
        control, mask = self._build_control(
            source_joints,
            lengths,
            frames,
            control_mode=control_mode,
            joint_indices=joint_indices,
            axes=axes,
            keyframe_indices=keyframe_indices,
            prefix_ratio=prefix_ratio,
            boundary_ratio=boundary_ratio,
            position_mask=position_mask,
        )
        scale = self.bundle.guidance_scale if guidance_scale is None else float(guidance_scale)
        steps = self.bundle.num_steps if num_steps is None else int(num_steps)
        model = self.bundle.net
        conditions = model.encode_text(list(captions)).float()
        attention = (
            (torch.arange(frames, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None])
            .unsqueeze(-1)
            .repeat(1, 1, model.patches_per_frame)
            .flatten(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        fork_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            noise = torch.randn((len(lengths), 3, frames, 22), device=self.device)
            if scale != 1.0:
                conditions = torch.cat([conditions, torch.zeros_like(conditions)], dim=0)
                attention = attention.repeat(2, 1, 1, 1)
                noise = torch.cat([noise, noise], dim=0)
                control = control.repeat(2, 1, 1, 1)
                mask = mask.repeat(2, 1, 1, 1)
            sample = model.gen_diffusion.sample_projflow(
                num_steps=steps,
                use_projflow=use_projflow,
            )(
                noise,
                model.forward_with_CFG,
                conds=conditions,
                attention_mask=attention,
                cfg=scale,
                A=mask,
                y=control,
            )[-1]
        if scale != 1.0:
            sample, _ = sample.chunk(2, dim=0)
        world = self.bundle.denormalize_joints(sample.permute(0, 2, 3, 1))
        return self._format_outputs(world, lengths, return_format)

    def infer_temporal_motion_completion(self, source_motion, generation_mask=None, **kwargs):
        """Run temporal completion using the standard task API.

        ``source_motion`` may be one motion or a batch. For exact arbitrary
        masks, pass ``position_mask`` through ``kwargs``. ``generation_mask``
        follows the framework convention (True means generate); it is converted
        to an observed-position mask over all joints and axes.
        """

        motions = source_motion if isinstance(source_motion, (list, tuple)) else [source_motion]
        if generation_mask is not None and "position_mask" not in kwargs:
            value = np.asarray(generation_mask, dtype=bool)
            if value.ndim == 1:
                value = value[None]
            kwargs["position_mask"] = np.broadcast_to(
                (~value)[:, :, None, None],
                (len(motions), value.shape[1], 22, 3),
            ).copy()
        captions = kwargs.pop("captions", [""] * len(motions))
        return self.infer_control(captions, motions, **kwargs)

    def infer_kinematic_motion_control(self, captions, source_motion, **kwargs):
        motions = source_motion if isinstance(source_motion, (list, tuple)) else [source_motion]
        texts = captions if isinstance(captions, (list, tuple)) else [captions]
        return self.infer_control(texts, motions, **kwargs)

    infer_part_level_motion_control = infer_kinematic_motion_control

    def __call__(self, captions, source_motion, **kwargs):
        return self.infer_kinematic_motion_control(captions, source_motion, **kwargs)


__all__ = ["ProjFlowPipeline"]
