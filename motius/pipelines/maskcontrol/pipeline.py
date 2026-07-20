"""MaskControl text, temporal, body-part, and sequential generation pipeline."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from motius.models.maskcontrol.network import (
    BODY_PART_JOINTS,
    CONTROL_JOINT_IDS,
    relative_hml263_positions,
)
from motius.models.momask.network import estimate_token_lengths
from motius.motion.representation.humanml import recover_from_ric
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


MASKCONTROL_UNIT_LENGTH = 4
MASKCONTROL_T2M_MIN_FRAMES = 40
MASKCONTROL_T2M_MAX_FRAMES = 196
MASKCONTROL_MAX_FRAMES = 392


def _round_up_four(value: int) -> int:
    return ((int(value) + 3) // 4) * 4


@PIPELINES.register_module()
class MaskControlPipeline(BasePipeline):
    """Motius-native inference for the released MaskControl checkpoint.

    All public task methods return physical-scale HumanML3D-263 arrays at
    20 fps.  Spatial conditions always use a separate boolean mask, so a valid
    target coordinate at the origin is never confused with an absent target.
    """

    BUNDLE_CLS = "motius.models.maskcontrol.MaskControlBundle"

    def __init__(self, bundle, device: Optional[str | torch.device] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @staticmethod
    def _validate_lengths(
        lengths: Sequence[int],
        *,
        t2m_limits: bool,
    ) -> list[int]:
        output = []
        for value in lengths:
            value = int(value)
            if t2m_limits:
                value = min(
                    MASKCONTROL_T2M_MAX_FRAMES,
                    max(MASKCONTROL_T2M_MIN_FRAMES, value),
                )
            elif value < 4:
                raise ValueError("MaskControl sequences require at least 4 frames")
            value = _round_up_four(value)
            if value > MASKCONTROL_MAX_FRAMES:
                raise ValueError("MaskControl supports at most 392 frames")
            output.append(value)
        return output

    @staticmethod
    def _canvas_frames(lengths: Sequence[int]) -> int:
        return 196 if max(lengths) <= 196 else 392

    def _empty_controls(
        self, batch_size: int, canvas_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros(
            (batch_size, canvas_frames, 22, 3),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros(
            (batch_size, canvas_frames, 22),
            dtype=torch.bool,
            device=self.device,
        )
        return targets, mask

    def _generate_batch(
        self,
        captions: Optional[Sequence[str]],
        lengths: Sequence[int],
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        seed: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_lengths = torch.as_tensor(
            lengths, dtype=torch.long, device=self.device
        )
        fork_devices = (
            [self.device.index or 0] if self.device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            normalized, physical = self.bundle.generate(
                None if captions is None else list(captions),
                frame_lengths,
                targets,
                target_mask,
                **kwargs,
            )
        return normalized, physical

    def infer_control(
        self,
        captions: Optional[Sequence[str]],
        lengths: Sequence[int],
        target_joints,
        target_mask,
        *,
        seed: int = 0,
        time_steps: int = 10,
        cond_scale: float = 4.0,
        residual_cond_scale: float = 5.0,
        temperature: float = 1.0,
        residual_temperature: float = 1.0,
        each_iterations: int = 100,
        final_iterations: int = 600,
        each_lr: float = 0.06,
        final_lr: float = 0.06,
        relative_control: bool = False,
        use_residual: bool = True,
        return_normalized: bool = False,
    ) -> list[np.ndarray]:
        """Generate from arbitrary anchor-joint positions and frame masks.

        Args:
            captions: one prompt per sample, or ``None`` for unconditional
                composition.
            lengths: requested frame lengths; values are rounded up to a
                multiple of four and capped at 392 frames.
            target_joints: ``(B,T,22,3)`` coordinates.
            target_mask: explicit ``(B,T,22)`` boolean observation mask.

        Only the checkpoint's six trained anchors (pelvis, feet, head, and
        wrists) affect generation. A mask selecting another joint is rejected
        instead of being silently ignored.
        """

        lengths = self._validate_lengths(lengths, t2m_limits=False)
        batch_size = len(lengths)
        if captions is not None and len(captions) != batch_size:
            raise ValueError("captions and lengths must have equal length")
        canvas = self._canvas_frames(lengths)
        targets, mask = self._empty_controls(batch_size, canvas)
        raw_targets = torch.as_tensor(
            target_joints, dtype=torch.float32, device=self.device
        )
        raw_mask = torch.as_tensor(
            target_mask, dtype=torch.bool, device=self.device
        )
        if raw_targets.ndim != 4 or raw_targets.shape[0] != batch_size or raw_targets.shape[2:] != (22, 3):
            raise ValueError("target_joints must have shape (B,T,22,3)")
        if raw_mask.shape != raw_targets.shape[:-1]:
            raise ValueError("target_mask must have shape (B,T,22)")
        if raw_targets.shape[1] > canvas:
            raise ValueError("target_joints exceed the selected MaskControl canvas")
        unsupported = raw_mask.clone()
        unsupported[..., list(CONTROL_JOINT_IDS)] = False
        if unsupported.any():
            bad = torch.nonzero(unsupported, as_tuple=False)[0]
            raise ValueError(
                "released MaskControl weights only support joints "
                f"{CONTROL_JOINT_IDS}; found joint {int(bad[2])}"
            )
        targets[:, : raw_targets.shape[1]] = raw_targets
        mask[:, : raw_mask.shape[1]] = raw_mask
        normalized, physical = self._generate_batch(
            captions,
            lengths,
            targets,
            mask,
            seed=seed,
            time_steps=time_steps,
            cond_scale=cond_scale,
            residual_cond_scale=residual_cond_scale,
            temperature=temperature,
            residual_temperature=residual_temperature,
            use_control=True,
            each_iterations=each_iterations,
            final_iterations=final_iterations,
            each_lr=each_lr,
            final_lr=final_lr,
            relative_control=relative_control,
            use_residual=use_residual,
        )
        values = normalized if return_normalized else physical
        return [
            values[index, :length].detach().cpu().numpy().astype(np.float32)
            for index, length in enumerate(lengths)
        ]

    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
        *,
        seed: int = 0,
        time_steps: int = 10,
        cond_scale: float = 4.0,
        residual_cond_scale: float = 5.0,
        temperature: float = 1.0,
        residual_temperature: float = 1.0,
        return_normalized: bool = False,
    ) -> list[np.ndarray]:
        captions = [str(value) for value in captions]
        if not captions:
            raise ValueError("captions must not be empty")
        if lengths is None:
            if self.bundle.length_estimator is None:
                raise RuntimeError(
                    "lengths=None requires the bundled length estimator"
                )
            tokens = estimate_token_lengths(
                self.bundle.control_model,
                self.bundle.length_estimator,
                captions,
            )
            lengths = [int(value) * 4 for value in tokens.tolist()]
        if len(lengths) != len(captions):
            raise ValueError("captions and lengths must have equal length")
        lengths = self._validate_lengths(lengths, t2m_limits=True)
        return self._infer_uncontrolled(
            captions,
            lengths,
            seed=seed,
            time_steps=time_steps,
            cond_scale=cond_scale,
            residual_cond_scale=residual_cond_scale,
            temperature=temperature,
            residual_temperature=residual_temperature,
            return_normalized=return_normalized,
        )

    def _infer_uncontrolled(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        *,
        seed: int,
        return_normalized: bool = False,
        time_steps: int = 10,
        cond_scale: float = 4.0,
        residual_cond_scale: float = 5.0,
        temperature: float = 1.0,
        residual_temperature: float = 1.0,
        use_residual: bool = True,
    ) -> list[np.ndarray]:
        targets, mask = self._empty_controls(
            len(lengths), self._canvas_frames(lengths)
        )
        normalized, physical = self._generate_batch(
            captions,
            lengths,
            targets,
            mask,
            seed=seed,
            time_steps=time_steps,
            cond_scale=cond_scale,
            residual_cond_scale=residual_cond_scale,
            temperature=temperature,
            residual_temperature=residual_temperature,
            use_control=False,
            each_iterations=0,
            final_iterations=0,
            use_residual=use_residual,
        )
        values = normalized if return_normalized else physical
        return [
            values[index, :length].detach().cpu().numpy().astype(np.float32)
            for index, length in enumerate(lengths)
        ]

    @staticmethod
    def temporal_frame_indices(
        mode: str,
        length: int,
        *,
        prefix_ratio: float = 0.2,
        boundary_ratio: float = 0.1,
        keyframes: Optional[Iterable[int]] = None,
    ) -> list[int]:
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
            return sorted({max(0, min(length - 1, int(value))) for value in keyframes})
        if mode in {"trajectory", "dense"}:
            return list(range(length))
        if mode in {"none", "t2m"}:
            return []
        raise ValueError(f"unsupported temporal control mode: {mode}")

    def infer_temporal(
        self,
        captions: Optional[Sequence[str]],
        condition_motions: Sequence[np.ndarray],
        *,
        lengths: Optional[Sequence[int]] = None,
        mode: str = "first_last",
        keyframe_indices: Optional[Sequence[Sequence[int]]] = None,
        prefix_ratio: float = 0.2,
        boundary_ratio: float = 0.1,
        seed: int = 0,
        **kwargs,
    ) -> list[np.ndarray]:
        """Temporal conditioning from physical HumanML3D-263 motions."""

        kwargs.setdefault("time_steps", 10)
        kwargs.setdefault("cond_scale", 4.0)
        kwargs.setdefault("residual_cond_scale", 5.0)
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("residual_temperature", 1.0)
        kwargs.setdefault("each_iterations", 100)
        kwargs.setdefault("final_iterations", 600)
        kwargs.setdefault("each_lr", 0.06)
        kwargs.setdefault("final_lr", 0.06)
        if captions is not None and len(captions) != len(condition_motions):
            raise ValueError("captions and condition_motions must have equal length")
        if lengths is None:
            lengths = [len(value) for value in condition_motions]
        lengths = self._validate_lengths(lengths, t2m_limits=True)
        canvas = self._canvas_frames(lengths)
        targets, mask = self._empty_controls(len(lengths), canvas)
        if keyframe_indices is not None and len(keyframe_indices) != len(lengths):
            raise ValueError("keyframe_indices must contain one list per sample")

        for index, (motion, length) in enumerate(zip(condition_motions, lengths)):
            motion = torch.as_tensor(
                motion, dtype=torch.float32, device=self.device
            )
            if motion.ndim != 2 or motion.shape[-1] != 263:
                raise ValueError("condition motions must have shape (T,263)")
            available = min(length, len(motion))
            padded = torch.zeros((1, length, 263), device=self.device)
            padded[0, :available] = motion[:available]
            joints = recover_from_ric(padded, 22)[0]
            keys = None if keyframe_indices is None else keyframe_indices[index]
            frames = self.temporal_frame_indices(
                mode,
                available,
                prefix_ratio=prefix_ratio,
                boundary_ratio=boundary_ratio,
                keyframes=keys,
            )
            if frames:
                frame_tensor = torch.as_tensor(frames, device=self.device)
                joint_tensor = torch.as_tensor(CONTROL_JOINT_IDS, device=self.device)
                targets[index][frame_tensor[:, None], joint_tensor[None, :]] = joints[
                    frame_tensor[:, None], joint_tensor[None, :]
                ]
                mask[index][frame_tensor[:, None], joint_tensor[None, :]] = True

        normalized, physical = self._generate_batch(
            captions,
            lengths,
            targets,
            mask,
            seed=seed,
            use_control=True,
            use_residual=True,
            **kwargs,
        )
        return [
            physical[index, :length].detach().cpu().numpy().astype(np.float32)
            for index, length in enumerate(lengths)
        ]

    @staticmethod
    def _parse_timeline_entry(entry) -> tuple[tuple[str, ...], str, int, int]:
        if isinstance(entry, dict):
            parts = entry.get("parts", entry.get("part"))
            text = entry.get("text", entry.get("caption"))
            start = entry.get("start")
            end = entry.get("end")
        else:
            if len(entry) != 3:
                raise ValueError("timeline tuples must be (parts, text, [start,end])")
            parts, text, interval = entry
            start, end = interval
        if isinstance(parts, str):
            parts = (parts,)
        else:
            parts = tuple(str(value) for value in parts)
        normalized_parts = tuple(value.lower().replace(" ", "_") for value in parts)
        unknown = [part for part in normalized_parts if part not in BODY_PART_JOINTS]
        if unknown:
            raise ValueError(f"unknown MaskControl body parts: {unknown}")
        return normalized_parts, str(text), int(start), int(end)

    def infer_body_part(
        self,
        timeline: Sequence,
        *,
        length: Optional[int] = None,
        seed: int = 0,
        each_iterations: int = 100,
        final_iterations: int = 600,
        **kwargs,
    ) -> np.ndarray:
        """Generate one body-part timeline using the paper's iterative recipe."""

        entries = [self._parse_timeline_entry(value) for value in timeline]
        if not entries:
            raise ValueError("timeline must not be empty")
        if length is None:
            length = max(value[3] for value in entries)
        rounded_length = self._validate_lengths([length], t2m_limits=False)[0]
        for parts, text, start, end in entries:
            if start < 0 or end <= start or end > length:
                raise ValueError(f"invalid body-part interval [{start},{end})")

        first_length = self._validate_lengths(
            [entries[0][3]], t2m_limits=False
        )[0]
        first_targets, first_mask = self._empty_controls(
            1, self._canvas_frames([first_length])
        )
        _, first_physical = self._generate_batch(
            [entries[0][1]],
            [first_length],
            first_targets,
            first_mask,
            seed=seed,
            use_control=False,
            use_residual=False,
            each_iterations=0,
            final_iterations=0,
            **kwargs,
        )
        current = first_physical[0]
        previous = [entries[0]]
        for entry_index, (_, text, _, _) in enumerate(entries[1:], start=1):
            motion = torch.as_tensor(current, device=self.device).unsqueeze(0)
            joints = recover_from_ric(motion, 22)[0]
            canvas = self._canvas_frames([rounded_length])
            targets, mask = self._empty_controls(1, canvas)
            available = min(canvas, len(joints))
            targets[0, :available] = joints[:available]
            for parts, _, start, end in previous:
                joint_ids = sorted(
                    {
                        joint
                        for part in parts
                        for joint in BODY_PART_JOINTS[part]
                    }
                )
                mask[0, start:end, joint_ids] = True
            _, physical = self._generate_batch(
                [text],
                [rounded_length],
                targets,
                mask,
                seed=seed + entry_index,
                use_control=True,
                use_residual=False,
                each_iterations=each_iterations,
                final_iterations=final_iterations,
                **kwargs,
            )
            current = physical[0, :rounded_length].detach().cpu().numpy().astype(np.float32)
            previous.append(entries[entry_index])
        return current[:length]

    def infer_sequential(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        *,
        transition_padding: int = 5,
        seed: int = 0,
        each_iterations: int = 300,
        final_iterations: int = 300,
        **kwargs,
    ) -> np.ndarray:
        """Zero-shot ordered composition on a single <=392-frame canvas.

        Each segment is first generated independently. Its six anchor-joint
        relative controls are then placed on the global timeline, leaving a
        short unconstrained band at internal boundaries. A final unconditional
        MaskControl pass composes the complete motion.
        """

        captions = [str(value) for value in captions]
        if len(captions) != len(lengths) or not captions:
            raise ValueError("captions and lengths must be non-empty and equal")
        raw_lengths = [int(value) for value in lengths]
        if min(raw_lengths) < 4:
            raise ValueError("sequential segments require at least four frames")
        total = sum(raw_lengths)
        rounded_total = self._validate_lengths([total], t2m_limits=False)[0]
        segment_lengths = self._validate_lengths(raw_lengths, t2m_limits=False)

        segment_normalized = self._infer_uncontrolled(
            captions,
            segment_lengths,
            seed=seed,
            return_normalized=True,
        )
        canvas = self._canvas_frames([rounded_total])
        targets, mask = self._empty_controls(1, canvas)
        cursor = 0
        anchor_ids = list(CONTROL_JOINT_IDS)
        for index, (motion, raw_length) in enumerate(
            zip(segment_normalized, raw_lengths)
        ):
            relative = relative_hml263_positions(
                torch.as_tensor(motion, device=self.device).unsqueeze(0)
            )[0]
            start = cursor
            end = cursor + raw_length
            control_start = start + (transition_padding if index > 0 else 0)
            control_end = end - (
                transition_padding if index < len(raw_lengths) - 1 else 0
            )
            if control_end > control_start:
                local_start = control_start - start
                local_end = control_end - start
                targets[0, control_start:control_end, anchor_ids] = relative[
                    local_start:local_end, anchor_ids
                ]
                mask[0, control_start:control_end, anchor_ids] = True
            cursor = end

        _, physical = self._generate_batch(
            None,
            [rounded_total],
            targets,
            mask,
            seed=seed + len(captions),
            use_control=True,
            use_residual=False,
            each_iterations=each_iterations,
            final_iterations=final_iterations,
            relative_control=True,
            **kwargs,
        )
        return physical[0, :total].detach().cpu().numpy().astype(np.float32)

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)


__all__ = ["MaskControlPipeline"]
