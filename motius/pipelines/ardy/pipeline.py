"""Offline and streaming inference for NVIDIA ARDY."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@dataclass
class ARDYStreamState:
    """Normalized ARDY history retained between streaming calls."""

    history: Optional[torch.Tensor] = None
    total_frames: int = 0


def _as_prompt_list(captions) -> list[str]:
    if isinstance(captions, str):
        return [captions]
    prompts = [str(value) for value in captions]
    if not prompts:
        raise ValueError("captions must contain at least one prompt")
    return prompts


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return value


def _to_tensor(value, *, dtype=torch.float32, device=None):
    if torch.is_tensor(value):
        output = value
    else:
        output = torch.as_tensor(value)
    if dtype is not None:
        output = output.to(dtype=dtype)
    if device is not None:
        output = output.to(device=device)
    return output


@PIPELINES.register_module()
class ARDYPipeline(BasePipeline):
    """ARDY T2M, online prompt, and kinematic-control pipeline."""

    BUNDLE_CLS = "motius.models.ardy.ARDYBundle"

    @property
    def model(self):
        return self.bundle.model

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @property
    def fps(self) -> float:
        return self.bundle.fps

    @property
    def representation(self) -> str:
        return "ardy_g1_414" if "g1" in self.model.skeleton.name.lower() else "ardy_core330"

    def _validate_steps(self, value: Optional[int]) -> int:
        maximum = int(self.model.diffusion.num_base_steps)
        steps = maximum if value is None else int(value)
        if not 1 <= steps <= maximum:
            raise ValueError(f"num_denoising_steps must be in [1, {maximum}], got {steps}")
        return steps

    def _history_frames(self, value: Optional[int]) -> int:
        patch = int(self.model.num_frames_per_token)
        if value is None:
            max_window = (int(10 * self.fps) // patch) * patch
            return max(patch, max_window - int(self.model.gen_horizon_len))
        value = int(value)
        if value < patch or value % patch:
            raise ValueError(f"history_frames must be a positive multiple of {patch}")
        return value

    def _text_inputs(
        self,
        prompts: Sequence[str],
        text_feat: Optional[torch.Tensor],
        text_pad_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if text_feat is None:
            if self.model.text_encoder is None:
                raise RuntimeError(
                    "This ARDY model was loaded without a text encoder. Pass both "
                    "text_feat and text_pad_mask, or load with the official LLM2Vec encoder."
                )
            return None, None
        text_feat = torch.as_tensor(text_feat, device=self.device)
        if text_feat.ndim != 3 or text_feat.shape[0] != len(prompts) or text_feat.shape[-1] != 4096:
            raise ValueError(
                "text_feat must have shape (batch, tokens, 4096) matching captions; "
                f"got {tuple(text_feat.shape)}"
            )
        if text_pad_mask is None:
            text_pad_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=self.device)
        else:
            text_pad_mask = torch.as_tensor(text_pad_mask, dtype=torch.bool, device=self.device)
        if tuple(text_pad_mask.shape) != tuple(text_feat.shape[:2]):
            raise ValueError("text_pad_mask must match text_feat's first two dimensions")
        return text_feat, text_pad_mask

    @torch.inference_mode()
    def generate(
        self,
        captions,
        lengths: Sequence[int] | int,
        *,
        constraints: Optional[Sequence[Any]] = None,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        num_denoising_steps: Optional[int] = None,
        cfg_weight: float | Tuple[float, float] = (2.0, 2.0),
        history_frames: Optional[int] = None,
        first_heading_angle: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        progress: bool = False,
        postprocess: bool = False,
        include_g1_qpos: bool = True,
        return_numpy: bool = True,
    ) -> Dict[str, Any]:
        prompts = _as_prompt_list(captions)
        if isinstance(lengths, int):
            lengths = [int(lengths)] * len(prompts)
        lengths = [int(value) for value in lengths]
        if len(lengths) != len(prompts) or min(lengths) < 1:
            raise ValueError("lengths must be positive and match captions")

        text_feat, text_pad_mask = self._text_inputs(prompts, text_feat, text_pad_mask)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        num_frames = max(lengths)
        pad_mask = torch.arange(num_frames, device=self.device)[None] < lengths_tensor[:, None]
        if first_heading_angle is None:
            first_heading_angle = torch.zeros(len(prompts), device=self.device)
        else:
            first_heading_angle = torch.as_tensor(first_heading_angle, device=self.device)

        observed_motion = motion_mask = None
        constraint_list = list(constraints or [])
        if constraint_list:
            observed_motion, motion_mask = self.model.motion_rep.create_conditions_from_constraints_batched(
                constraint_list,
                lengths_tensor,
                to_normalize=True,
                device=str(self.device),
            )

        fork_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            if seed is not None:
                torch.manual_seed(int(seed))
            features = self.model(
                prompts,
                num_frames,
                num_denoising_steps=self._validate_steps(num_denoising_steps),
                pad_mask=pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                crop_history_length=self._history_frames(history_frames),
                progress_bar=tqdm if progress else (lambda value: value),
            )

        output = self.model.motion_rep.inverse(features, is_normalized=True)
        if postprocess:
            if "g1" in self.model.skeleton.name.lower():
                raise ValueError("ARDY's official G1 inference disables motion postprocessing")
            from motius.models.ardy.network.postprocess import post_process_motion

            corrected = post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                self.model.skeleton,
                constraint_lst=constraint_list or None,
            )
            output.update(corrected)

        output["features"] = features
        output["lengths"] = lengths_tensor
        output["fps"] = self.fps
        output["representation"] = self.representation
        if include_g1_qpos and "g1" in self.model.skeleton.name.lower():
            from motius.models.ardy.network.exports.mujoco import MujocoQposConverter

            output["qpos"] = MujocoQposConverter(self.model.skeleton).dict_to_qpos(
                output, device=str(self.device), numpy=False
            )
        return _to_numpy(output) if return_numpy else output

    def text_to_motion(self, caption: str, num_frames: int, **kwargs):
        return self.generate(caption, num_frames, **kwargs)

    def infer_t2m(self, captions, lengths, **kwargs):
        return self.generate(captions, lengths, **kwargs)

    @torch.inference_mode()
    def stream_step(
        self,
        caption: str,
        state: Optional[ARDYStreamState] = None,
        *,
        constraints: Optional[Sequence[Any]] = None,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        num_denoising_steps: Optional[int] = None,
        cfg_weight: float | Tuple[float, float] = (2.0, 2.0),
        history_frames: Optional[int] = None,
        seed: Optional[int] = None,
        return_numpy: bool = True,
    ):
        """Generate one ARDY horizon and return ``(new_motion, new_state)``."""
        state = state or ARDYStreamState()
        prompts = [str(caption)]
        text_feat, text_pad_mask = self._text_inputs(prompts, text_feat, text_pad_mask)
        history = state.history
        if history is not None:
            history = history.to(self.device)
        history_len = 0 if history is None else int(history.shape[1])
        horizon = int(self.model.gen_horizon_len)
        num_frames = history_len + horizon

        motion_mask = observed_motion = None
        constraint_list = list(constraints or [])
        if constraint_list:
            observed_motion, motion_mask = self.model.motion_rep.create_conditions_from_constraints_batched(
                constraint_list,
                torch.tensor([num_frames], device=self.device),
                to_normalize=True,
                device=str(self.device),
            )

        fork_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            if seed is not None:
                torch.manual_seed(int(seed))
            features = self.model.autoregressive_step(
                num_frames=num_frames,
                num_denoising_steps=self._validate_steps(num_denoising_steps),
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                texts=prompts,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                init_history_sequence=history,
            )

        new_features = features[:, history_len:]
        decoded = self.model.motion_rep.inverse(new_features, is_normalized=True)
        decoded["features"] = new_features
        decoded["fps"] = self.fps
        decoded["representation"] = self.representation
        keep = self._history_frames(history_frames)
        new_state = ARDYStreamState(
            history=features[:, -keep:].detach(),
            total_frames=state.total_frames + int(new_features.shape[1]),
        )
        return (_to_numpy(decoded) if return_numpy else decoded), new_state

    def load_constraints(self, path_or_data, *, device=None, dtype=torch.float32):
        from motius.models.ardy.network.constraints import load_constraints_lst

        return load_constraints_lst(
            str(path_or_data) if isinstance(path_or_data, Path) else path_or_data,
            self.model.skeleton,
            device=device,
            dtype=dtype,
        )

    def root2d_constraint(
        self,
        frame_indices,
        root_2d,
        global_root_heading: Optional[Any] = None,
        *,
        device=None,
    ):
        """Create sparse root-path and optional heading constraints."""
        from motius.models.ardy.network.constraints import Root2DConstraintSet

        return Root2DConstraintSet(
            self.model.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(root_2d, device=device),
            global_root_heading=(
                None
                if global_root_heading is None
                else _to_tensor(global_root_heading, device=device)
            ),
        )

    def fullbody_keyframe_constraint(
        self,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        root_2d: Optional[Any] = None,
        *,
        device=None,
    ):
        """Create full-body position keyframes in the native skeleton space."""
        from motius.models.ardy.network.constraints import FullBodyConstraintSet

        return FullBodyConstraintSet(
            self.model.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            root_2d=None if root_2d is None else _to_tensor(root_2d, device=device),
        )

    def end_effector_constraint(
        self,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        joint_names: Sequence[str],
        root_2d: Optional[Any] = None,
        *,
        device=None,
    ):
        """Create sparse native-joint position and rotation constraints."""
        from motius.models.ardy.network.constraints import EndEffectorConstraintSet

        return EndEffectorConstraintSet(
            self.model.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            None if root_2d is None else _to_tensor(root_2d, device=device),
            joint_names=list(joint_names),
        )


__all__ = ["ARDYPipeline", "ARDYStreamState"]
