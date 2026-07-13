"""CondMDI text-to-motion and motion-control pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from motius.models.condmdi.network import (
    ClassifierFreeSampleModel,
    absolute_to_relative,
    build_observation_mask,
    relative_to_absolute,
)
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


CONDMDI_MIN_FRAMES = 24
CONDMDI_MAX_FRAMES = 196


@PIPELINES.register_module()
class CondMDIPipeline(BasePipeline):
    """Flexible HumanML3D motion synthesis with frame/joint controls."""

    BUNDLE_CLS = "motius.models.condmdi.CondMDIBundle"

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
        length = max(CONDMDI_MIN_FRAMES, min(CONDMDI_MAX_FRAMES, int(length)))
        return max(CONDMDI_MIN_FRAMES, (length // 4) * 4)

    def _prepare_inputs(self, motions, lengths):
        lengths = [self.clamp_length(x) for x in lengths]
        n_frames = max(lengths)
        batch = torch.zeros((len(lengths), n_frames, 263), dtype=torch.float32, device=self.device)
        if motions is not None:
            if len(motions) != len(lengths):
                raise ValueError("motions and lengths must have equal batch size")
            for index, (motion, length) in enumerate(zip(motions, lengths)):
                value = torch.as_tensor(motion, dtype=torch.float32, device=self.device)
                if value.ndim != 2 or value.shape[1] != 263:
                    raise ValueError(f"motion {index} must have shape (T,263), got {tuple(value.shape)}")
                batch[index, :length] = value[:length]
        return relative_to_absolute(batch), lengths, n_frames

    @torch.no_grad()
    def _sample(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        motions=None,
        observation_mask: Optional[torch.Tensor] = None,
        control_mode: str = "none",
        transition_length: int = 10,
        feature_mode: str = "pos_rot_vel",
        joint_indices: Optional[Iterable[int]] = None,
        keyframe_indices: Optional[Sequence[Sequence[int]]] = None,
        guidance_param: Optional[float] = None,
        impute: bool = True,
        seed: int = 0,
        progress: bool = False,
        return_absolute: bool = False,
    ) -> List[np.ndarray]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal batch size")
        absolute, lengths, n_frames = self._prepare_inputs(motions, lengths)
        normalized = self.bundle.normalize_absolute(absolute)

        if observation_mask is None:
            obs_mask = build_observation_mask(
                lengths,
                n_frames,
                mode=control_mode,
                transition_length=transition_length,
                feature_mode=feature_mode,
                joint_indices=joint_indices,
                keyframe_indices=keyframe_indices,
            )
        else:
            obs_mask = torch.as_tensor(observation_mask, dtype=torch.bool)
        expected_shape = (len(lengths), 263, 1, n_frames)
        if tuple(obs_mask.shape) != expected_shape:
            raise ValueError(f"observation_mask must have shape {expected_shape}, got {tuple(obs_mask.shape)}")
        obs_mask = obs_mask.to(self.device)
        obs_x0 = normalized.permute(0, 2, 1).unsqueeze(2).contiguous()

        valid = torch.arange(n_frames, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        scale = self.bundle.guidance_param if guidance_param is None else float(guidance_param)
        options = {
            "mask": valid[:, None, None, :],
            "lengths": torch.tensor(lengths, device=self.device),
            "text": list(captions),
            "diffusion_steps": self.bundle.config["diffusion_steps"],
            "text_scale": torch.full((len(lengths),), scale, device=self.device),
            "reconstruction_guidance": False,
        }
        has_control = bool(obs_mask.any().item())
        if has_control and impute:
            options.update(
                {
                    "imputate": True,
                    "stop_imputation_at": 0,
                    "replacement_distribution": "conditional",
                    "inpainted_motion": obs_x0,
                    "inpainting_mask": obs_mask,
                }
            )
        model_kwargs = {"y": options, "obs_x0": obs_x0, "obs_mask": obs_mask}

        model = self.bundle.net
        if scale != 1.0:
            model = ClassifierFreeSampleModel(model)
        fork_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            noise = torch.randn(
                (len(lengths), 263, 1, n_frames),
                device=self.device,
            )
            sample = self.bundle.diffusion.p_sample_loop(
                model,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=progress,
            )
        absolute_output = self.bundle.denormalize_absolute(
            sample[:, :, 0, :].permute(0, 2, 1).contiguous()
        )
        output = absolute_output if return_absolute else absolute_to_relative(absolute_output)
        arrays = output.detach().cpu().numpy().astype(np.float32)
        return [arrays[index, :length] for index, length in enumerate(lengths)]

    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        **kwargs,
    ) -> List[np.ndarray]:
        """Generate standard physical-scale HumanML3D-263 motions from text."""
        return self._sample(captions, lengths, motions=None, control_mode="none", **kwargs)

    def infer_control(
        self,
        captions: Sequence[str],
        motions: Sequence[np.ndarray],
        lengths: Optional[Sequence[int]] = None,
        control_mode: str = "first_last",
        **kwargs,
    ) -> List[np.ndarray]:
        """Generate motion while preserving selected frames or joints.

        ``motions`` use standard physical-scale HML263. ``control_mode`` may be
        ``first_last``, ``start``, ``sparse``, ``prefix``, ``suffix``,
        ``middle``, ``trajectory``, ``lower_body``, ``pelvis_feet``,
        ``pelvis_vr``, or ``joints``.
        """
        if lengths is None:
            lengths = [len(motion) for motion in motions]
        return self._sample(
            captions,
            lengths,
            motions=motions,
            control_mode=control_mode,
            **kwargs,
        )

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)


__all__ = ["CondMDIPipeline"]
