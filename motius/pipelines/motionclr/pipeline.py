"""Diffusers sampling pipeline for MotionCLR text-to-motion inference.

Sampling follows ``models/gaussian_diffusion.py`` from
IDEA-Research/MotionCLR at revision
``a6f44a791940682fe335c82f1b436bae05a1cebb``. MotionCLR is licensed under
the IDEA License 1.0, Copyright (c) IDEA. All Rights Reserved.
"""

from __future__ import annotations

import inspect
from typing import Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


_SCHEDULER_CONFIGS = {
    "dpmsolver": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "ddpm": (
        "DDPMScheduler",
        {"variance_type": "fixed_small", "clip_sample": False},
    ),
    "ddim": ("DDIMScheduler", {"clip_sample": False}),
    "deis": ("DEISMultistepScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
}


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cpu" else device)
    generator.manual_seed(int(seed))
    return generator


@PIPELINES.register_module()
class MotionCLRPipeline(BasePipeline):
    """Official MotionCLR HumanML3D-263 text-to-motion inference."""

    BUNDLE_CLS = "motius.models.motionclr.MotionCLRBundle"

    def __init__(
        self,
        bundle,
        diffuser_name: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        scheduler=None,
        device: Optional[str | torch.device] = None,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.bundle.to_device(device)
        self.diffuser_name = diffuser_name or bundle.diffuser_name
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else bundle.num_inference_steps
        )
        self.guidance_scale = float(
            guidance_scale if guidance_scale is not None else bundle.guidance_scale
        )
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        self.scheduler = scheduler or self._build_scheduler(self.diffuser_name)

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @staticmethod
    def _build_scheduler(name: str):
        try:
            class_name, extra = _SCHEDULER_CONFIGS[name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported MotionCLR diffuser_name {name!r}; "
                f"choose one of {sorted(_SCHEDULER_CONFIGS)}"
            ) from exc
        import diffusers

        scheduler_class = getattr(diffusers, class_name)
        return scheduler_class(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="sample",
            **extra,
        )

    @staticmethod
    def _validate_inputs(
        captions: Sequence[str],
        lengths: Sequence[int],
    ) -> tuple[list[str], list[int]]:
        captions = [str(caption) for caption in captions]
        lengths = [int(length) for length in lengths]
        if not captions:
            raise ValueError("captions must contain at least one text prompt")
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        if any(length < 1 or length > 196 for length in lengths):
            raise ValueError("MotionCLR lengths must be in [1, 196] frames at 20 fps")
        return captions, lengths

    @torch.inference_mode()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        *,
        seed: int = 0,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        enc_text: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
        return_tensor: bool = False,
    ):
        """Generate one HumanML3D-263 motion per caption.

        Each output is cropped to its requested frame length. By default the
        result is de-normalized to the physical HumanML3D feature scale.
        """
        captions, lengths = self._validate_inputs(captions, lengths)
        network = self.bundle.network
        if network is None:
            raise RuntimeError("MotionCLR network was loaded with load_model=False")
        steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.num_inference_steps
        )
        if steps < 1:
            raise ValueError("num_inference_steps must be positive")
        cfg_scale = float(
            guidance_scale if guidance_scale is not None else self.guidance_scale
        )
        try:
            dtype = next(network.parameters()).dtype
        except StopIteration:
            dtype = self.bundle.inference_dtype or torch.float32

        batch_size = len(captions)
        max_frames = max(lengths)
        generator = _device_generator(self.device, seed)
        fork_devices = (
            [self.device.index or 0] if self.device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            sample = torch.randn(
                (batch_size, max_frames, 263),
                generator=generator,
                device=self.device,
                dtype=dtype,
            )
            self.scheduler.set_timesteps(steps, device=self.device)
            if enc_text is None:
                enc_text = network.encode_text(captions, self.device)
            else:
                enc_text = torch.as_tensor(enc_text, device=self.device, dtype=dtype)
                if enc_text.shape[0] != batch_size:
                    raise ValueError("enc_text batch dimension must match captions")

            accepts_generator = "generator" in inspect.signature(
                self.scheduler.step
            ).parameters
            for timestep in self.scheduler.timesteps:
                timestep_value = int(timestep.item())
                timestep_batch = torch.full(
                    (batch_size,),
                    timestep_value,
                    dtype=torch.long,
                    device=self.device,
                )
                if getattr(network, "cond_mask_prob", 0.0) > 0:
                    prediction = network.forward_with_cfg(
                        sample,
                        timestep_batch,
                        enc_text=enc_text,
                        cfg_scale=cfg_scale,
                    )
                else:
                    prediction = network(
                        sample,
                        timestep_batch,
                        enc_text=enc_text,
                    )
                step_kwargs = {"generator": generator} if accepts_generator else {}
                sample = self.scheduler.step(
                    prediction,
                    timestep,
                    sample,
                    **step_kwargs,
                ).prev_sample

        outputs = []
        for index, length in enumerate(lengths):
            motion = sample[index, :length].float()
            if not return_normalized:
                motion = self.bundle.denormalize(motion)
            outputs.append(motion if return_tensor else motion.cpu().numpy().astype(np.float32))
        return outputs

    def text_to_motion(self, caption: str, num_frames: int, **kwargs):
        return self.infer_t2m([caption], [num_frames], **kwargs)[0]

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)


__all__ = ["MotionCLRPipeline"]
