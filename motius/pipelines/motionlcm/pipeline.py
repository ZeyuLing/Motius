"""MotionLCM text-to-motion pipeline.

Drives the Motius-native MotionLCM implementation
(``motius.models.motionlcm.network``): sentence-t5-large text
features -> latent **consistency** sampling with the diffusers ``LCMScheduler``
(distilled CFG via timestep conditioning, default **1** step) -> ``MldVae.decode``
-> 263-dim motion -> de-normalise to physical scale.

Logic matches the official MotionLCM test path (``MLD.t2m_eval`` /
``MLD._diffusion_reverse`` for the LCM denoiser): same ``guidance_scale`` folded
into ``timestep_cond``, same ``num_inference_steps``, 20 fps, ``unit_length=4``.
Fully independent of the original repo.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

# MotionLCM runs at 20 fps; HumanML3D was trained on 40..196 frames.
MOTIONLCM_UNIT_LENGTH = 4
MOTIONLCM_MIN_FRAMES = 40
MOTIONLCM_MAX_FRAMES = 196


@PIPELINES.register_module()
class MotionLCMPipeline(BasePipeline):
    """Inference pipeline for the MotionLCM bundle."""

    BUNDLE_CLS = "motius.models.motionlcm.MotionLCMBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
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
    def clamp_length(n_frames: int) -> int:
        ml = (int(n_frames) // MOTIONLCM_UNIT_LENGTH) * MOTIONLCM_UNIT_LENGTH
        return max(MOTIONLCM_MIN_FRAMES, min(MOTIONLCM_MAX_FRAMES, ml))

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        clamp: bool = True,
    ) -> List[np.ndarray]:
        """Generate HumanML3D-263 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: per-sample target length in **frames** (20 fps).
            guidance_scale: distilled-CFG guidance scale; defaults to the
                bundle's configured value (paper default 7.5).
            num_inference_steps: LCM steps (NFE); defaults to the bundle value
                (1). Use 2/4 for higher quality.
            clamp: clamp/round given ``lengths`` to ``[40, 196]`` multiples of 4.

        Returns:
            List of B arrays, each ``(T_i, 263)`` un-standardized.
        """
        from motius.models.motionlcm.network import generate_motion

        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        bundle = self.bundle
        captions = list(captions)
        bs = len(captions)

        frame_lens = [self.clamp_length(x) if clamp else int(x) for x in lengths]

        gs = bundle.guidance_scale if guidance_scale is None else float(guidance_scale)
        steps = (bundle.num_inference_steps if num_inference_steps is None
                 else int(num_inference_steps))

        motions = generate_motion(
            bundle.text_encoder,
            bundle.vae,
            bundle.denoiser,
            bundle.scheduler,
            captions,
            frame_lens,
            guidance_scale=gs,
            num_inference_steps=steps,
        )  # list of (T_i, 263) normalized

        out = []
        for i in range(bs):
            m = bundle.denormalize(motions[i].float())
            out.append(m.cpu().numpy().astype(np.float32))
        return out

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
