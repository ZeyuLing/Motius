"""MLD text-to-motion pipeline.

Runs the Motius-native MLD latent diffusion path:
SentenceT5 text features -> DDIM latent diffusion with classifier-free guidance
-> MLD VAE decode -> HumanML3D-263 de-normalisation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

MLD_UNIT_LENGTH = 4
MLD_MIN_FRAMES = 40
MLD_MAX_FRAMES = 196


@PIPELINES.register_module()
class MLDPipeline(BasePipeline):
    """Inference pipeline for the MLD bundle."""

    BUNDLE_CLS = "motius.models.mld.MLDBundle"

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
        ml = (int(n_frames) // MLD_UNIT_LENGTH) * MLD_UNIT_LENGTH
        return max(MLD_MIN_FRAMES, min(MLD_MAX_FRAMES, ml))

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        clamp: bool = True,
    ) -> List[np.ndarray]:
        """Generate HumanML3D-263 motions (physical scale) from text."""
        from motius.models.motionlcm.network import generate_motion

        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        bundle = self.bundle
        captions = list(captions)
        frame_lens = [self.clamp_length(x) if clamp else int(x) for x in lengths]

        gs = bundle.guidance_scale if guidance_scale is None else float(guidance_scale)
        steps = (
            bundle.num_inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        motions = generate_motion(
            bundle.text_encoder,
            bundle.vae,
            bundle.denoiser,
            bundle.scheduler,
            captions,
            frame_lens,
            guidance_scale=gs,
            num_inference_steps=steps,
        )

        out = []
        for i, m in enumerate(motions):
            denorm = bundle.denormalize(m.float())
            out.append(denorm[: frame_lens[i]].cpu().numpy().astype(np.float32))
        return out

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
