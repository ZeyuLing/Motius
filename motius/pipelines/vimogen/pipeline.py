"""ViMoGen text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class ViMoGenPipeline(BasePipeline):
    """Inference pipeline for the ViMoGen 276D HumanML3D checkpoint."""

    BUNDLE_CLS = "motius.models.vimogen.ViMoGenBundle"

    def __init__(self, bundle, device=None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        denoising_strength: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        motions = self.bundle.generate_motion276(
            captions=captions,
            lengths=lengths,
            seed=seed,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            show_progress=show_progress,
        )
        return [motion.numpy().astype(np.float32) for motion in motions]

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
