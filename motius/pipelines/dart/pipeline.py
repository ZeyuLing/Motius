"""DART text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class DARTPipeline(BasePipeline):
    """Inference pipeline for DART / DartControl HumanML3D rollout."""

    BUNDLE_CLS = "motius.models.dart.DARTBundle"

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

    @torch.no_grad()
    def infer_t2m_smpl(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 0,
        sample_offset: int = 0,
        guidance_param: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[dict]:
        return self.bundle.generate_smpl_sequences(
            captions,
            lengths,
            seed=seed,
            sample_offset=sample_offset,
            guidance_param=guidance_param,
            show_progress=show_progress,
        )

    @torch.no_grad()
    def infer_t2m_motion135(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 0,
        sample_offset: int = 0,
        guidance_param: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        return self.bundle.generate_motion135(
            captions,
            lengths,
            seed=seed,
            sample_offset=sample_offset,
            guidance_param=guidance_param,
            show_progress=show_progress,
        )

    def infer_t2m(self, captions, lengths, **kwargs):
        return self.infer_t2m_motion135(captions, lengths, **kwargs)

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
