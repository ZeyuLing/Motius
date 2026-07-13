"""InterGen two-person text-to-motion pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class InterGenPipeline(BasePipeline):
    """Generate paired InterHuman-262 tracks from interaction captions."""

    BUNDLE_CLS = "motius.models.intergen.InterGenBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None and bundle.device != torch.device(device):
            bundle.device_name = device
            bundle.model.to(device)
            bundle.to(device)

    def infer_t2m(
        self,
        captions: str | Sequence[str],
        *,
        motion_len: int = 210,
        seed: Optional[int] = None,
        return_numpy: bool = True,
    ):
        return self.bundle.generate(
            captions,
            motion_len=motion_len,
            seed=seed,
            return_numpy=return_numpy,
        )

    def __call__(self, captions, **kwargs):
        return self.infer_t2m(captions, **kwargs)


__all__ = ["InterGenPipeline"]
