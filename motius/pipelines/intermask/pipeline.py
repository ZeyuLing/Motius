"""InterMask two-person text-to-motion pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class InterMaskPipeline(BasePipeline):
    """Generate InterHuman-262 or Inter-X interaction motion tokens."""

    BUNDLE_CLS = "motius.models.intermask.InterMaskBundle"

    def infer_t2m(
        self,
        captions: str | Sequence[str],
        *,
        motion_len: int = 120,
        seed: Optional[int] = None,
        return_numpy: bool = True,
        cond_scale: Optional[float] = None,
        time_steps: Optional[int] = None,
        topk_filter_thres: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        return self.bundle.generate(
            captions,
            motion_len=motion_len,
            seed=seed,
            return_numpy=return_numpy,
            cond_scale=cond_scale,
            time_steps=time_steps,
            topk_filter_thres=topk_filter_thres,
            temperature=temperature,
        )

    def __call__(self, captions, **kwargs):
        return self.infer_t2m(captions, **kwargs)


__all__ = ["InterMaskPipeline"]
