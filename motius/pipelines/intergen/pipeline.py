"""InterGen two-person text-to-motion pipeline."""

from __future__ import annotations
from motius.motion.conditioning.clip_text import encode_clip_text

from typing import Optional, Sequence

import random
import numpy as np
import torch

from motius.diffusion.intergen.factory import create_diffusion
from .guidance import ClassifierFreeSampleModel

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

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: str | Sequence[str],
        *,
        motion_len: int = 210,
        seed: Optional[int] = None,
        return_numpy: bool = True,
    ):
        prompts = [captions] if isinstance(captions, str) else list(captions)
        if not prompts:
            raise ValueError("InterGen needs at least one text prompt")
        if not 15 <= int(motion_len) <= 300:
            raise ValueError("InterGen motion_len must be between 15 and 300 frames")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        cond = encode_clip_text(self.bundle, prompts)
        cfg = self.bundle.network.cfg
        diffusion = create_diffusion(cfg, cfg.STRATEGY)
        model = ClassifierFreeSampleModel(self.bundle.model, cfg.CFG_WEIGHT)
        out = diffusion.ddim_sample_loop(
            model, (len(prompts), int(motion_len), cfg.INPUT_DIM * 2),
            clip_denoised=False, progress=True,
            model_kwargs={"mask": None, "cond": cond}, x_start=None)
        out = out.reshape(len(prompts), int(motion_len), 2, -1)
        out = out * self.bundle.std.to(out.device) + self.bundle.mean.to(out.device)
        return out.detach().cpu().numpy() if return_numpy else out

    def __call__(self, captions, **kwargs):
        return self.infer_t2m(captions, **kwargs)

    infer_text_to_multi_person_motion = infer_t2m


__all__ = ["InterGenPipeline"]
