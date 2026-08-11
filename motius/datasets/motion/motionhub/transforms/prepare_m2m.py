"""Prepare arbitrary motion evidence for MotionCanvas training."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS

from .condition_sampler import MOTION_DIM, sample_condition


@TRANSFORMS.register_module()
class PrepareM2MCondition(BaseTransform):
    """Sample a condition-target mask on valid, unpadded motion frames.

    Paired edit loaders may replace ``src_motion`` after this transform. They
    preserve ``src_mask`` so pure edit and edit-plus-motion-condition use the
    same model interface.
    """

    def __init__(
        self,
        key: str = "motion",
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.key = key
        self.sampler_config = dict(sampler_config or {})

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        if not isinstance(motion, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor for key {self.key!r}, got {type(motion)}"
            )
        length, dim = motion.shape[-2:]
        if dim != MOTION_DIM:
            raise ValueError(f"Expected motion_dim={MOTION_DIM}, got {dim}")

        valid_length = int(results.get("num_frames", length))
        valid_length = max(1, min(valid_length, length))
        mask_valid, _ = sample_condition(
            valid_length,
            np.random.RandomState(),
            **self.sampler_config,
        )
        src_mask = np.ones((length, MOTION_DIM), dtype=np.float32)
        src_mask[:valid_length] = mask_valid

        results["src_motion"] = motion.clone()
        results["tgt_motion"] = motion.clone()
        results["src_mask"] = torch.from_numpy(src_mask)
        results["tgt_length"] = valid_length
        results["src_length"] = valid_length
        results["edit_mode"] = False
        return results
