"""Dataset transform: convert local rotation 6D to global rotation 6D.

Place this transform in the pipeline AFTER ``LoadSmplx55`` and BEFORE
``RandomCropPadding``.  ``LoadSmplx55`` outputs (T, 135) as a
``torch.FloatTensor`` with row-major local rotation 6D.  This transform
replaces the rotation part (dims 3:135) with global (world-frame) rotation
6D while keeping translation (dims 0:3) unchanged.

Why this ordering is correct:
    ``LoadSmplx55`` internally applies Y-axis yaw augmentation on the root's
    local rotation.  After FK this is equivalent to rotating the entire
    skeleton in world space, which is semantically correct for global
    rotation training.
"""

from __future__ import annotations

import torch
from torch import Tensor
from mmcv import BaseTransform

from motius.datasets.motion.motionhub.transforms.fk_utils import (
    local_to_global_rot6d_torch,
)
from motius.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class LocalToGlobalRotation(BaseTransform):
    """Convert local rot6d to global rot6d in the 135-dim motion representation.

    Handles both torch.Tensor (from LoadSmplx55) and numpy.ndarray inputs.

    Args:
        key: Key in ``results`` dict holding the motion array.
            Shape: ``(T, 135)`` for single-person or ``(P, T, 135)`` for multi-person.
    """

    def __init__(self, key: str = 'motion'):
        self.key = key

    def transform(self, results: dict) -> dict:
        motion = results[self.key]  # (T, D) or (P, T, D), Tensor or ndarray
        D = motion.shape[-1]
        assert D in (135, 198), (
            f"LocalToGlobalRotation expects 135 or 198-dim motion, got {D}"
        )

        is_tensor = isinstance(motion, Tensor)

        if not is_tensor:
            # Convert numpy to torch for unified processing
            import numpy as np
            motion = torch.from_numpy(motion)

        results[self.key] = self._convert(motion)

        if not is_tensor:
            # Convert back to numpy if input was numpy
            results[self.key] = results[self.key].numpy()

        return results

    @staticmethod
    def _convert(motion: Tensor) -> Tensor:
        """Convert (*, T, D) motion from local to global rotation.

        Only affects rotation dims [3:135]. Translation [0:3] and optional
        position channels [135:198] are preserved unchanged.
        """
        D = motion.shape[-1]
        transl = motion[..., 0:3]  # (*, T, 3)
        rot6d_local = motion[..., 3:135].reshape(*motion.shape[:-1], 22, 6)

        rot6d_global = local_to_global_rot6d_torch(rot6d_local)

        parts = [transl, rot6d_global.reshape(*motion.shape[:-1], 132)]
        if D > 135:
            parts.append(motion[..., 135:])  # position channels unchanged
        return torch.cat(parts, dim=-1)
