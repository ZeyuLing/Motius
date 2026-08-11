"""Full-mask transform for Phase 1 T2M-only training.

All frames and dimensions are masked (mask=1), making every sample a
pure text-to-motion generation task. Used for Phase 1 curriculum
training before introducing completion/editing in Phase 2.
"""

from __future__ import annotations

from typing import Dict

import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PrepareM2Mv2FullMask(BaseTransform):
    """Phase 1 condition: full mask (T2M only).

    Every sample has mask=1 everywhere → pure generation.
    No editing, no completion, no constraints.

    Parameters
    ----------
    key : str
        Motion key in results dict.
    """

    def __init__(self, key: str = 'motion'):
        self.key = key

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor)

        T, D = motion.shape[-2], motion.shape[-1]

        # tgt_length / src_length must be the number of VALID frames (pre-pad);
        # RandomCropPadding writes `num_frames` = real content length so that
        # right-padded tail frames are excluded from loss and attention.
        valid_length = int(results.get('num_frames', T))
        results['src_motion'] = torch.zeros_like(motion)  # all zero (masked)
        results['tgt_motion'] = motion.clone()
        results['src_mask'] = torch.ones(T, D, dtype=torch.float32)
        results['tgt_length'] = valid_length
        results['src_length'] = valid_length
        results['edit_mode'] = False

        return results
