"""Deterministic overfit-case condition masks for M2M v2.

This transform is intentionally small and explicit: each sample carries an
``overfit_task`` tag in the annotation, and the tag maps to one fixed mask
pattern.  It is used for 100-case overfitting tests where we want stable
coverage instead of relying on the random training sampler to eventually hit
every task family.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


UPPER_BODY_JOINTS = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
LOWER_BODY_JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11)
END_EFFECTOR_JOINTS = (7, 8, 10, 11, 20, 21)
ANKLE_JOINTS = (7, 8)


def _mark_joints(mask: torch.Tensor, joints: Iterable[int], value: float) -> None:
    for j in joints:
        mask[:, 3 + j * 6 : 3 + (j + 1) * 6] = value
        if j > 0:
            pos = 135 + (j - 1) * 3
            mask[:, pos : pos + 3] = value


def _full_mask(T: int, D: int, value: float) -> torch.Tensor:
    return torch.full((T, D), float(value), dtype=torch.float32)


@TRANSFORMS.register_module(force=True)
class PrepareM2Mv2OverfitCase(BaseTransform):
    """Prepare deterministic M2M v2 conditions from ``overfit_task``.

    Mask convention follows the rest of M2M v2: ``1 = generate`` and
    ``0 = known``.
    """

    SUPPORTED_TASKS = {
        't2m_full',
        'prefix',
        'suffix',
        'inbetween',
        'sparse_keyframes',
        'rot_keyframes',
        'pos_keyframes',
        'trajectory',
        'end_effector',
        'foot_ground',
        'upper_body',
        'lower_body',
        'style_edit',
        'synthetic_edit',
        'real_edit_permo',
        'real_edit_motionfix',
    }

    def __init__(
        self,
        key: str = 'motion',
        task_key: str = 'overfit_task',
        period: int = 10,
    ) -> None:
        self.key = key
        self.task_key = task_key
        self.period = int(period)

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor)
        T, D = motion.shape[-2], motion.shape[-1]
        task = str(results.get(self.task_key, 't2m_full'))
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported overfit_task={task!r}")

        mask = self._build_mask(task, T, D)
        valid_length = int(results.get('num_frames', T))

        results['src_motion'] = motion.clone()
        results['tgt_motion'] = motion.clone()
        results['src_mask'] = mask
        results['tgt_length'] = valid_length
        results['src_length'] = valid_length
        results['edit_mode'] = task in {
            'synthetic_edit', 'real_edit_permo', 'real_edit_motionfix'
        }
        results['mask_strategy'] = task

        if task == 'synthetic_edit':
            src_motion = motion.clone()
            corrupted = mask > 0.5
            src_motion[corrupted] = src_motion[corrupted] + 0.05
            results['src_motion'] = src_motion

        return results

    def _build_mask(self, task: str, T: int, D: int) -> torch.Tensor:
        if task in {'t2m_full', 'real_edit_permo', 'real_edit_motionfix'}:
            return _full_mask(T, D, 1.0)

        if task == 'prefix':
            mask = _full_mask(T, D, 1.0)
            n = max(1, min(T, T // 5))
            mask[:n] = 0.0
            return mask

        if task == 'suffix':
            mask = _full_mask(T, D, 1.0)
            n = max(1, min(T, T // 5))
            mask[-n:] = 0.0
            return mask

        if task == 'inbetween':
            mask = _full_mask(T, D, 1.0)
            n = max(1, min(T, T // 6))
            mask[:n] = 0.0
            mask[-n:] = 0.0
            return mask

        if task == 'sparse_keyframes':
            mask = _full_mask(T, D, 1.0)
            mask[:: self.period] = 0.0
            return mask

        if task == 'rot_keyframes':
            mask = _full_mask(T, D, 1.0)
            mask[:: self.period, 3:135] = 0.0
            return mask

        if task == 'pos_keyframes':
            mask = _full_mask(T, D, 1.0)
            mask[:: self.period, 135:198] = 0.0
            return mask

        if task == 'trajectory':
            mask = _full_mask(T, D, 1.0)
            mask[:, 0] = 0.0
            mask[:, 2] = 0.0
            return mask

        if task == 'end_effector':
            mask = _full_mask(T, D, 1.0)
            frames = torch.arange(0, T, max(1, self.period))
            for j in END_EFFECTOR_JOINTS:
                if j > 0:
                    pos = 135 + (j - 1) * 3
                    mask[frames, pos : pos + 3] = 0.0
            return mask

        if task == 'foot_ground':
            mask = _full_mask(T, D, 1.0)
            for j in ANKLE_JOINTS:
                pos = 135 + (j - 1) * 3
                mask[:, pos + 1] = 0.0
            return mask

        if task == 'upper_body':
            mask = _full_mask(T, D, 0.0)
            _mark_joints(mask, UPPER_BODY_JOINTS, 1.0)
            return mask

        if task == 'lower_body':
            mask = _full_mask(T, D, 0.0)
            mask[:, 0:3] = 1.0
            _mark_joints(mask, LOWER_BODY_JOINTS, 1.0)
            return mask

        if task == 'style_edit':
            mask = _full_mask(T, D, 1.0)
            mask[:, 0:9] = 0.0
            return mask

        if task == 'synthetic_edit':
            mask = _full_mask(T, D, 0.0)
            start = max(0, T // 3)
            end = max(start + 1, min(T, start + max(1, T // 3)))
            mask[start:end] = 1.0
            _mark_joints(mask, UPPER_BODY_JOINTS, 1.0)
            return mask

        raise ValueError(f"Unsupported overfit_task={task!r}")
