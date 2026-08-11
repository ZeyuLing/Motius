"""Prepare MotionFix-style semantic editing triplets for M2M training."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from mmcv import BaseTransform

from motius.registry import TRANSFORMS

from .motion_condition_ir import build_motion_condition_ir


def _as_motion_tensor(value, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'Expected torch.Tensor for {key!r}, got {type(value)}')
    if value.ndim != 2:
        raise ValueError(f'Expected {key!r} with shape (T, D), got {tuple(value.shape)}')
    return value.float()


def _dilate_temporal(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    x = mask.transpose(0, 1).unsqueeze(0)
    x = F.max_pool1d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return x.squeeze(0).transpose(0, 1)


def infer_editable_mask_from_triplet(
    source_motion: torch.Tensor,
    target_motion: torch.Tensor,
    threshold: float = 0.08,
    temporal_dilate: int = 4,
    min_changed_frames: int = 1,
) -> torch.Tensor:
    """Infer a 0/1 editable mask from a source-target motion pair.

    The output follows M2M convention: ``1`` means generate/edit and ``0`` means
    preserve. The heuristic intentionally over-masks around changed frames so
    semantic edits have enough context to become physically plausible.
    """

    T = min(source_motion.shape[0], target_motion.shape[0])
    D = min(source_motion.shape[1], target_motion.shape[1])
    src = source_motion[:T, :D]
    tgt = target_motion[:T, :D]

    diff = (tgt - src).abs()
    dim_scale = diff.detach().median(dim=0).values.clamp_min(1e-3)
    score = diff / dim_scale
    changed_dim = (score > threshold).float()

    frame_score = changed_dim.mean(dim=-1)
    changed_frame = frame_score > 0.02
    if int(changed_frame.sum().item()) < min_changed_frames:
        topk = min(max(min_changed_frames, 1), T)
        _, indices = frame_score.topk(topk)
        changed_frame = torch.zeros(T, dtype=torch.bool, device=source_motion.device)
        changed_frame[indices] = True

    mask = torch.zeros(T, D, dtype=torch.float32, device=source_motion.device)
    mask[changed_frame] = 1.0
    mask = torch.maximum(mask, changed_dim)
    mask = _dilate_temporal(mask, temporal_dilate)
    return mask


@TRANSFORMS.register_module()
class PrepareMotionFixTriplet(BaseTransform):
    """Convert semantic edit triplets into current M2M training keys.

    Expected inputs after loading/cropping:

    - ``source_key``: source motion tensor, shape ``(T, D)``
    - ``target_key``: target/edited motion tensor, shape ``(T, D)``
    - ``instruction_key``: edit instruction text

    Produced keys follow the existing M2M trainer contract:

    - ``src_motion``: source motion
    - ``tgt_motion``: target motion
    - ``src_mask``: editable mask, ``1=edit/generate``
    - ``edit_mode``: ``True`` so VACE reactive carries source motion
    - ``preserve_mask``: complement of editable mask
    - ``motion_condition_ir``: role-tagged sample metadata
    """

    def __init__(
        self,
        source_key: str = 'source_motion',
        target_key: str = 'target_motion',
        instruction_key: str = 'edit_text',
        explicit_mask_key: Optional[str] = None,
        threshold: float = 0.08,
        temporal_dilate: int = 4,
        meta_keys: Optional[Sequence[str]] = None,
    ):
        self.source_key = source_key
        self.target_key = target_key
        self.instruction_key = instruction_key
        self.explicit_mask_key = explicit_mask_key
        self.threshold = threshold
        self.temporal_dilate = temporal_dilate
        self.meta_keys = list(meta_keys or ['motion_path', 'fps'])

    def transform(self, results: Dict) -> Dict:
        source_motion = _as_motion_tensor(results[self.source_key], self.source_key)
        target_motion = _as_motion_tensor(results[self.target_key], self.target_key)

        T = min(source_motion.shape[0], target_motion.shape[0])
        D = min(source_motion.shape[1], target_motion.shape[1])
        source_motion = source_motion[:T, :D].clone()
        target_motion = target_motion[:T, :D].clone()

        if self.explicit_mask_key and self.explicit_mask_key in results:
            src_mask = _as_motion_tensor(results[self.explicit_mask_key], self.explicit_mask_key)
            src_mask = src_mask[:T, :D].clone().float()
        else:
            src_mask = infer_editable_mask_from_triplet(
                source_motion,
                target_motion,
                threshold=self.threshold,
                temporal_dilate=self.temporal_dilate,
            )

        valid_length = int(results.get('num_frames', T))
        valid_length = min(valid_length, T)
        preserve_mask = 1.0 - src_mask

        instruction = results.get(self.instruction_key, results.get('caption', ''))
        if instruction is None:
            instruction = ''

        results['src_motion'] = source_motion
        results['tgt_motion'] = target_motion
        results['src_mask'] = src_mask
        results['editable_mask'] = src_mask
        results['preserve_mask'] = preserve_mask
        results['condition_adherence_mask'] = preserve_mask
        results['tgt_length'] = valid_length
        results['src_length'] = valid_length
        results['edit_mode'] = True
        results['caption'] = str(instruction)

        meta = {key: results.get(key) for key in self.meta_keys if key in results}
        results['motion_condition_ir'] = build_motion_condition_ir(
            task='semantic_edit',
            instruction=str(instruction),
            target_motion_key='tgt_motion',
            condition_blocks=[
                dict(
                    role='source_motion_for_edit',
                    type='motion',
                    motion_key='src_motion',
                    mask_key='editable_mask',
                    time_range=(0, valid_length),
                    fidelity='reference',
                    conflict_policy='text_overrides_editable',
                ),
                dict(
                    role='must_preserve_region',
                    type='mask',
                    mask_key='preserve_mask',
                    time_range=(0, valid_length),
                    fidelity='strict',
                    conflict_policy='condition_overrides',
                ),
            ],
            loss_roles={
                'editable': 'editable_mask',
                'preserve': 'preserve_mask',
                'condition_adherence': 'condition_adherence_mask',
            },
            meta=meta,
        )
        return results
