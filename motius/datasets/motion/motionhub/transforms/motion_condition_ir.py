"""MotionConditionIR transforms for unified motion foundation training.

The IR is metadata only. It lets data transforms describe *why* a tensor is
present (source edit motion, keypose, trajectory, strict preserve region, etc.)
without changing the current VACE tensor path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


@dataclass
class MotionConditionBlock:
    """A single role-tagged condition block."""

    role: str
    type: str = 'motion'
    motion_key: Optional[str] = None
    mask_key: Optional[str] = None
    time_range: Optional[Tuple[int, int]] = None
    fidelity: str = 'strict'
    conflict_policy: str = 'condition_overrides'
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionConditionIR:
    """Unified sample description consumed by future role-aware training."""

    task: str
    instruction: str = ''
    target_motion_key: str = 'tgt_motion'
    condition_blocks: List[MotionConditionBlock] = field(default_factory=list)
    loss_roles: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['condition_blocks'] = [block.to_dict() for block in self.condition_blocks]
        return data


def build_motion_condition_ir(
    task: str,
    instruction: str = '',
    target_motion_key: str = 'tgt_motion',
    condition_blocks: Optional[Sequence[Dict[str, Any]]] = None,
    loss_roles: Optional[Dict[str, str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable MotionConditionIR dictionary."""

    blocks = [
        block if isinstance(block, MotionConditionBlock) else MotionConditionBlock(**block)
        for block in (condition_blocks or [])
    ]
    ir = MotionConditionIR(
        task=task,
        instruction=instruction,
        target_motion_key=target_motion_key,
        condition_blocks=list(blocks),
        loss_roles=loss_roles or {},
        meta=meta or {},
    )
    return ir.to_dict()


@TRANSFORMS.register_module()
class BuildMotionConditionIR(BaseTransform):
    """Attach a MotionConditionIR dict to ``results``.

    This generic transform is useful for existing M2M data. More specialized
    transforms, such as MotionFix triplets, can build richer IR directly.
    """

    def __init__(
        self,
        task: str,
        instruction_key: Optional[str] = 'caption',
        instruction: str = '',
        target_motion_key: str = 'tgt_motion',
        condition_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        loss_roles: Optional[Dict[str, str]] = None,
        output_key: str = 'motion_condition_ir',
        meta_keys: Optional[Sequence[str]] = None,
    ):
        self.task = task
        self.instruction_key = instruction_key
        self.instruction = instruction
        self.target_motion_key = target_motion_key
        self.condition_blocks = list(condition_blocks or [])
        self.loss_roles = loss_roles or {}
        self.output_key = output_key
        self.meta_keys = list(meta_keys or [])

    def transform(self, results: Dict) -> Dict:
        instruction = self.instruction
        if self.instruction_key:
            instruction = results.get(self.instruction_key, instruction)
            if instruction is None:
                instruction = ''
        meta = {key: results.get(key) for key in self.meta_keys if key in results}
        results[self.output_key] = build_motion_condition_ir(
            task=self.task,
            instruction=str(instruction),
            target_motion_key=self.target_motion_key,
            condition_blocks=self.condition_blocks,
            loss_roles=self.loss_roles,
            meta=meta,
        )
        return results


@TRANSFORMS.register_module()
class ApplyRoleConditionDropout(BaseTransform):
    """Drop or corrupt condition roles described by ``motion_condition_ir``.

    This is the training-side counterpart of role-aware guidance. By randomly
    removing a role-tagged condition, the model learns which output behavior
    depends on text, source motion, strict preserve regions, or soft references.
    """

    def __init__(
        self,
        role_drop_probs: Optional[Dict[str, float]] = None,
        text_drop_prob: float = 0.0,
        ir_key: str = 'motion_condition_ir',
        dropped_roles_key: str = 'dropped_condition_roles',
    ):
        self.role_drop_probs = role_drop_probs or {}
        self.text_drop_prob = text_drop_prob
        self.ir_key = ir_key
        self.dropped_roles_key = dropped_roles_key

    def transform(self, results: Dict) -> Dict:
        ir = results.get(self.ir_key) or {}
        dropped = []

        if self.text_drop_prob > 0 and torch.rand(()) < self.text_drop_prob:
            for key in ('caption', 'edit_text', 'instruction'):
                if key in results:
                    results[key] = ''
            if isinstance(ir, dict):
                ir['instruction'] = ''
            dropped.append('instruction')

        blocks = ir.get('condition_blocks', []) if isinstance(ir, dict) else []
        for block in blocks:
            role = block.get('role')
            prob = float(self.role_drop_probs.get(role, 0.0))
            if prob <= 0 or torch.rand(()) >= prob:
                continue

            motion_key = block.get('motion_key')
            mask_key = block.get('mask_key')
            if motion_key in results and isinstance(results[motion_key], torch.Tensor):
                motion = results[motion_key]
                if mask_key in results and isinstance(results[mask_key], torch.Tensor):
                    mask = results[mask_key].to(device=motion.device, dtype=motion.dtype)
                    while mask.ndim < motion.ndim:
                        mask = mask.unsqueeze(-1)
                    results[motion_key] = motion * (1.0 - mask)
                else:
                    results[motion_key] = torch.zeros_like(motion)
            dropped.append(str(role))

        results[self.dropped_roles_key] = dropped
        return results
