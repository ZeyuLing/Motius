"""Compute 147-dim motion representation from 135-dim motion.

Adds 12-dim end-effector position channels (L_Hand, R_Hand, L_Foot, R_Foot, each 3D)
to the base 135-dim (3 trans + 132 rot6d).

End-effector joint indices in SMPL-22:
- L_Wrist (20): parent of L_Hand (computed via FK)
- R_Wrist (21): parent of R_Hand (computed via FK)
- L_Foot (10): end-effector for left leg
- R_Foot (11): end-effector for right leg

The transform:
1. Runs differentiable FK on the 135-dim motion (always using LOCAL rotation)
2. Extracts world-space xyz positions for the 4 end-effectors
3. Concatenates positions to produce 147-dim output: [trans(3), rot6d(132), ee_pos(12)]

Layout:
    dims [0:3]      — translation (3)
    dims [3:135]    — rot6d (22*6 = 132)
    dims [135:138]  — L_Wrist position (xyz)
    dims [138:141]  — R_Wrist position (xyz)
    dims [141:144]  — L_Foot position (xyz)
    dims [144:147]  — R_Foot position (xyz)

Usage in pipeline:
    # For local rotation models:
    dict(type='LoadSmplx55', ...),
    dict(type='Compute147DimEndEffector', key='motion'),  # 135 -> 147
    dict(type='RandomCropPadding', ...),

    # For global rotation models:
    dict(type='LoadSmplx55', ...),
    dict(type='Compute147DimEndEffector', key='motion'),  # 135 -> 147 (FK uses local rot)
    dict(type='LocalToGlobalRotation', key='motion'),  # rotation channels -> global
    dict(type='RandomCropPadding', ...),

    Note: Compute147DimEndEffector MUST come BEFORE LocalToGlobalRotation because
    FK requires local rotation. LocalToGlobalRotation only changes the rotation
    channels (dims 3:135), end-effector position channels (dims 135:147) are unaffected.
"""

from __future__ import annotations

import os.path as osp
from typing import Dict, Optional, Tuple

import torch
from mmcv import BaseTransform
from torch import Tensor

from motius.registry import TRANSFORMS

# SMPL-22 end-effector joint indices
EE_JOINT_INDICES = {
    'L_Wrist': 20,
    'R_Wrist': 21,
    'L_Foot': 10,
    'R_Foot': 11,
}

EE_JOINTS_LIST = ['L_Wrist', 'R_Wrist', 'L_Foot', 'R_Foot']  # Order for output


def compute_end_effector_positions(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Compute 12-dim end-effector position channels from 135-dim local-rotation motion.

    Args:
        motion_135: (*, 135) motion tensor in local rotation space.
        bone_offsets: (22, 3) bone offsets tensor.

    Returns:
        (*, 12) end-effector position channels (4 joints × 3D).
    """
    from motius.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    leading = motion_135.shape[:-1]

    # FK always uses local rotation
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_135, bone_offsets, rotation_space='local')

    # world_pos: (*, 22, 3)
    # Extract end-effector positions in order: L_Wrist, R_Wrist, L_Foot, R_Foot
    ee_indices = [EE_JOINT_INDICES[name] for name in EE_JOINTS_LIST]
    ee_positions = world_pos[..., ee_indices, :]  # (*, 4, 3)

    return ee_positions.reshape(*leading, 12)


def motion135_to_147(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Convert 135-dim motion to 147-dim by appending end-effector position channels.

    Args:
        motion_135: (*, 135) motion tensor.
        bone_offsets: (22, 3) bone offsets.

    Returns:
        (*, 147) motion tensor.
    """
    ee_12 = compute_end_effector_positions(motion_135, bone_offsets)
    return torch.cat([motion_135, ee_12], dim=-1)


def motion147_to_135(motion_147: Tensor) -> Tensor:
    """Extract 135-dim (trans + rot6d) from 147-dim motion.

    Simply takes the first 135 dimensions, discarding end-effector position channels.

    Args:
        motion_147: (*, 147) motion tensor.

    Returns:
        (*, 135) motion tensor.
    """
    return motion_147[..., :135]


@TRANSFORMS.register_module(force=True)
class Compute147DimEndEffector(BaseTransform):
    """Append 12-dim end-effector position channels to 135-dim motion.

    This transform runs differentiable FK to compute world-space positions of
    4 end-effectors (L_Wrist, R_Wrist, L_Foot, R_Foot) and appends them to
    the 135-dim base motion.

    Args:
        key: Key to motion data (default 'motion').
        bone_offsets_dir: Directory containing bone_offsets tensor. If None,
            loads from standard location.
    """

    def __init__(
        self,
        key: str = 'motion',
        bone_offsets_dir: Optional[str] = None,
    ):
        super().__init__()
        self.key = key
        self.bone_offsets_dir = bone_offsets_dir or 'data/bone_offsets'

        # Load bone_offsets (cached)
        self._bone_offsets_cache = None

    def _load_bone_offsets(self) -> Tensor:
        """Load bone offsets from disk or cache."""
        if self._bone_offsets_cache is not None:
            return self._bone_offsets_cache

        # Try loading from torch file
        offsets_path = osp.join(self.bone_offsets_dir, 'smpl22_bone_offsets.pt')
        if osp.exists(offsets_path):
            self._bone_offsets_cache = torch.load(offsets_path, map_location='cpu')
            return self._bone_offsets_cache

        # Fallback: use standard SMPL-22 offsets (T-pose)
        # These are the standard SMPL-22 bone offsets
        from motius.datasets.motion.motionhub.smpl_data import SMPL22_BONE_OFFSETS
        self._bone_offsets_cache = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
        return self._bone_offsets_cache

    def transform(self, results: Dict) -> Dict:
        """Apply the transform.

        Args:
            results: Dictionary with results[f'{key}_path'] or results[key].

        Returns:
            Updated results with motion expanded to 147-dim.
        """
        key = self.key
        motion_key = key

        # Get motion tensor
        if motion_key not in results:
            raise KeyError(f"Motion key {motion_key} not found in results")

        motion = results[motion_key]
        if isinstance(motion, dict):
            # Multi-person case
            for person_key in motion:
                m = motion[person_key]
                if m.shape[-1] == 135:
                    bone_offsets = self._load_bone_offsets()
                    motion[person_key] = motion135_to_147(m, bone_offsets)
        else:
            # Single-person case
            if motion.shape[-1] == 135:
                bone_offsets = self._load_bone_offsets()
                results[motion_key] = motion135_to_147(motion, bone_offsets)
            elif motion.shape[-1] != 147:
                raise ValueError(
                    f"Motion dimension must be 135 or 147, got {motion.shape[-1]}"
                )

        return results


__all__ = [
    'Compute147DimEndEffector',
    'compute_end_effector_positions',
    'motion135_to_147',
    'motion147_to_135',
    'EE_JOINT_INDICES',
    'EE_JOINTS_LIST',
]
