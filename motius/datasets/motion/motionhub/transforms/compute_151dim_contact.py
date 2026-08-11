"""Compute 151-dim motion representation from 147-dim motion.

Adds 4-dim foot contact channel to the 147-dim representation:
    - Binary indicators for L_Foot, R_Foot, L_Wrist, R_Wrist contact
    - Computed via velocity-based contact detection using FK
    - Layout: [L_Foot_contact, R_Foot_contact, L_Wrist_contact, R_Wrist_contact]

151-dim layout:
    dims [0:147]    — original 147-dim motion (trans + rot6d + end-effector positions)
    dims [147:151]  — foot contact binary indicators (4)

Contact detection algorithm (from Momask):
    A joint is "in contact" if its velocity < threshold (default 0.002)
    Velocity is computed between consecutive frames: v[t] = pos[t+1] - pos[t]
    Velocity magnitude: sqrt(v_x^2 + v_y^2 + v_z^2) < threshold

Usage in pipeline:
    dict(type='LoadSmplx55', ...),
    dict(type='Compute147DimEndEffector', key='motion'),  # 135 -> 147
    dict(type='Compute151DimFootContact', key='motion'),  # 147 -> 151
    dict(type='RandomCropPadding', ...),
"""

from __future__ import annotations

import os.path as osp
from typing import Dict, Optional

import torch
from mmcv import BaseTransform
from torch import Tensor

from motius.registry import TRANSFORMS


def detect_foot_contact(
    positions: Tensor,
    velocity_threshold: float = 0.002,
) -> Tensor:
    """Detect foot contact based on joint velocity.

    Args:
        positions: Joint positions, shape (L, 4, 3) or (T, 4, 3).
                   Should be world-space positions of [L_Foot, R_Foot, L_Wrist, R_Wrist].
        velocity_threshold: Velocity threshold for contact detection (default 0.002).

    Returns:
        Contact indicators, shape (L-1, 4) or (T-1, 4).
        Binary tensor (0 or 1) where 1 = in contact, 0 = not in contact.
    """
    device = positions.device
    dtype = positions.dtype

    # Handle batched input: (B, T, J, 3) -> loop over batch
    if positions.dim() == 4:
        B, T, J, _ = positions.shape
        contact_list = []
        for i in range(B):
            contact_i = detect_foot_contact(positions[i], velocity_threshold)
            contact_list.append(contact_i)
        # Pad to same length and stack
        max_len = max(c.shape[0] for c in contact_list)
        contact_padded = []
        for c in contact_list:
            if c.shape[0] < max_len:
                pad_size = max_len - c.shape[0]
                c = torch.cat([c, c[-1:].expand(pad_size, -1)], dim=0)
            contact_padded.append(c)
        return torch.stack(contact_padded, dim=0)

    # Single sequence: (T, J, 3)
    assert positions.dim() == 3, f"Expected 3D or 4D positions, got {positions.dim()}D"
    T, J, _ = positions.shape
    assert J == 4, f"Expected 4 joints, got {J}"

    # Compute velocity between consecutive frames
    # velocity[t] = positions[t+1] - positions[t]  -> shape (T-1, 4, 3)
    velocity = positions[1:, :, :] - positions[:-1, :, :]

    # Compute velocity magnitude for each joint
    velocity_mag = torch.norm(velocity, dim=-1)  # (T-1, 4)

    # Detect contact: velocity < threshold
    contact = (velocity_mag < velocity_threshold).float()  # (T-1, 4)

    return contact


def motion147_to_151(
    motion_147: Tensor,
    bone_offsets: Tensor,
    velocity_threshold: float = 0.002,
) -> Tensor:
    """Convert 147-dim motion to 151-dim by computing foot contact.

    Args:
        motion_147: (L, 147) or (B, L, 147) motion tensor.
        bone_offsets: (22, 3) bone offsets tensor.
        velocity_threshold: Velocity threshold for contact detection.

    Returns:
        (L, 151) or (B, L, 151) motion tensor with foot contact appended.
    """
    from motius.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    # Handle batch dimension
    if motion_147.dim() == 3:
        B, L, D = motion_147.shape
        contact_list = []
        for i in range(B):
            m_151 = motion147_to_151(motion_147[i], bone_offsets, velocity_threshold)
            contact_list.append(m_151)
        return torch.stack(contact_list, dim=0)

    # Single sequence
    assert motion_147.dim() == 2, f"Expected 2D or 3D motion, got {motion_147.dim()}D"
    L, D = motion_147.shape
    assert D == 147, f"Expected 147-dim motion, got {D}"

    device = motion_147.device
    dtype = motion_147.dtype

    # Extract components
    trans = motion_147[:, 0:3]          # (L, 3)
    rot6d = motion_147[:, 3:135]       # (L, 132)
    pos_pred = motion_147[:, 135:147]  # (L, 12)

    # Construct 135-dim for FK
    motion_135 = torch.cat([trans, rot6d], dim=-1)  # (L, 135)

    # Compute FK to get world-space positions
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_135, bone_offsets, rotation_space='local')
    # world_pos: (L, 22, 3)

    # Extract end-effector positions: [L_Foot(10), R_Foot(11), L_Wrist(20), R_Wrist(21)]
    ee_indices = [10, 11, 20, 21]
    ee_pos = world_pos[:, ee_indices, :]  # (L, 4, 3)

    # Detect foot contact
    contact = detect_foot_contact(ee_pos, velocity_threshold)  # (L-1, 4)

    # Pad contact to match length L (duplicate first frame contact as first element)
    if contact.shape[0] < L:
        # Replicate first contact for the first frame
        contact_padded = torch.cat([contact[:1], contact], dim=0)
    else:
        contact_padded = contact

    # Ensure correct length
    if contact_padded.shape[0] != L:
        # Should not happen, but handle gracefully
        contact_padded = contact_padded[:L]
        if contact_padded.shape[0] < L:
            pad_size = L - contact_padded.shape[0]
            contact_padded = torch.cat(
                [contact_padded, contact_padded[-1:].expand(pad_size, -1)],
                dim=0
            )

    # Combine 147-dim + 4-dim contact -> 151-dim
    motion_151 = torch.cat([motion_147, contact_padded], dim=-1)  # (L, 151)

    return motion_151


@TRANSFORMS.register_module(force=True)
class Compute151DimFootContact(BaseTransform):
    """Compute foot contact binary indicators and append to 147-dim motion.

    This transform uses differentiable FK to compute end-effector positions,
    then detects contact based on joint velocity. Results in 151-dim motion:
    147-dim base + 4-dim contact indicators.

    Args:
        key: Key to motion data (default 'motion').
        bone_offsets_dir: Directory containing bone_offsets tensor. If None,
            loads from standard location.
        velocity_threshold: Velocity threshold for contact detection (default 0.002).
    """

    def __init__(
        self,
        key: str = 'motion',
        bone_offsets_dir: Optional[str] = None,
        velocity_threshold: float = 0.002,
    ):
        super().__init__()
        self.key = key
        self.bone_offsets_dir = bone_offsets_dir or 'data/bone_offsets'
        self.velocity_threshold = velocity_threshold

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
        from motius.datasets.motion.motionhub.smpl_data import SMPL22_BONE_OFFSETS
        self._bone_offsets_cache = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
        return self._bone_offsets_cache

    def transform(self, results: Dict) -> Dict:
        """Apply the transform.

        Args:
            results: Dictionary with results[f'{key}'].

        Returns:
            Updated results with motion expanded to 151-dim.
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
                if m.shape[-1] == 147:
                    bone_offsets = self._load_bone_offsets()
                    motion[person_key] = motion147_to_151(m, bone_offsets, self.velocity_threshold)
        else:
            # Single-person case
            if motion.shape[-1] == 147:
                bone_offsets = self._load_bone_offsets()
                results[motion_key] = motion147_to_151(motion, bone_offsets, self.velocity_threshold)
            elif motion.shape[-1] != 151:
                raise ValueError(
                    f"Motion dimension must be 147 or 151, got {motion.shape[-1]}"
                )

        return results


__all__ = [
    'Compute151DimFootContact',
    'detect_foot_contact',
    'motion147_to_151',
]
