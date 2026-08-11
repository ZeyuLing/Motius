"""Compute HYMotion-aligned 198/201-dim O6DP motion representations.

The 198-dim representation used by M2M is the official HY-Motion Lite 201-dim
O6DP representation with the redundant pelvis RIC triplet removed:

    198 = 201[:135] + 201[138:201]

The transform:
1. Runs differentiable FK on the 135-dim motion (always using LOCAL rotation)
2. Computes the official 66-dim root-invariant coordinate (RIC) block
3. Removes the pelvis RIC triplet (always [0, 0, 0])
4. Concatenates position channels to produce 198-dim output

Usage in pipeline:
    # For local rotation models:
    dict(type='LoadSmplx55', ...),
    dict(type='Compute198DimPosition', key='motion'),  # 135 -> 198
    dict(type='RandomCropPadding', ...),

    # For global rotation models:
    dict(type='LoadSmplx55', ...),
    dict(type='Compute198DimPosition', key='motion'),  # 135 -> 198 (FK uses local rot)
    dict(type='LocalToGlobalRotation', key='motion'),  # rotation channels -> global
    dict(type='RandomCropPadding', ...),

    Note: Compute198DimPosition MUST come BEFORE LocalToGlobalRotation because
    FK requires local rotation. LocalToGlobalRotation only changes the rotation
    channels (dims 3:135), position channels (dims 135:198) are unaffected.
"""

from __future__ import annotations

import os.path as osp
from typing import Dict, Optional, Tuple

import torch
from mmcv import BaseTransform
from torch import Tensor

from motius.registry import TRANSFORMS


def compute_position_channels(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Compute strict HYMotion 63-dim non-pelvis RIC channels.

    Args:
        motion_135: (*, 135) motion tensor in local rotation space.
        bone_offsets: (22, 3) bone offsets tensor.

    Returns:
        (*, 63) position channels for joints 1:22, root-relative in XYZ.
    """
    ric66 = compute_ric66_channels(motion_135, bone_offsets)
    return ric66[..., 3:66]


def motion135_to_198(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Convert 135-dim motion to strict HYMotion 198-dim O6DP.

    Args:
        motion_135: (*, 135) motion tensor.
        bone_offsets: (22, 3) bone offsets.

    Returns:
        (*, 198) motion tensor.
    """
    pos_63 = compute_position_channels(motion_135, bone_offsets)
    return torch.cat([motion_135, pos_63], dim=-1)


def motion201_to_198(motion_201: Tensor, pelvis_eps: float = 1e-6) -> Tensor:
    """Drop the redundant pelvis RIC triplet from official HYMotion 201-dim O6DP.

    The official 201-dim layout is:
      [0:135]   translation + rot6d
      [135:138] pelvis RIC, expected to be exactly zero
      [138:201] non-pelvis joint RIC
    """
    if motion_201.shape[-1] != 201:
        raise ValueError(f"Expected last dim 201, got {motion_201.shape[-1]}")
    pelvis = motion_201[..., 135:138]
    if torch.max(torch.abs(pelvis)).item() > pelvis_eps:
        raise ValueError(
            "HYMotion 201-dim pelvis RIC channels are expected to be zero; "
            f"max_abs={torch.max(torch.abs(pelvis)).item():.6g}"
        )
    return torch.cat([motion_201[..., :135], motion_201[..., 138:201]], dim=-1)


def compute_ric66_channels(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Compute HY-Motion Lite 66-dim RIC channels from 135-dim motion.

    HY-Motion 1.0 Lite uses 22 joints x 3 local positions appended after the
    135-dim translation + rotation channels. The pelvis position is always
    exactly zero after root-relative conversion.
    """
    from motius.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    leading = motion_135.shape[:-1]

    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_135, bone_offsets, rotation_space='local')

    pelvis_world = world_pos[..., 0:1, :]
    joint_pos = world_pos - pelvis_world
    return joint_pos.reshape(*leading, 66)


def motion135_to_201(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Convert 135-dim motion to HY-Motion Lite 201-dim O6DP motion."""
    pos_66 = compute_ric66_channels(motion_135, bone_offsets)
    return torch.cat([motion_135, pos_66], dim=-1)


def motion198_to_135(motion_198: Tensor) -> Tensor:
    """Extract 135-dim (trans + rot6d) from 198-dim motion.

    Simply takes the first 135 dimensions, discarding position channels.

    Args:
        motion_198: (*, 198) motion tensor.

    Returns:
        (*, 135) motion tensor.
    """
    return motion_198[..., :135]


def recompute_position_from_rotation(
    motion_198: Tensor,
    bone_offsets: Tensor,
    rotation_space: str = 'local',
) -> Tensor:
    """Recompute strict 198-dim position channels from rotation.

    This is used for FK consistency loss: given a 198-dim prediction, extract
    the rotation+translation (first 135 dims), run FK, and compute what the
    position channels SHOULD be. Compare with the predicted position channels
    to enforce FK consistency.

    Args:
        motion_198: (*, 198) denormalized motion tensor.
        bone_offsets: (22, 3) bone offsets.
        rotation_space: 'local' or 'global'.

    Returns:
        (*, 63) FK-derived non-pelvis RIC channels.
    """
    motion_135 = motion_198[..., :135]

    # If global rotation, convert to local for FK
    if rotation_space == 'global':
        from motius.motion.pipeline_utils.differentiable_fk import motion135_to_fk
        # motion135_to_fk handles global->local conversion internally
        world_pos, _, _, _ = motion135_to_fk(motion_135, bone_offsets, rotation_space='global')
    else:
        from motius.motion.pipeline_utils.differentiable_fk import motion135_to_fk
        world_pos, _, _, _ = motion135_to_fk(motion_135, bone_offsets, rotation_space='local')

    pelvis_world = world_pos[..., 0:1, :]
    joint_pos = world_pos[..., 1:, :] - pelvis_world

    leading = motion_198.shape[:-1]
    return joint_pos.reshape(*leading, 63)


def motion198_fk_loss(
    pred_198_norm: Tensor,
    mean: Tensor,
    std: Tensor,
    bone_offsets: Tensor,
    rotation_space: str = 'local',
    timesteps: Optional[Tensor] = None,
    data_mask_temporal: Optional[Tensor] = None,
) -> Tensor:
    """Compute FK consistency loss between predicted position and FK-derived position.

    The loss penalizes inconsistency between the rotation/translation channels
    and the position channels in the 198-dim prediction.

    Args:
        pred_198_norm: (B, L, 198) predicted motion in NORMALIZED space.
        mean: (198,) normalization mean.
        std: (198,) normalization std.
        bone_offsets: (22, 3) bone offsets.
        rotation_space: 'local' or 'global'.
        timesteps: (B,) diffusion timesteps for t² weighting. If None, no weighting.
        data_mask_temporal: (B, L) boolean/{0,1} mask, 1 = valid frame, 0 = padded.
            When provided, padded frames are excluded from the loss so the
            replicated tail (RandomCropPadding pad_mode='replicate') and
            zeroed-out tgt frames cannot leak into the FK-consistency signal.

    Returns:
        Scalar FK consistency loss.
    """
    std_safe = torch.where(std < 1e-3, torch.ones_like(std), std)
    pred_denorm = pred_198_norm * std_safe + mean

    pred_pos = pred_denorm[..., 135:]  # (B, L, 63)

    fk_pos = recompute_position_from_rotation(
        pred_denorm, bone_offsets, rotation_space
    )  # (B, L, 63)

    loss_per_dim = torch.nn.functional.smooth_l1_loss(
        pred_pos, fk_pos, reduction='none'
    )  # (B, L, 63)

    loss = loss_per_dim.mean(dim=-1)  # (B, L)

    if timesteps is not None:
        t_sq = (timesteps ** 2).unsqueeze(-1)  # (B, 1)
        loss = loss * t_sq

    if data_mask_temporal is not None:
        m = data_mask_temporal.to(loss.device).to(loss.dtype)
        # Align temporal length: prepare_padding may have prepended ref_pose
        # tokens onto tgt_padding_mask, making it longer than the FK loss
        # (which only covers tgt_motion frames). Slice from the right.
        if m.shape[-1] != loss.shape[-1]:
            m = m[..., -loss.shape[-1]:]
        denom = torch.clamp(m.sum(), min=1.0)
        return (loss * m).sum() / denom

    return loss.mean()


def motion198_get_positions(motion_198: Tensor) -> Tensor:
    """Extract 63-dim position channels from 198-dim motion.

    Args:
        motion_198: (*, 198) motion tensor.

    Returns:
        (*, 63) position channels.
    """
    return motion_198[..., 135:]


# Default bone offsets path (relative to repo root)
_DEFAULT_BONE_OFFSETS_PATH = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))),
    'data', 'hymotion_m2m_data', 'bone_offsets_22.pt',
)


@TRANSFORMS.register_module()
class Compute198DimPosition(BaseTransform):
    """Transform to compute 198-dim motion from 135/201-dim motion.

    When the input is official HY-Motion Lite O6DP-201, this transform simply
    removes the redundant pelvis RIC triplet [135:138].  When the input is
    135-dim translation+rotation, it computes the official 201-dim RIC block
    via FK and then removes the pelvis triplet.  In both cases the output is
    aligned with the HYMotion-Lite 201-dim checkpoint adapter.

    Must be placed BEFORE LocalToGlobalRotation in the pipeline
    (FK requires local rotation).

    Parameters
    ----------
    key : str
        Key for the motion tensor in results dict.
    bone_offsets_path : str or None
        Path to bone offsets file. If None, uses default path.
    """

    def __init__(
        self,
        key: str = 'motion',
        bone_offsets_path: Optional[str] = None,
    ):
        self.key = key
        self._bone_offsets_path = bone_offsets_path or _DEFAULT_BONE_OFFSETS_PATH
        self._bone_offsets: Optional[Tensor] = None

    def _load_bone_offsets(self) -> Tensor:
        if self._bone_offsets is None:
            path = self._bone_offsets_path
            if not osp.isfile(path):
                raise FileNotFoundError(
                    f'Bone offsets not found at {path}. '
                    'Run: PYTHONPATH=. python3 tools/precompute_bone_offsets.py first.'
                )
            self._bone_offsets = torch.load(path, map_location='cpu').float()
        return self._bone_offsets

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), (
            f'Expected torch.Tensor for key {self.key!r}, got {type(motion)}'
        )

        orig_shape = motion.shape
        if motion.ndim == 3:
            P, T, D = motion.shape
            assert D in (135, 201), f"Expected motion_dim=135 or 201, got {D}"
            if D == 201:
                motion_198 = motion201_to_198(motion.reshape(P * T, D)).reshape(P, T, 198)
                results[self.key] = motion_198
                return results
            motion_flat = motion.reshape(P * T, 135)
        elif motion.ndim == 2:
            T, D = motion.shape
            assert D in (135, 201), f"Expected motion_dim=135 or 201, got {D}"
            if D == 201:
                results[self.key] = motion201_to_198(motion)
                return results
            motion_flat = motion
        else:
            raise ValueError(f"Unexpected motion shape: {orig_shape}")

        bone_offsets = self._load_bone_offsets()
        motion_198 = motion135_to_198(motion_flat, bone_offsets)

        if motion.ndim == 3:
            motion_198 = motion_198.reshape(P, T, 198)
        else:
            motion_198 = motion_198.reshape(T, 198)

        results[self.key] = motion_198
        return results


@TRANSFORMS.register_module()
class Compute201DimO6DP(BaseTransform):
    """Transform to compute HY-Motion Lite 201-dim O6DP motion from 135-dim.

    HY-Motion Lite's 201-dim representation appends all 22 root-relative joint
    positions and keeps the pelvis triplet fixed at zero.
    """

    def __init__(
        self,
        key: str = 'motion',
        bone_offsets_path: Optional[str] = None,
    ):
        self.key = key
        self._bone_offsets_path = bone_offsets_path or _DEFAULT_BONE_OFFSETS_PATH
        self._bone_offsets: Optional[Tensor] = None

    def _load_bone_offsets(self) -> Tensor:
        if self._bone_offsets is None:
            path = self._bone_offsets_path
            if not osp.isfile(path):
                raise FileNotFoundError(
                    f'Bone offsets not found at {path}. '
                    'Run: PYTHONPATH=. python3 tools/precompute_bone_offsets.py first.'
                )
            self._bone_offsets = torch.load(path, map_location='cpu').float()
        return self._bone_offsets

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), (
            f'Expected torch.Tensor for key {self.key!r}, got {type(motion)}'
        )

        orig_shape = motion.shape
        if motion.ndim == 3:
            P, T, D = motion.shape
            assert D == 135, f"Expected motion_dim=135, got {D}"
            motion_flat = motion.reshape(P * T, D)
        elif motion.ndim == 2:
            T, D = motion.shape
            assert D == 135, f"Expected motion_dim=135, got {D}"
            motion_flat = motion
        else:
            raise ValueError(f"Unexpected motion shape: {orig_shape}")

        bone_offsets = self._load_bone_offsets()
        motion_201 = motion135_to_201(motion_flat, bone_offsets)

        if motion.ndim == 3:
            motion_201 = motion_201.reshape(P, T, 201)
        else:
            motion_201 = motion_201.reshape(T, 201)

        results[self.key] = motion_201
        return results
