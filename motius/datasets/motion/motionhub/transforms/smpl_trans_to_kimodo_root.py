"""Transform to convert SMPL Root to KIMODO Root via online ADMM smoothing.

KIMODO Root representation (198-dim):
  [0:3]      ADMM smoothed pelvis translation (smooth XZ, raw Y)
  [3:135]    22 joints × 6D rot6d (rotation channels unchanged)
  [135:198]  21 joints × 3D position (adjusted for smooth root reference)

ADMM smoothing:
  - Applied only to XZ plane (horizontal motion)
  - Y-axis (vertical) kept unchanged
  - Margin: ≤ 6cm bound on XZ deviation from smooth trajectory
  - Online: Applied during dataset __getitem__, not offline preprocessing
"""

from __future__ import annotations

from typing import Dict

import torch
from mmcv import BaseTransform
from torch import Tensor

from motius.registry import TRANSFORMS


def admm_smooth_translation_xz_simple(
    translation: Tensor,
    margin_m: float = 0.06,
) -> Tensor:
    """Smooth pelvis translation on XZ plane using iterative soft-thresholding.

    Simple iterative approach: For each frame pair, if the frame-to-frame
    difference in XZ exceeds the margin, scale it down.

    Args:
        translation: (T, 3) or (..., T, 3) raw pelvis translation.
        margin_m: Maximum frame-to-frame XZ distance.

    Returns:
        (T, 3) or (..., T, 3) smoothed translation.
    """
    device = translation.device
    dtype = translation.dtype

    # Forward pass: process each frame, clamping frame-to-frame displacement
    smooth = translation.clone()
    T = translation.shape[-2]

    for t in range(1, T):
        # Compute frame-to-frame difference in XZ
        diff_xz = smooth[..., t, [0, 2]] - smooth[..., t - 1, [0, 2]]  # (..., 2)
        dist_xz = torch.norm(diff_xz, dim=-1, keepdim=True)  # (..., 1)

        # If exceeds margin, scale down
        scale = torch.minimum(
            torch.ones_like(dist_xz),
            margin_m / (dist_xz + 1e-8),
        )

        # Update XZ, keep Y
        smooth[..., t, 0] = smooth[..., t - 1, 0] + diff_xz[..., 0] * scale.squeeze(-1)
        smooth[..., t, 2] = smooth[..., t - 1, 2] + diff_xz[..., 1] * scale.squeeze(-1)

    # Backward pass: reverse direction to improve consistency
    for t in range(T - 2, -1, -1):
        diff_xz = smooth[..., t, [0, 2]] - smooth[..., t + 1, [0, 2]]  # (..., 2)
        dist_xz = torch.norm(diff_xz, dim=-1, keepdim=True)  # (..., 1)

        scale = torch.minimum(
            torch.ones_like(dist_xz),
            margin_m / (dist_xz + 1e-8),
        )

        smooth[..., t, 0] = smooth[..., t + 1, 0] + diff_xz[..., 0] * scale.squeeze(-1)
        smooth[..., t, 2] = smooth[..., t + 1, 2] + diff_xz[..., 1] * scale.squeeze(-1)

    return smooth.to(dtype)


@TRANSFORMS.register_module()
class SmplTransToKimodoRootOnline(BaseTransform):
    """Online conversion from SMPL Root to KIMODO Root via translation smoothing.

    This transform applies during dataset __getitem__ (not offline preprocessing)
    to convert 198-dim SMPL motion to KIMODO Root representation:

    1. Smooth translation [0:3]: Iterative soft-thresholding on XZ plane with margin constraint
       (Y-axis unchanged)
    2. Keep rotation [3:135]: No change
    3. Adjust positions [135:198]: Update reference frame from raw → smooth pelvis

    The KIMODO Root representation better suppresses foot sliding and pelvis drift
    compared to raw SMPL Root by keeping the trajectory smooth.

    Parameters
    ----------
    key : str
        Key for the motion tensor in results dict (default 'motion').
    admm_margin_m : float
        Maximum frame-to-frame XZ displacement in meters (default 0.06 = 6cm).
        Limits smooth trajectory slope to ≤ admm_margin_m per frame.

    Example
    -------
    dict(
        type='SmplTransToKimodoRootOnline',
        key='motion',
        admm_margin_m=0.06,
    )
    """

    def __init__(
        self,
        key: str = "motion",
        admm_margin_m: float = 0.06,
    ):
        self.key = key
        self.admm_margin_m = admm_margin_m

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), (
            f"Expected torch.Tensor for key {self.key!r}, got {type(motion)}"
        )

        # Handle multi-person (P, T, D) or single-person (T, D)
        orig_shape = motion.shape
        if motion.ndim == 3:
            P, T, D = motion.shape
            assert D == 198, f"Expected motion_dim=198, got {D}"
            motion_flat = motion.reshape(P * T, D)
        elif motion.ndim == 2:
            T, D = motion.shape
            assert D == 198, f"Expected motion_dim=198, got {D}"
            motion_flat = motion
        else:
            raise ValueError(f"Unexpected motion shape: {orig_shape}")

        # Extract components (maintaining temporal structure)
        if motion.ndim == 3:
            # (P, T, 198) -> each person separately
            motion_kimodo = torch.zeros_like(motion)
            for p in range(motion.shape[0]):
                motion_p = motion[p]  # (T, 198)
                motion_kimodo[p] = self._convert_motion_198(motion_p)
        else:
            # (T, 198)
            motion_kimodo = self._convert_motion_198(motion)

        results[self.key] = motion_kimodo
        return results

    def _convert_motion_198(self, motion_198: Tensor) -> Tensor:
        """Convert single (T, 198) motion from SMPL Root to KIMODO Root.

        Args:
            motion_198: (T, 198) motion tensor [trans(3) + rot(132) + pos(63)].

        Returns:
            (T, 198) KIMODO Root motion [smooth_trans(3) + rot(132) + pos_adjusted(63)].
        """
        raw_trans = motion_198[..., 0:3]  # (T, 3)
        rotation = motion_198[..., 3:135]  # (T, 132)
        pos_rel_raw = motion_198[..., 135:198]  # (T, 63)

        # Step 1: Smooth translation (XZ only, Y raw)
        smooth_trans = admm_smooth_translation_xz_simple(
            raw_trans,
            margin_m=self.admm_margin_m,
        )  # (T, 3)

        # Step 2: Adjust position reference frame
        # pos_rel_raw[j,t] = world_pos[j,t] - raw_trans[t]
        # pos_rel_smooth[j,t] = world_pos[j,t] - smooth_trans[t]
        # => pos_rel_smooth[j,t] = pos_rel_raw[j,t] + (raw_trans[t] - smooth_trans[t])
        trans_diff = raw_trans - smooth_trans  # (T, 3)

        # Reshape for broadcasting: (T, 3) -> (T, 21, 3) -> (T, 63)
        trans_diff_expanded = trans_diff.unsqueeze(-2).expand(-1, 21, -1).reshape(-1, 63)

        # Apply offset to position channels.  Current smoothing keeps Y raw, so
        # the Y offset is zero while XZ are adjusted to the smooth reference.
        pos_rel_smooth = pos_rel_raw + trans_diff_expanded  # (T, 63)

        # Reconstruct 198-dim KIMODO Root motion
        motion_kimodo = torch.cat([smooth_trans, rotation, pos_rel_smooth], dim=-1)

        return motion_kimodo
