# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I/O helpers for saving and loading SOMA animation parameters.

An .npz file contains everything needed to replay an animation with
SOMALayer: identity model type, identity coefficients, poses, root
translation, and metadata describing the representation.
"""

from pathlib import Path

import numpy as np
import torch

from .units import Unit


def add_npz_args(parser):
    """Add common NPZ output arguments to an argparse parser."""
    parser.add_argument(
        "--output-npz",
        default=None,
        help="Output .npz file with SOMA pose parameters.",
    )
    parser.add_argument(
        "--keep-root",
        action="store_true",
        help="Include the virtual Root joint (J=78). Off by default (J=77) "
        "to match SOMALayer.pose() input convention.",
    )
    parser.add_argument(
        "--output-unit",
        choices=[u.unit_name for u in Unit],
        default=Unit.METERS.unit_name,
        help="Unit for translational quantities in the output .npz. Default: meters.",
    )


def _to_f32(x):
    """Convert tensor or array to float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    return arr


def save_soma_npz(
    out_path,
    poses,
    transl,
    *,
    joint_names,
    identity_model_type,
    identity_coeffs,
    scale_params=None,
    joint_orient=None,
    unit="meters",
    keep_root=False,
    extra_arrays=None,
):
    """Save SOMA animation to an .npz file.

    This function performs no conversion — it saves data exactly as
    provided.  Callers are responsible for unit conversion, rotation
    representation, and absolute-to-relative conversion beforehand.

    The rotation representation is inferred from the shape of *poses*:
    ``(N, J, 3)`` for axis-angle (rotvec), ``(N, J, 3, 3)`` for matrices.

    Whether the poses are absolute or relative to the T-pose is inferred
    from *joint_orient*: if provided, the poses are assumed to be
    T-pose-relative (and the joint orient is stored so a reader can
    convert back if needed).  If omitted, the poses are assumed absolute.

    Args:
        out_path: Output file path.
        poses: (N, J, 3) axis-angle or (N, J, 3, 3) rotation matrices,
            matching ``SOMALayer.pose()``'s *poses* argument.
        transl: (N, 3) root translation, matching ``SOMALayer.pose()``'s
            *transl* argument.
        joint_names: list of J joint name strings.
        identity_model_type: string identifying the identity model
            (e.g. ``"smpl"``, ``"mhr"``, ``"anny"``).
        identity_coeffs: (N, C) or (1, C) identity coefficients.
        scale_params: (N, S) or (1, S) optional body-part scale
            parameters (e.g. required for MHR and Anny).
        joint_orient: (J, 3, 3) per-joint orientation from
            :func:`~soma.geometry.rig_utils.precompute_joint_orient`.
            If provided, poses are stored as T-pose-relative.  If None,
            poses are stored as absolute.
        unit: Unit label for translational quantities (``"meters"``,
            ``"centimeters"``, or ``"millimeters"``).
        keep_root: Include virtual Root joint (J=78 vs J=77).
        extra_arrays: Optional dict of additional arrays to include.
    """
    joint_names = list(joint_names)

    poses_np = _to_f32(poses)
    ndim = poses_np.ndim
    if ndim == 3 and poses_np.shape[-1] == 3:
        rotation_repr = "rotvec"
    elif ndim == 4 and poses_np.shape[-2:] == (3, 3):
        rotation_repr = "matrix"
    else:
        raise ValueError(
            f"Cannot infer rotation representation from poses shape {poses_np.shape}. "
            "Expected (N, J, 3) for rotvec or (N, J, 3, 3) for matrix."
        )

    absolute_pose = joint_orient is None

    # Strip Root joint (index 0) unless keep_root
    if not keep_root:
        poses_np = poses_np[:, 1:]
        joint_names = joint_names[1:]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        # Pose data
        "poses": poses_np,
        "transl": _to_f32(transl),
        "joint_names": np.array(joint_names),
        # Identity
        "identity_model_type": np.array(identity_model_type),
        "identity_coeffs": _to_f32(identity_coeffs),
        # Metadata
        "keep_root": np.bool_(keep_root),
        "absolute_pose": np.bool_(absolute_pose),
        "rotation_repr": np.array(rotation_repr),
        "unit": np.array(unit),
    }
    if scale_params is not None:
        save_dict["scale_params"] = _to_f32(scale_params)
    if joint_orient is not None:
        save_dict["joint_orient"] = _to_f32(joint_orient)
    if extra_arrays:
        save_dict.update(extra_arrays)

    np.savez_compressed(str(out_path), allow_pickle=False, **save_dict)

    pose_label = "absolute" if absolute_pose else "relative"
    root_label = "with Root (J=78)" if keep_root else "no Root (J=77)"
    print(f"\nSaved: {out_path}")
    print(f"  identity_model_type: {identity_model_type}")
    print(f"  identity_coeffs: {_to_f32(identity_coeffs).shape}")
    if scale_params is not None:
        print(f"  scale_params: {_to_f32(scale_params).shape}")
    print(f"  poses: {poses_np.shape} ({rotation_repr}, {pose_label}, {root_label})")
    print(f"  transl: {_to_f32(transl).shape} ({unit})")
    print(f"  joint_names: {len(joint_names)} joints")
