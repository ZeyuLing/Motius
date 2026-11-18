"""Shared position-to-rotation IK for SMPL-22 target skeletons.

Representation adapters decode positions and provide rest joints. All
cross-representation position-to-SMPL rotations use this outgoing-frame solve;
source representation rotation channels are never mixed into the result.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from motius.motion.skeleton.names import SMPL22_PARENTS
from .limb_frames import limb_orientation_guides

N_JOINTS = len(SMPL22_PARENTS)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = n[..., 0] > eps
    return v / np.maximum(n, eps), valid


def _preserve_twist_continuity(
    current: np.ndarray,
    previous: np.ndarray,
    bone_axis: np.ndarray,
) -> np.ndarray:
    """Choose the one-bone IK twist closest to the preceding frame.

    Aligning one rest bone to one observed bone determines swing but leaves
    rotation around the rest bone unconstrained. Right-multiplying by a twist
    around that rest axis preserves the observed child position. Projecting the
    current-to-previous relative quaternion onto that axis selects the smoothest
    member of this equivalent family without changing the positional fit.
    """

    axis = np.array(bone_axis, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return current
    axis /= norm
    relative = np.asarray(current, dtype=np.float64).T @ np.asarray(
        previous, dtype=np.float64
    )
    quat = R.from_matrix(relative).as_quat()
    projected = axis * float(np.dot(quat[:3], axis))
    twist = np.concatenate([projected, quat[3:4]])
    twist_norm = float(np.linalg.norm(twist))
    if twist_norm <= 1e-8:
        return current
    twist /= twist_norm
    return np.asarray(current, dtype=np.float64) @ R.from_quat(twist).as_matrix()


def estimate_local_rotations(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
    temporal_twist_stabilization: bool = True,
    *,
    orientation_guides: Callable[[int, int, list, list, list], None] | None = None,
) -> np.ndarray:
    """Estimate local rotations by aligning SMPL rest bones to target bones.

    Optional ``orientation_guides`` appends weighted rest/target vectors derived
    from extra observed landmarks (e.g. SOMA fingers). It supplies constraints
    to this same solver; it must not return or replace joint rotations.
    """
    target_joints = np.asarray(target_joints, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_JOINTS], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(N_JOINTS)]
    for j in range(1, N_JOINTS):
        p = int(parents[j])
        if 0 <= p < N_JOINTS:
            children[p].append(j)

    offsets = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for j in range(1, N_JOINTS):
        offsets[j] = rest_joints[j] - rest_joints[int(parents[j])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))
    limb_guides = limb_orientation_guides(target_joints, rest_joints, parents) if orientation_mode == "bone" else {}

    for t, joints in enumerate(target_joints):
        for j in range(N_JOINTS):
            child_ids = children[j]
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            rest_vecs_list = [offsets[c] for c in child_ids]
            target_vecs_list = [joints[c] - joints[j] for c in child_ids]
            weights = [1.0] * len(rest_vecs_list)
            if (t, j) in limb_guides:
                rest_guide, target_guide = limb_guides[t, j]
                rest_vecs_list.append(rest_guide)
                target_vecs_list.append(target_guide)
                weights.append(1.0)
            if orientation_mode == "parent_frame" and parent >= 0:
                # Position-only IK leaves twist around a single bone undefined.
                # A weak joint-to-parent reference chooses a stable local frame
                # without letting the virtual axis dominate child-bone fitting.
                rest_vecs_list.append(rest_joints[parent] - rest_joints[j])
                target_vecs_list.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            if orientation_guides is not None:
                orientation_guides(t, j, rest_vecs_list, target_vecs_list, weights)
            if not rest_vecs_list:
                local[t, j] = np.eye(3)
                global_r[t, j] = parent_global @ local[t, j]
                continue

            rest_vecs = np.stack(rest_vecs_list, axis=0)
            target_vecs = np.stack(target_vecs_list, axis=0)
            rest_unit, rest_valid = _safe_normalize(rest_vecs)
            target_unit, target_valid = _safe_normalize(target_vecs)
            valid = rest_valid & target_valid
            if not np.any(valid):
                rot_local = np.eye(3)
            else:
                src = rest_unit[valid]
                dst_world = target_unit[valid]
                dst_local = (parent_global.T @ dst_world.T).T
                valid_weights = np.asarray(weights, dtype=np.float64)[valid]
                if temporal_twist_stabilization and t > 0 and len(src) == 1 and orientation_mode == "bone":
                    # Transport the previous frame directly. Solving from rest
                    # first is numerically unstable when the bone crosses its
                    # rest antipode, even if the inter-frame change is tiny.
                    previous = local[t - 1, j]
                    correction = R.align_vectors(dst_local, (previous @ src.T).T)[0].as_matrix()
                    rot_local = correction @ previous
                else:
                    rot_local = R.align_vectors(dst_local, src, weights=valid_weights)[0].as_matrix()
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)
