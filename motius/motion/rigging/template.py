"""Deterministic humanoid skeleton fitting and skin-weight estimation.

The numerical core intentionally depends only on NumPy.  Blender loads this
module in its bundled Python when it imports and binds a character asset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

# Kept local so Blender can load this file without importing the Torch-based
# Motius package.  A unit test guards equality with the canonical FBX mapping.
SMPL22_RIG_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)

SMPL22_RIG_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int32,
)


@dataclass(frozen=True)
class TemplateRiggingConfig:
    """Controls sparse geometry-based skinning for the template rig."""

    top_k: int = 4
    weight_falloff: float = 1.75
    side_penalty: float = 0.025
    chunk_size: int = 16384

    def __post_init__(self) -> None:
        if not 1 <= int(self.top_k) <= 4:
            raise ValueError("top_k must be between 1 and 4 for GLTF/FBX skinning.")
        if not np.isfinite(self.weight_falloff) or self.weight_falloff <= 0:
            raise ValueError("weight_falloff must be a positive finite value.")
        if not np.isfinite(self.side_penalty) or not 0 < self.side_penalty <= 1:
            raise ValueError("side_penalty must be in the interval (0, 1].")
        if int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be positive.")


@dataclass(frozen=True)
class FittedHumanoidSkeleton:
    """A fitted, Blender-coordinate SMPL22 rest skeleton."""

    joints: np.ndarray
    names: tuple[str, ...]
    parents: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class SkinWeightResult:
    """Normalized vertex-by-joint weights and their diagnostics."""

    weights: np.ndarray
    diagnostics: Mapping[str, object]


def _vertices(value) -> np.ndarray:
    vertices = np.asarray(value, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V,3), got {vertices.shape}.")
    if len(vertices) < 4:
        raise ValueError("At least four vertices are required to fit a humanoid rig.")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite coordinates.")
    return vertices


def _nearest_distance(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    distances = np.empty(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        distances[index] = np.sqrt(np.min(np.sum((vertices - point) ** 2, axis=1)))
    return distances


def fit_humanoid_skeleton(vertices) -> FittedHumanoidSkeleton:
    """Fit a symmetric SMPL22-compatible skeleton to an upright humanoid mesh.

    The fit is scale free and expects Blender coordinates (Z up, -Y forward).
    Width controls whether the inferred arms are closer to a T or an A pose.
    The returned diagnostics are deliberately conservative: they describe
    geometric support and do not claim semantic understanding of arbitrary
    meshes.
    """

    vertices = _vertices(vertices)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = bounds_max - bounds_min
    height = float(extent[2])
    if height <= 1e-7:
        raise ValueError("Cannot fit a humanoid rig to a zero-height mesh.")

    width = float(extent[0])
    depth = float(extent[1])
    width_ratio = width / height
    depth_ratio = depth / height
    center_x = float((bounds_min[0] + bounds_max[0]) * 0.5)
    center_y = float((bounds_min[1] + bounds_max[1]) * 0.5)
    base_z = float(bounds_min[2])
    half_width = max(width * 0.5, height * 0.04)

    hip_x = min(height * 0.095, max(height * 0.028, half_width * 0.24))
    shoulder_x = min(
        height * 0.19,
        max(hip_x * 1.45, half_width * 0.36),
    )
    relaxed_factor = float(np.clip((0.58 - width_ratio) / 0.30, 0.0, 1.0))
    # The bounding-box extreme is normally a fingertip, not the SMPL wrist
    # joint. Move inward toward the palm root, with a little more inset for a
    # relaxed/A pose where fingers dominate the lateral silhouette.
    wrist_fraction = 0.86 - 0.08 * relaxed_factor
    wrist_limit = max(shoulder_x * 1.08, half_width * wrist_fraction)
    wrist_x = min(height * 0.46, wrist_limit)
    wrist_x = max(wrist_x, shoulder_x + min(height * 0.055, half_width * 0.2))

    shoulder_z = base_z + height * 0.785
    # Estimate an A/relaxed-pose wrist height from the outer silhouette.  A
    # width-only heuristic cannot distinguish a slightly narrow T pose from a
    # steep A pose and previously placed the wrists inside the chest of the
    # Blender CC0 validation mesh.  The outer 16% on either side is dominated
    # by hands/forearms for an upright, isolated humanoid; robust quantiles
    # avoid single fingertip or accessory vertices controlling the estimate.
    relative_x = np.abs(vertices[:, 0] - center_x)
    outer = relative_x >= np.quantile(relative_x, 0.84)
    outer_z = vertices[outer, 2]
    silhouette_wrist_z = (
        float(np.median(outer_z)) if len(outer_z) >= 16 else shoulder_z
    )
    minimum_wrist_z = base_z + height * 0.49
    maximum_wrist_z = shoulder_z - height * 0.015
    wrist_z = float(np.clip(silhouette_wrist_z, minimum_wrist_z, maximum_wrist_z))
    arm_drop = shoulder_z - wrist_z
    elbow_z = shoulder_z * 0.48 + wrist_z * 0.52
    foot_forward = min(height * 0.06, depth * 0.35)
    # SMPL/HumanML3D's Head joint denotes the cranium center, which is in
    # front of the cervical spine rather than vertically above Neck.  A
    # vertical rest Neck->Head bone makes this anatomical offset look like a
    # 40--50 degree nod during position-only retargeting.  Fit a forward rest
    # offset from mesh depth so the delta rotation represents actual head
    # motion instead of the joint-definition mismatch.
    head_forward = min(height * 0.075, depth * 0.44)

    def point(x: float, y: float, z_ratio: float | None = None, *, z=None):
        if z is None:
            z = base_z + height * float(z_ratio)
        return (center_x + x, center_y + y, float(z))

    joints = np.asarray(
        [
            point(0.0, 0.0, 0.535),
            point(+hip_x, 0.0, 0.505),
            point(-hip_x, 0.0, 0.505),
            point(0.0, 0.0, 0.610),
            point(+hip_x, 0.0, 0.285),
            point(-hip_x, 0.0, 0.285),
            point(0.0, 0.0, 0.685),
            point(+hip_x, 0.0, 0.065),
            point(-hip_x, 0.0, 0.065),
            point(0.0, 0.0, 0.760),
            point(+hip_x, -foot_forward, 0.035),
            point(-hip_x, -foot_forward, 0.035),
            point(0.0, 0.0, 0.855),
            point(+shoulder_x * 0.55, 0.0, 0.790),
            point(-shoulder_x * 0.55, 0.0, 0.790),
            point(0.0, -head_forward, 0.935),
            point(+shoulder_x, 0.0, z=shoulder_z),
            point(-shoulder_x, 0.0, z=shoulder_z),
            point(+(shoulder_x + wrist_x) * 0.5, 0.0, z=elbow_z),
            point(-(shoulder_x + wrist_x) * 0.5, 0.0, z=elbow_z),
            point(+wrist_x, 0.0, z=wrist_z),
            point(-wrist_x, 0.0, z=wrist_z),
        ],
        dtype=np.float64,
    )

    supported = (10, 11, 15, 20, 21)
    endpoint_distances = _nearest_distance(vertices, joints[list(supported)]) / height
    endpoint_error = float(np.mean(endpoint_distances))
    width_score = float(np.clip((width_ratio - 0.25) / 0.35, 0.0, 1.0))
    endpoint_score = float(np.clip(1.0 - endpoint_error / 0.12, 0.0, 1.0))
    quality_score = 0.55 * width_score + 0.45 * endpoint_score
    warnings: list[str] = []
    if width_ratio < 0.32:
        warnings.append(
            "The silhouette is narrow for a T/A pose; arm joints and arm weights may be unreliable."
        )
    if width_ratio > 1.20:
        warnings.append(
            "Mesh width exceeds height; verify that the character is upright and the up axis is correct."
        )
    if depth_ratio > 0.55:
        warnings.append(
            "Mesh depth is large relative to height; verify orientation and exclude non-character props."
        )
    if endpoint_error > 0.10:
        warnings.append(
            "Several template endpoints have weak surface support; inspect wrists, feet, and head before animation."
        )

    diagnostics = MappingProxyType(
        {
            "quality_score": float(quality_score),
            "pose_width_ratio": float(width_ratio),
            "depth_height_ratio": float(depth_ratio),
            "endpoint_surface_distance_mean_ratio": endpoint_error,
            "inferred_pose": "T" if arm_drop < height * 0.06 else "A/relaxed",
            "silhouette_wrist_height_ratio": float((wrist_z - base_z) / height),
            "warnings": tuple(warnings),
        }
    )
    return FittedHumanoidSkeleton(
        joints=np.asarray(joints, dtype=np.float32),
        names=SMPL22_RIG_NAMES,
        parents=SMPL22_RIG_PARENTS.copy(),
        bounds_min=np.asarray(bounds_min, dtype=np.float32),
        bounds_max=np.asarray(bounds_max, dtype=np.float32),
        diagnostics=diagnostics,
    )


def _segment_distances(
    vertices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    direction = ends - starts
    length_sq = np.sum(direction * direction, axis=1)
    relative = vertices[:, None, :] - starts[None, :, :]
    projection = np.einsum("vjc,jc->vj", relative, direction)
    projection /= np.maximum(length_sq[None, :], 1e-12)
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[..., None] * direction[None, :, :]
    return np.linalg.norm(vertices[:, None, :] - closest, axis=-1)


def compute_skin_weights(
    vertices,
    skeleton: FittedHumanoidSkeleton,
    *,
    config: TemplateRiggingConfig | None = None,
) -> SkinWeightResult:
    """Estimate normalized, at-most-four-joint weights from bone capsules."""

    config = config or TemplateRiggingConfig()
    vertices = _vertices(vertices)
    joints = np.asarray(skeleton.joints, dtype=np.float64)
    parents = np.asarray(skeleton.parents, dtype=np.int64)
    if joints.shape != (len(SMPL22_RIG_NAMES), 3):
        raise ValueError(f"skeleton joints must have shape (22,3), got {joints.shape}.")
    if parents.shape != (len(joints),):
        raise ValueError(
            f"skeleton parents must have shape (22,), got {parents.shape}."
        )

    # A vertex group is named after the joint whose rotation deforms the
    # outgoing body segment.  For example L_Hip owns the thigh (hip -> knee),
    # and L_Shoulder owns the upper arm (shoulder -> elbow).  Using the
    # incoming segment here silently shifts every limb by one joint: the thigh
    # follows the knee and the upper arm follows the elbow, producing severe
    # pinching as soon as the character moves.
    children = [np.flatnonzero(parents == joint) for joint in range(len(joints))]
    axial_children = {0: 3, 3: 6, 6: 9, 9: 12, 12: 15}
    starts = joints.copy()
    ends = joints.copy()
    for joint, child_ids in enumerate(children):
        preferred = axial_children.get(joint)
        if preferred is not None and preferred in child_ids:
            child = preferred
        elif len(child_ids):
            child = max(
                child_ids,
                key=lambda index: np.linalg.norm(joints[index] - joints[joint]),
            )
        else:
            parent = parents[joint]
            direction = (
                joints[joint] - joints[parent]
                if parent >= 0
                else np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            )
            ends[joint] = joints[joint] + direction * 0.45
            continue
        ends[joint] = joints[int(child)]

    scale = max(
        float(skeleton.bounds_max[2] - skeleton.bounds_min[2]),
        float(np.ptp(joints[:, 2])),
        1e-6,
    )
    lengths = np.linalg.norm(ends - starts, axis=1)
    radii = np.maximum(scale * 0.022, lengths * 0.22)
    torso = (0, 3, 6, 9, 12, 15)
    radii[list(torso)] = np.maximum(radii[list(torso)], scale * 0.075)
    radii[[1, 2, 13, 14, 16, 17]] = np.maximum(
        radii[[1, 2, 13, 14, 16, 17]], scale * 0.045
    )
    # Terminal joints need a spherical influence around their pivot. Their
    # proxy segment is artificial (there is no child landmark), so a radius
    # based on that short segment otherwise lets Neck/Elbow/Ankle own the
    # whole head, hand, or foot.
    radii[15] = max(radii[15], scale * 0.105)
    radii[[10, 11, 20, 21]] = np.maximum(
        radii[[10, 11, 20, 21]], scale * 0.048
    )

    weights = np.zeros((len(vertices), len(joints)), dtype=np.float32)
    center_x = float(joints[0, 0])
    neutral_band = scale * 0.015
    log_side_penalty = float(np.log(config.side_penalty))
    left_joints = np.asarray(
        [index for index, name in enumerate(skeleton.names) if name.startswith("L_")]
    )
    right_joints = np.asarray(
        [index for index, name in enumerate(skeleton.names) if name.startswith("R_")]
    )
    top_k = min(int(config.top_k), len(joints))

    for start in range(0, len(vertices), int(config.chunk_size)):
        stop = min(start + int(config.chunk_size), len(vertices))
        chunk = vertices[start:stop]
        distances = _segment_distances(chunk, starts, ends)
        logits = -0.5 * (distances / radii[None, :] * float(config.weight_falloff)) ** 2
        left_side = chunk[:, 0] < center_x - neutral_band
        right_side = chunk[:, 0] > center_x + neutral_band
        if np.any(left_side):
            logits[np.ix_(left_side, left_joints)] += log_side_penalty
        if np.any(right_side):
            logits[np.ix_(right_side, right_joints)] += log_side_penalty

        # Geometry-only endpoint priors resolve capsule ties at chain ends.
        # Masks are expressed relative to the fitted skeleton, with no
        # asset-specific vertex indices or source rig metadata.
        endpoint_penalty = float(np.log(0.035))
        head_floor = float((joints[12, 2] + joints[15, 2]) * 0.5)
        head_region = chunk[:, 2] >= head_floor
        if np.any(head_region):
            non_head = np.asarray(
                [index for index in range(len(joints)) if index != 15]
            )
            logits[np.ix_(head_region, non_head)] += endpoint_penalty

        for elbow, wrist in ((18, 20), (19, 21)):
            sign = 1.0 if joints[wrist, 0] > center_x else -1.0
            boundary = float(joints[elbow, 0] * 0.25 + joints[wrist, 0] * 0.75)
            hand_region = sign * chunk[:, 0] >= sign * boundary
            if np.any(hand_region):
                non_wrist = np.asarray(
                    [index for index in range(len(joints)) if index != wrist]
                )
                logits[np.ix_(hand_region, non_wrist)] += endpoint_penalty

        foot_ceiling = float(np.max(joints[[7, 8], 2]) + scale * 0.015)
        foot_region = chunk[:, 2] <= foot_ceiling
        if np.any(foot_region):
            foot_chunk = chunk[foot_region]
            logits[foot_region, 10] += np.where(
                foot_chunk[:, 0] >= center_x, 2.0, endpoint_penalty
            )
            logits[foot_region, 11] += np.where(
                foot_chunk[:, 0] <= center_x, 2.0, endpoint_penalty
            )

        selected = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        selected_logits = np.take_along_axis(logits, selected, axis=1)
        selected_logits -= selected_logits.max(axis=1, keepdims=True)
        selected_weights = np.exp(selected_logits)
        selected_weights /= selected_weights.sum(axis=1, keepdims=True)
        chunk_weights = np.zeros_like(logits, dtype=np.float32)
        np.put_along_axis(
            chunk_weights,
            selected,
            selected_weights.astype(np.float32),
            axis=1,
        )
        weights[start:stop] = chunk_weights

    sums = weights.sum(axis=1)
    active = np.flatnonzero(np.any(weights > 1e-7, axis=0))
    diagnostics = MappingProxyType(
        {
            "vertices": len(vertices),
            "top_k": int(top_k),
            "active_joints": len(active),
            "weight_sum_error_max": float(np.max(np.abs(sums - 1.0))),
            "dominant_weight_mean": float(np.mean(np.max(weights, axis=1))),
            "unbound_vertices": int(np.count_nonzero(sums <= 1e-8)),
        }
    )
    return SkinWeightResult(weights=weights, diagnostics=diagnostics)


__all__ = [
    "SMPL22_RIG_NAMES",
    "SMPL22_RIG_PARENTS",
    "FittedHumanoidSkeleton",
    "SkinWeightResult",
    "TemplateRiggingConfig",
    "compute_skin_weights",
    "fit_humanoid_skeleton",
]
