"""Online multi-person composition transform.

Composes multiple independent single-person motions into a collision-free
multi-person scene with automatically generated multi-person captions.
Used as a training-time data augmentation to dramatically increase the
amount of multi-person training data.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from scipy.spatial.transform import Rotation as R

from motius.datasets.motion.motionhub.common import read_json
from motius.datasets.motion.motionhub.transforms.load_smplx import (
    _build_Ry_from_deg,
    _read_one_person_npz,
    apply_root_yaw_to_axis_angle,
    process_smplx_pose,
    process_transl,
)
from motius.models.vermo.task_utils.modality import Audio
from motius.registry import TRANSFORMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMPL-22 kinematic tree (parent indices; -1 = root)
# ---------------------------------------------------------------------------
SMPL_22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

# ---------------------------------------------------------------------------
# Capsule definitions for collision detection
# (joint_a, joint_b) pairs defining each capsule's axis
# ---------------------------------------------------------------------------
CAPSULE_PAIRS: List[Tuple[int, int]] = [
    # Torso
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left arm
    (16, 18), (18, 20),
    # Right arm
    (17, 19), (19, 21),
    # Left leg
    (1, 4), (4, 7),
    # Right leg
    (2, 5), (5, 8),
]

# Per-capsule radii in meters
CAPSULE_RADII: List[float] = [
    0.12, 0.10, 0.10, 0.08, 0.10,  # torso
    0.06, 0.05,                      # left arm
    0.06, 0.05,                      # right arm
    0.08, 0.06,                      # left leg
    0.08, 0.06,                      # right leg
]


# ---------------------------------------------------------------------------
# Numpy forward kinematics for SMPL-22
# ---------------------------------------------------------------------------
def _rodrigues_batch(rotvecs: np.ndarray) -> np.ndarray:
    """Batch Rodrigues formula: axis-angle [N, 3] -> rotation matrices [N, 3, 3]."""
    return R.from_rotvec(rotvecs).as_matrix().astype(np.float32)


def _load_J_template(smplx_model_path: str, gender: str = "neutral") -> np.ndarray:
    """Load the 22-joint rest-pose template from an SMPLX npz file.

    Returns:
        J_template: [22, 3] rest-pose joint positions.
    """
    gender_map = {"neutral": "SMPLX_NEUTRAL", "male": "SMPLX_MALE", "female": "SMPLX_FEMALE"}
    npz_name = gender_map.get(gender, "SMPLX_NEUTRAL")
    npz_path = os.path.join(smplx_model_path, f"{npz_name}.npz")
    data = np.load(npz_path, allow_pickle=True)
    J_regressor = np.asarray(data["J_regressor"], dtype=np.float64)  # [55, 10475]
    v_template = np.asarray(data["v_template"], dtype=np.float64)    # [10475, 3]
    J_full = (J_regressor @ v_template)  # [55, 3]
    return J_full[:22].astype(np.float32)


def numpy_fk_22joints(
    transl: np.ndarray,
    poses_aa: np.ndarray,
    parents: List[int],
    J_template: np.ndarray,
) -> np.ndarray:
    """Pure-numpy forward kinematics for SMPL-22 joints.

    Args:
        transl: [T, 3] root translation (world coords).
        poses_aa: [T, 165] or [T, 55*3] axis-angle poses.
        parents: SMPL_22_PARENTS list of length 22.
        J_template: [22, 3] rest-pose joint positions.

    Returns:
        joints: [T, 22, 3] world-space joint positions.
    """
    T = transl.shape[0]
    # Extract 22 joints from 55-joint axis-angle
    aa_full = poses_aa.reshape(T, -1, 3)  # [T, 55, 3] or [T, 52, 3]
    n_joints_in = aa_full.shape[1]
    if n_joints_in >= 55:
        aa = aa_full[:, :22, :]  # [T, 22, 3]
    elif n_joints_in >= 22:
        aa = aa_full[:, :22, :]
    else:
        raise ValueError(f"Expected >= 22 joints in poses, got {n_joints_in}")

    # Convert to local rotation matrices [T, 22, 3, 3]
    local_rots = _rodrigues_batch(aa.reshape(-1, 3)).reshape(T, 22, 3, 3)

    # Build local offsets from rest-pose template
    # offset[j] = J_template[j] - J_template[parent[j]]  for j > 0
    # offset[0] = J_template[0] (root)
    offsets = np.zeros((22, 3), dtype=np.float32)
    offsets[0] = J_template[0]
    for j in range(1, 22):
        offsets[j] = J_template[j] - J_template[parents[j]]

    # Forward pass through kinematic tree
    # global_transform[j] = (R_global, t_global) where t_global is joint position
    global_positions = np.zeros((T, 22, 3), dtype=np.float32)
    global_rotations = np.zeros((T, 22, 3, 3), dtype=np.float32)

    # Root joint
    global_rotations[:, 0] = local_rots[:, 0]  # [T, 3, 3]
    global_positions[:, 0] = transl + offsets[0]  # [T, 3]

    for j in range(1, 22):
        p = parents[j]
        # Global rotation = parent_global_rot @ local_rot
        global_rotations[:, j] = np.matmul(
            global_rotations[:, p], local_rots[:, j]
        )
        # Global position = parent_pos + parent_global_rot @ offset
        offset_rotated = np.einsum("tij,j->ti", global_rotations[:, p], offsets[j])
        global_positions[:, j] = global_positions[:, p] + offset_rotated

    return global_positions


# ---------------------------------------------------------------------------
# Capsule collision detector
# ---------------------------------------------------------------------------
def _segment_segment_distance_batch(
    p1: np.ndarray,   # [N, 3] start of segment 1
    d1: np.ndarray,   # [N, 3] direction of segment 1 (end - start)
    p2: np.ndarray,   # [N, 3] start of segment 2
    d2: np.ndarray,   # [N, 3] direction of segment 2 (end - start)
) -> np.ndarray:      # [N] closest distances
    """Vectorized segment-segment closest-point distance computation.

    Computes the minimum distance between N pairs of line segments.
    Each segment i goes from p_i to p_i + d_i.
    """
    r = p1 - p2  # [N, 3]
    a = np.sum(d1 * d1, axis=-1)  # [N]
    e = np.sum(d2 * d2, axis=-1)  # [N]
    f = np.sum(d2 * r, axis=-1)   # [N]

    eps = 1e-8
    # Both segments degenerate to points
    b = np.sum(d1 * d2, axis=-1)  # [N]
    c = np.sum(d1 * r, axis=-1)   # [N]

    denom = a * e - b * b  # [N]
    denom_safe = np.where(np.abs(denom) < eps, eps, denom)

    # Compute unclamped parameters
    s = np.clip((b * f - c * e) / denom_safe, 0.0, 1.0)
    t = np.clip((b * s + f) / np.where(np.abs(e) < eps, eps, e), 0.0, 1.0)
    # Recompute s based on clamped t
    s = np.clip((t * b - c) / np.where(np.abs(a) < eps, eps, a), 0.0, 1.0)

    closest1 = p1 + s[:, None] * d1  # [N, 3]
    closest2 = p2 + t[:, None] * d2  # [N, 3]
    dist = np.linalg.norm(closest1 - closest2, axis=-1)  # [N]
    return dist


class CapsuleCollisionDetector:
    """Checks for inter-person collisions using capsule approximations.

    Each body part is approximated as a capsule (line segment + radius).
    """

    def __init__(
        self,
        capsule_pairs: Optional[List[Tuple[int, int]]] = None,
        capsule_radii: Optional[List[float]] = None,
        coarse_threshold: float = 3.0,
    ):
        self.capsule_pairs = capsule_pairs or CAPSULE_PAIRS
        self.capsule_radii = np.asarray(capsule_radii or CAPSULE_RADII, dtype=np.float32)
        self.coarse_threshold = coarse_threshold
        self.n_capsules = len(self.capsule_pairs)

    def check_collision(
        self,
        joints_a: np.ndarray,  # [T, 22, 3]
        joints_b: np.ndarray,  # [T, 22, 3]
    ) -> bool:
        """Return True if any frame has interpenetrating capsules between two persons."""
        T = min(joints_a.shape[0], joints_b.shape[0])
        joints_a = joints_a[:T]
        joints_b = joints_b[:T]

        # Coarse check: pelvis (joint 0) distance
        pelvis_dist = np.linalg.norm(joints_a[:, 0, :] - joints_b[:, 0, :], axis=-1)  # [T]
        close_mask = pelvis_dist < self.coarse_threshold
        close_frames = np.where(close_mask)[0]

        if len(close_frames) == 0:
            return False

        # Fine check on close frames only
        ja = joints_a[close_frames]  # [F, 22, 3]
        jb = joints_b[close_frames]  # [F, 22, 3]
        F = ja.shape[0]

        # Extract capsule endpoints for each person
        # caps_a_start/end: [F, n_capsules, 3]
        idx_a = np.array([p[0] for p in self.capsule_pairs])
        idx_b = np.array([p[1] for p in self.capsule_pairs])

        caps_a_start = ja[:, idx_a, :]  # [F, C, 3]
        caps_a_end = ja[:, idx_b, :]    # [F, C, 3]
        caps_b_start = jb[:, idx_a, :]  # [F, C, 3]
        caps_b_end = jb[:, idx_b, :]    # [F, C, 3]

        dir_a = caps_a_end - caps_a_start  # [F, C, 3]
        dir_b = caps_b_end - caps_b_start  # [F, C, 3]

        # All C_a x C_b capsule pairs across all F frames
        # Expand to [F, C_a, C_b, 3]
        C = self.n_capsules
        p1 = np.broadcast_to(caps_a_start[:, :, None, :], (F, C, C, 3)).reshape(-1, 3)
        d1 = np.broadcast_to(dir_a[:, :, None, :], (F, C, C, 3)).reshape(-1, 3)
        p2 = np.broadcast_to(caps_b_start[:, None, :, :], (F, C, C, 3)).reshape(-1, 3)
        d2 = np.broadcast_to(dir_b[:, None, :, :], (F, C, C, 3)).reshape(-1, 3)

        distances = _segment_segment_distance_batch(p1, d1, p2, d2)  # [F*C*C]
        distances = distances.reshape(F, C, C)

        # Combined radii for each pair
        radii_a = self.capsule_radii[None, :, None]  # [1, C, 1]
        radii_b = self.capsule_radii[None, None, :]  # [1, 1, C]
        combined = radii_a + radii_b  # [1, C, C]

        # Signed distance: negative means penetration
        signed_dist = distances - combined
        return bool(np.any(signed_dist < 0))


# ---------------------------------------------------------------------------
# ComposeMultiPerson transform
# ---------------------------------------------------------------------------
@TRANSFORMS.register_module(force=True)
class ComposeMultiPerson(BaseTransform):
    """Online data augmentation that composes multiple independent single-person
    motions into a multi-person scene with collision avoidance and composed captions.

    Pipeline position: after LoadSmplx55 + LoadCompatibleCaption, before RandomCropPadding.
    """

    def __init__(
        self,
        compose_prob: float = 0.5,
        max_persons: int = 2,
        min_persons: int = 2,
        max_retries: int = 10,
        max_total_frames: int = 1440,
        placement_radius_range: Tuple[float, float] = (1.0, 3.0),
        placement_stats_file: Optional[str] = None,
        yaw_range: float = 180.0,
        collision_check: bool = True,
        capsule_radii: Optional[List[float]] = None,
        smplx_model_path: str = "checkpoints/smpl_models/smplx",
        gender: str = "neutral",
        motion_key: str = "motion",
        caption_key: str = "caption",
        caption_template: str = "Person {idx}: {caption}",
        caption_separator: str = " ",
        skip_with_audio: bool = True,
    ):
        super().__init__()
        self.compose_prob = compose_prob
        self.max_persons = max_persons
        self.min_persons = min_persons
        self.max_retries = max_retries
        self.max_total_frames = max_total_frames
        self.placement_radius_range = placement_radius_range
        self.placement_stats_file = placement_stats_file
        self.yaw_range = yaw_range
        self.collision_check = collision_check
        self.skip_with_audio = skip_with_audio
        self.motion_key = motion_key
        self.caption_key = caption_key
        self.caption_template = caption_template
        self.caption_separator = caption_separator

        # Collision detector
        self._detector = CapsuleCollisionDetector(capsule_radii=capsule_radii)

        # FK template (lazy loaded)
        self._smplx_model_path = smplx_model_path
        self._gender = gender
        self._J_template: Optional[np.ndarray] = None

        # Dataset reference (set by dataset after construction)
        self._dataset = None
        self._single_person_indices: Optional[List[int]] = None
        self._placement_radii: Optional[List[float]] = None

    @property
    def J_template(self) -> np.ndarray:
        if self._J_template is None:
            self._J_template = _load_J_template(self._smplx_model_path, self._gender)
        return self._J_template

    def set_dataset(self, dataset) -> None:
        """Called by the dataset after pipeline construction to provide a reference
        for sampling additional single-person motions."""
        self._dataset = dataset
        self._build_single_person_indices()

    def _build_single_person_indices(self) -> None:
        """Build a list of dataset indices that are single-person entries with valid captions."""
        if self._dataset is None:
            self._single_person_indices = []
            return

        indices = []
        motion_key = getattr(self._dataset, "motion_key", "smplx")
        for i, data_info in enumerate(self._dataset.data_list):
            mp = data_info.get(f"{motion_key}_path")
            if isinstance(mp, str):
                # Check that a caption path exists
                caption_path = data_info.get(
                    "hierarchical_caption_path", data_info.get("caption_path")
                )
                if caption_path is not None:
                    indices.append(i)
        self._single_person_indices = indices
        logger.info(
            f"ComposeMultiPerson: built index of {len(indices)} single-person samples"
        )

    def _sample_additional_entry(self, exclude_motion_paths: Optional[set] = None) -> Optional[Dict]:
        """Sample a random single-person entry from the dataset, loading its raw npz and caption."""
        if not self._single_person_indices:
            return None
        exclude_motion_paths = exclude_motion_paths or set()

        idx = random.choice(self._single_person_indices)
        raw_data_info = self._dataset.data_list[idx]
        data_dir = self._dataset.data_dir
        motion_key = getattr(self._dataset, "motion_key", "smplx")

        # Load motion
        motion_path = raw_data_info[f"{motion_key}_path"]
        if isinstance(motion_path, list):
            return None
        if str(motion_path) in exclude_motion_paths:
            return None
        full_motion_path = os.path.join(data_dir, motion_path)
        if not os.path.exists(full_motion_path):
            return None

        try:
            abs_trans, poses, fps = _read_one_person_npz(full_motion_path)
        except Exception:
            return None

        # Load caption
        caption_path = raw_data_info.get(
            "hierarchical_caption_path", raw_data_info.get("caption_path")
        )
        caption = None
        if caption_path is not None:
            full_caption_path = os.path.join(data_dir, caption_path)
            if os.path.exists(full_caption_path):
                caption = self._load_caption(full_caption_path)

        return {
            "abs_trans": abs_trans,
            "poses": poses,
            "fps": fps,
            "caption": caption,
            "motion_path": str(motion_path),
        }

    def _load_caption(self, caption_path: str) -> Optional[str]:
        """Load a caption string from a hierarchical caption JSON file."""
        try:
            data = read_json(caption_path)
        except Exception:
            return None

        caption_list = []
        # Hierarchical format
        if all(k in data and isinstance(data[k], list) for k in ["macro", "meso", "micro"]):
            for granularity in ["macro", "meso"]:
                for cap in data[granularity]:
                    if isinstance(cap, str) and len(cap.strip()) > 0:
                        caption_list.append(cap.strip())
        # HYMotion format
        elif "result" in data and isinstance(data["result"], list):
            for item in data["result"]:
                if not isinstance(item, dict):
                    continue
                if "short_caption_rewritten" in item and isinstance(
                    item["short_caption_rewritten"], list
                ):
                    for rc in item["short_caption_rewritten"]:
                        if isinstance(rc, str) and len(rc.strip()) > 0:
                            caption_list.append(rc.strip())
                elif "short_caption" in item and isinstance(item["short_caption"], str):
                    s = item["short_caption"].strip()
                    if len(s) > 0:
                        caption_list.append(s)

        if not caption_list:
            return None
        return random.choice(caption_list)

    def _load_placement_radii(self) -> List[float]:
        if self._placement_radii is not None:
            return self._placement_radii
        if not self.placement_stats_file:
            self._placement_radii = []
            return self._placement_radii

        payload = read_json(self.placement_stats_file)
        radii = payload.get("radii", [])
        self._placement_radii = [
            float(radius)
            for radius in radii
            if isinstance(radius, (int, float)) and float(radius) > 0.0
        ]
        if not self._placement_radii:
            raise ValueError(
                f"ComposeMultiPerson: no positive radii in {self.placement_stats_file}"
            )
        return self._placement_radii

    def _sample_placement_radius(self) -> float:
        radii = self._load_placement_radii()
        if radii:
            return float(random.choice(radii))
        r_min, r_max = self.placement_radius_range
        return float(np.random.uniform(r_min, r_max))

    def _apply_independent_augmentation(
        self,
        abs_trans: np.ndarray,  # [T, 3]
        poses: np.ndarray,      # [T, 165]
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Apply random yaw rotation + polar XZ offset to a single person.

        Returns:
            augmented_trans: [T, 3]
            augmented_poses: [T, 165]
            yaw_deg: applied yaw in degrees
            offset: [3] applied offset
        """
        yaw_deg = float(np.random.uniform(-self.yaw_range, self.yaw_range))
        R_y = _build_Ry_from_deg(yaw_deg)

        # Polar placement offset in XZ plane
        radius = self._sample_placement_radius()
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        offset = np.array(
            [radius * np.cos(angle), 0.0, radius * np.sin(angle)],
            dtype=np.float32,
        )

        # Apply yaw to translation and root orient
        aug_trans = (abs_trans @ R_y.T) + offset[None, :]
        aug_poses = apply_root_yaw_to_axis_angle(poses, R_y)

        return aug_trans, aug_poses, yaw_deg, offset

    def _compute_joints(
        self,
        abs_trans: np.ndarray,
        poses: np.ndarray,
    ) -> np.ndarray:
        """Compute FK joints [T, 22, 3] from raw motion data."""
        return numpy_fk_22joints(abs_trans, poses, SMPL_22_PARENTS, self.J_template)

    def _process_motion_vector(
        self,
        abs_trans: np.ndarray,
        poses: np.ndarray,
        rot_type: str,
        transl_type: str,
        smpl_type: str,
        rot6d_convention: str = "row",
    ) -> torch.Tensor:
        """Convert raw (trans, poses) into the same motion vector format as LoadSmplx55."""
        transl = process_transl(abs_trans, transl_type)
        pose = process_smplx_pose(
            poses,
            rot_type,
            smpl_type,
            rot6d_convention=rot6d_convention,
        )
        out = np.concatenate([transl, pose], axis=-1)
        return torch.from_numpy(out).float()

    @staticmethod
    def _task_has_audio(task_cls) -> bool:
        """Check whether a task class involves any audio/music modality.

        Works by inspecting all_modality() and checking if any modality is
        Audio or a subclass (Music, PastMusic, FutureMusic, …).  This is
        future-proof: new audio-related tasks are automatically excluded
        without maintaining a hardcoded abbreviation list.
        """
        try:
            for modal in task_cls.all_modality():
                if isinstance(modal, type) and issubclass(modal, Audio):
                    return True
                # Modalities may also be instances rather than classes
                if isinstance(modal, Audio):
                    return True
            return False
        except Exception:
            return False

    def transform(self, results: Dict) -> Dict:
        # Guard: already multi-person — pass through
        num_person = results.get("num_person", 1)
        if num_person > 1:
            return results

        # Probabilistic gate: skip composition with (1 - compose_prob) chance
        if np.random.rand() >= self.compose_prob:
            return results

        # Skip audio-related tasks: composing multi-person over a single audio
        # track destroys beat alignment. Instead of maintaining an explicit list
        # of task abbreviations, inspect the task's modalities — any task that
        # involves Audio or its subclasses (Music, PastMusic, FutureMusic, …)
        # is automatically excluded.
        if self.skip_with_audio:
            task_cls = results.get("task")
            if task_cls is not None and self._task_has_audio(task_cls):
                return results

        # --- Below this point, composition will be attempted. Failures raise so
        #     that the dataset's refetch logic can retry with another sample. ---

        if self._dataset is None or not self._single_person_indices:
            raise ValueError(
                "ComposeMultiPerson: dataset reference not set or no single-person indices available"
            )

        # Get current sample's metadata
        motion = results.get(self.motion_key)
        if motion is None or not isinstance(motion, torch.Tensor):
            raise ValueError(
                f"ComposeMultiPerson: missing or invalid motion (got {type(motion).__name__})"
            )

        # We need the raw motion path to re-read the npz for FK
        motion_path = results.get("motion_path")
        if motion_path is None or not isinstance(motion_path, (str, np.str_)):
            raise ValueError(
                f"ComposeMultiPerson: missing or invalid motion_path (got {type(motion_path).__name__})"
            )

        rot_type = results.get("rot_type", "rotation_6d")
        transl_type = results.get("transl_type", "abs")
        smpl_type = results.get("smpl_type", "smpl_22")
        rot6d_convention = results.get("rot6d_convention", "row")
        caption_0 = results.get(self.caption_key)

        # Re-read person 0's raw npz to get FK joints with current augmentation
        abs_trans_0, poses_0, fps_0 = _read_one_person_npz(str(motion_path))

        # Reconstruct augmented person 0 using stored augmentation params
        aug_yaw_deg = results.get("aug_yaw_deg", 0.0)
        aug_offset = np.array(results.get("aug_offset", [0.0, 0.0, 0.0]), dtype=np.float32)

        if aug_yaw_deg != 0.0 or np.any(aug_offset != 0.0):
            R_y_0 = _build_Ry_from_deg(float(aug_yaw_deg))
            abs_trans_0_aug = (abs_trans_0 @ R_y_0.T) + aug_offset[None, :]
            poses_0_aug = apply_root_yaw_to_axis_angle(poses_0, R_y_0)
        else:
            abs_trans_0_aug = abs_trans_0
            poses_0_aug = poses_0

        joints_0 = self._compute_joints(abs_trans_0_aug, poses_0_aug)

        # Determine number of persons to compose
        num_target = random.randint(self.min_persons, self.max_persons)
        if num_target <= 1:
            raise ValueError(
                f"ComposeMultiPerson: num_target={num_target} <= 1 with "
                f"min_persons={self.min_persons}, max_persons={self.max_persons}"
            )

        # Collect person motions and captions
        person_motions = [motion]  # person 0's already-processed motion vector
        person_joints = [joints_0]
        person_captions = [caption_0]
        used_motion_paths = {str(motion_path)}
        T_0 = motion.shape[0]

        for _ in range(num_target - 1):
            placed = False
            for _retry in range(self.max_retries):
                entry = self._sample_additional_entry(used_motion_paths)
                if entry is None:
                    continue

                abs_trans_k = entry["abs_trans"]
                poses_k = entry["poses"]

                # Apply independent augmentation
                aug_trans_k, aug_poses_k, _, _ = self._apply_independent_augmentation(
                    abs_trans_k, poses_k
                )

                # Compute FK joints for collision check
                joints_k = self._compute_joints(aug_trans_k, aug_poses_k)

                # Check collision against all existing persons
                if self.collision_check:
                    has_collision = False
                    for existing_joints in person_joints:
                        if self._detector.check_collision(existing_joints, joints_k):
                            has_collision = True
                            break
                    if has_collision:
                        continue

                # Convert to motion vector with same representation
                motion_vec_k = self._process_motion_vector(
                    aug_trans_k,
                    aug_poses_k,
                    rot_type,
                    transl_type,
                    smpl_type,
                    rot6d_convention=rot6d_convention,
                )

                # OOM guard: check total frame budget before accepting.
                # Total frames ≈ max(all T_i) * num_persons (after padding).
                # We approximate with sum of individual T_i which is an upper bound.
                candidate_frames = [m.shape[0] for m in person_motions] + [motion_vec_k.shape[0]]
                total_frames = max(candidate_frames) * len(candidate_frames)
                if total_frames > self.max_total_frames:
                    # Budget exceeded — stop adding more persons
                    break

                person_motions.append(motion_vec_k)
                person_joints.append(joints_k)
                person_captions.append(entry.get("caption"))
                used_motion_paths.add(entry.get("motion_path"))
                placed = True
                break

            if not placed:
                # Could not place this person; stop trying to add more
                break

        if len(person_motions) <= 1:
            raise ValueError(
                f"ComposeMultiPerson: failed to place any additional person after "
                f"{self.max_retries} retries (collision_check={self.collision_check})"
            )

        # Align frame counts: pad shorter sequences to the longest with last-frame replication.
        # Truncating would risk cutting off key actions described by the caption.
        per_person_num_frames = [int(m.shape[0]) for m in person_motions]
        T_max = max(per_person_num_frames)
        aligned = []
        for m in person_motions:
            if m.shape[0] < T_max:
                pad = m[-1:].expand(T_max - m.shape[0], -1)
                m = torch.cat([m, pad], dim=0)
            aligned.append(m)

        # Stack: [P, T, D]
        stacked = torch.stack(aligned, dim=0)
        if torch.any(torch.isnan(stacked)):
            raise ValueError("ComposeMultiPerson: NaN in composed multi-person motion")

        # Keep per-person captions structured. VermoProcessor serializes this
        # list into explicit "Person N caption" text tokens at training time.
        structured_captions = [
            cap.strip()
            for cap in person_captions
            if cap is not None and isinstance(cap, str) and len(cap.strip()) > 0
        ]

        # Update results
        results[self.motion_key] = stacked
        results["num_person"] = len(person_motions)
        results["sample_domain"] = "pseudo_multi"
        results["num_frames"] = T_max
        # Store per-person original frame counts so downstream (VermoProcessor)
        # can trim padding tokens per person and avoid training on replicated frames.
        results["per_person_num_frames"] = np.array(per_person_num_frames, dtype=np.int64)
        if structured_captions:
            results[self.caption_key] = structured_captions
            results["person_captions"] = structured_captions
        else:
            results[self.caption_key] = results.get(self.caption_key)

        return results
