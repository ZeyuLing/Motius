#!/usr/bin/env python3
"""Convert HyMotion motion_135 to PyRoki keypoints format for retargeting.

Bridge between HyMotion T2M output and ProtoMotions PyRoki retargeting pipeline.

Input:  motion_135 NPZ (from HyMotion eval)
Output: .npy dict with 18 keypoints (15 base + 3 auxiliary) for PyRoki

Pipeline:
    motion_135 (T, 135)
      -> decode rot6d (reorder row->col, Gram-Schmidt)
      -> SMPL FK (smplx package) -> world-space positions (T, 22, 3)
      -> compute world rotations via kinematic chain -> (T, 22, 3, 3)
      -> Y-up -> Z-up coordinate transform
      -> extract 15 base keypoints (SMPL joint subset)
      -> geometric surgery (pelvis/elbow/ankle/foot offsets per keypoint_utils.py)
      -> add 3 auxiliary keypoints (hand_aux x2, pelvis_aux)
      -> detect foot contacts (velocity + height threshold)
      -> save .npy dict {positions, orientations, left_foot_contacts, right_foot_contacts}

Output format (consumed by batch_retarget_to_g1_from_keypoints.py):
    positions:           (T, 18, 3)   - keypoint world positions (Z-up)
    orientations:        (T, 18, 3, 3) - keypoint world rotation matrices
    left_foot_contacts:  (T, 2)       - [ankle, toebase] binary contact (0/1)
    right_foot_contacts: (T, 2)       - [ankle, toebase] binary contact (0/1)
"""
import argparse
import numpy as np
import torch
from pathlib import Path

# ============================================================================
# SMPL constants
# ============================================================================

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

# SMPL kinematic tree: parent[i] = parent joint index for joint i (-1 = root)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# 15 base keypoints: indices into the SMPL 22-joint array.
# Order matches KEYPOINT_MAPPING_SMPL in ProtoMotions/data/scripts/keypoint_utils.py
KEYPOINT_SMPL_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]

KEYPOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

# Y-up -> Z-up rotation matrix: rotate +90 deg around X axis
# [x, y, z]_Yup -> [x, -z, y]_Zup
RX_Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


# ============================================================================
# Rotation utilities (adapted from motion135_to_smplx.py)
# ============================================================================

def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix.

    HyMotion outputs rot6d in row-major layout: [R00,R01, R10,R11, R20,R21]
    Gram-Schmidt expects column-major layout: [R00,R10,R20, R01,R11,R21]
    We reorder [0,2,4,1,3,5] to convert row-major -> column-major before decoding.

    Args:
        rot6d: (..., 6) array of 6D rotation representations (row-major)
    Returns:
        rotmat: (..., 3, 3) array of rotation matrices
    """
    # Row-major -> column-major reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    # Second column: Gram-Schmidt orthogonalization
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

    # Third column: cross product
    b3 = np.cross(b1, b2)

    rotmat = np.stack([b1, b2, b3], axis=-1)
    return rotmat


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    from scipy.spatial.transform import Rotation as R

    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    rot = R.from_matrix(rotmat_flat)
    aa_flat = rot.as_rotvec()
    return aa_flat.reshape(*orig_shape, 3)


# ============================================================================
# SMPL forward kinematics
# ============================================================================

def smpl_forward_kinematics(root_orient, pose_body, transl, smpl_model_path):
    """Run SMPL-X FK to get world-space joint positions.

    Args:
        root_orient: (T, 3) root orientation in axis-angle
        pose_body:   (T, 63) body pose in axis-angle (21 joints x 3)
        transl:      (T, 3) translation
        smpl_model_path: path to directory containing smplx/ model files

    Returns:
        positions: (T, 22, 3) world-space body joint positions (Y-up)
    """
    import smplx

    body_model = smplx.create(
        model_path=smpl_model_path,
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        use_pca=False,
    )
    body_model.eval()

    T = root_orient.shape[0]
    with torch.no_grad():
        output = body_model(
            global_orient=torch.tensor(root_orient, dtype=torch.float32),
            body_pose=torch.tensor(pose_body, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
            betas=torch.zeros(T, 10, dtype=torch.float32),
            jaw_pose=torch.zeros(T, 3, dtype=torch.float32),
            leye_pose=torch.zeros(T, 3, dtype=torch.float32),
            reye_pose=torch.zeros(T, 3, dtype=torch.float32),
            left_hand_pose=torch.zeros(T, 45, dtype=torch.float32),
            right_hand_pose=torch.zeros(T, 45, dtype=torch.float32),
            expression=torch.zeros(T, 10, dtype=torch.float32),
        )
    # SMPLX returns body joints as the first 22 entries
    positions = output.joints[:, :22, :].cpu().numpy().astype(np.float64)
    print(f"  SMPL FK: output.joints shape = {output.joints.shape}, using first 22")
    return positions


def compute_world_rotations(local_rotmat: np.ndarray) -> np.ndarray:
    """Compute world-space rotation matrices from local rotations via kinematic chain.

    Args:
        local_rotmat: (T, 22, 3, 3) local rotation matrices per joint

    Returns:
        world_rotmat: (T, 22, 3, 3) world-space rotation matrices
    """
    T, num_joints = local_rotmat.shape[:2]
    world_rotmat = np.zeros_like(local_rotmat)

    for j in range(num_joints):
        p = SMPL_PARENTS[j]
        if p == -1:
            # Root joint: local = world
            world_rotmat[:, j] = local_rotmat[:, j]
        else:
            # R_world[j] = R_world[parent] @ R_local[j]
            world_rotmat[:, j] = np.matmul(world_rotmat[:, p], local_rotmat[:, j])

    return world_rotmat


# ============================================================================
# Coordinate transforms
# ============================================================================

def transform_y_up_to_z_up(positions, rotations):
    """Transform from SMPL Y-up to MuJoCo Z-up coordinate system.

    Applies rotation Rx(+90 deg around X): [x,y,z] -> [x, -z, y]

    Args:
        positions:  (T, N, 3) joint positions
        rotations:  (T, N, 3, 3) rotation matrices

    Returns:
        positions_zup:  (T, N, 3)
        rotations_zup:  (T, N, 3, 3)
    """
    # Transform positions: p_new = Rx @ p
    positions_zup = np.einsum('ij,tkj->tki', RX_Y2Z, positions)

    # Transform rotations: R_new = Rx @ R @ Rx^T
    rotations_zup = RX_Y2Z[None, None] @ rotations @ RX_Y2Z.T[None, None]

    return positions_zup, rotations_zup


# ============================================================================
# Geometric surgery (matching keypoint_utils.py SMPL skeleton)
# ============================================================================

def apply_geometric_surgery(kp_positions, kp_orientations):
    """Apply geometric surgery to keypoints per keypoint_utils.py SMPL path.

    Matches extract_keypoints_from_motion_smpl_skel() in ProtoMotions exactly.

    Offsets applied in each keypoint's local frame:
        - pelvis:  [-0.04, 0, 0]
        - elbows:  [0, 0, 0.045]
        - ankles:  [0.03, 0, 0]  (applied after foot calculation)
        - feet:    [0.18, 0, 0] from ankle (flat feet approximation)

    Args:
        kp_positions:    (T, 15, 3) keypoint positions (Z-up, after extraction)
        kp_orientations: (T, 15, 3, 3) keypoint rotation matrices

    Returns:
        kp_positions, kp_orientations (modified in-place and returned)
    """
    # --- Pelvis offset: [-0.04, 0, 0] in local frame ---
    root_idx = KEYPOINT_NAMES.index("pelvis")
    root_offset = np.array([-0.04, 0.0, 0.0])
    R_root = kp_orientations[:, root_idx]  # (T, 3, 3)
    kp_positions[:, root_idx] += np.einsum('tij,j->ti', R_root, root_offset)

    # --- Elbow offset: [0, 0, 0.045] in local frame ---
    elbow_offset = np.array([0.0, 0.0, 0.045])
    for name in ["left_elbow", "right_elbow"]:
        idx = KEYPOINT_NAMES.index(name)
        R = kp_orientations[:, idx]
        kp_positions[:, idx] += np.einsum('tij,j->ti', R, elbow_offset)

    # --- Flat feet + ankle offset ---
    # IMPORTANT: foot position is calculated from PRE-surgery ankle position,
    # then ankle gets its own offset. This matches keypoint_utils.py exactly.
    foot_offset = np.array([0.15 + 0.03, 0.0, 0.0])   # [0.18, 0, 0]
    ankle_offset = np.array([0.03, 0.0, 0.0])

    for side in ["left", "right"]:
        ankle_idx = KEYPOINT_NAMES.index(f"{side}_ankle")
        foot_idx = KEYPOINT_NAMES.index(f"{side}_foot")

        # Save pre-surgery ankle position
        p_ankle_orig = kp_positions[:, ankle_idx].copy()
        R_ankle = kp_orientations[:, ankle_idx]

        # Flat foot: toe = ankle_orig + R_ankle @ [0.18, 0, 0]
        kp_positions[:, foot_idx] = p_ankle_orig + np.einsum(
            'tij,j->ti', R_ankle, foot_offset
        )

        # Ankle offset: ankle = ankle_orig + R_ankle @ [0.03, 0, 0]
        kp_positions[:, ankle_idx] = p_ankle_orig + np.einsum(
            'tij,j->ti', R_ankle, ankle_offset
        )

    return kp_positions, kp_orientations


def add_auxiliary_keypoints(kp_positions, kp_orientations):
    """Add 3 auxiliary keypoints: left_hand_aux, right_hand_aux, pelvis_aux.

    Matches keypoint_utils.py SMPL skeleton auxiliary point generation.

    Order appended: [left_hand_aux, right_hand_aux, pelvis_aux] -> total 18

    Args:
        kp_positions:    (T, 15, 3)
        kp_orientations: (T, 15, 3, 3)

    Returns:
        positions:    (T, 18, 3)
        orientations: (T, 18, 3, 3)
    """
    # --- Hand auxiliary: wrist + R_wrist @ [0.2, 0, 0] ---
    hand_offset = np.array([0.2, 0.0, 0.0])

    left_wrist_idx = KEYPOINT_NAMES.index("left_wrist")
    right_wrist_idx = KEYPOINT_NAMES.index("right_wrist")

    p_lw = kp_positions[:, left_wrist_idx]
    R_lw = kp_orientations[:, left_wrist_idx]
    p_left_hand_aux = p_lw + np.einsum('tij,j->ti', R_lw, hand_offset)

    p_rw = kp_positions[:, right_wrist_idx]
    R_rw = kp_orientations[:, right_wrist_idx]
    p_right_hand_aux = p_rw + np.einsum('tij,j->ti', R_rw, hand_offset)

    # --- Pelvis auxiliary: pelvis + R_pelvis @ [0.16, 0, 0] ---
    # 0.16 = 0.2 - 0.04 (accounting for the pelvis surgery offset already applied)
    pelvis_idx = KEYPOINT_NAMES.index("pelvis")
    pelvis_offset = np.array([0.2 - 0.04, 0.0, 0.0])  # [0.16, 0, 0]

    p_pelvis = kp_positions[:, pelvis_idx]
    R_pelvis = kp_orientations[:, pelvis_idx]
    p_pelvis_aux = p_pelvis + np.einsum('tij,j->ti', R_pelvis, pelvis_offset)

    # --- Concatenate: [15 base, left_hand_aux, right_hand_aux, pelvis_aux] = 18 ---
    positions = np.concatenate([
        kp_positions,
        p_left_hand_aux[:, None, :],
        p_right_hand_aux[:, None, :],
        p_pelvis_aux[:, None, :],
    ], axis=1)  # (T, 18, 3)

    orientations = np.concatenate([
        kp_orientations,
        R_lw[:, None, :, :],       # left_hand_aux uses wrist orientation
        R_rw[:, None, :, :],       # right_hand_aux uses wrist orientation
        R_pelvis[:, None, :, :],   # pelvis_aux uses pelvis orientation
    ], axis=1)  # (T, 18, 3, 3)

    return positions, orientations


# ============================================================================
# Foot contact detection
# ============================================================================

def detect_foot_contacts(kp_positions, fps=30):
    """Detect foot contacts using velocity and height thresholds.

    Uses adaptive ground level detection based on minimum foot height.

    Args:
        kp_positions: (T, 15, 3) keypoint positions (Z-up, after geometric surgery)
        fps: frames per second

    Returns:
        left_foot_contacts:  (T, 2) - [ankle, toebase] binary int
        right_foot_contacts: (T, 2) - [ankle, toebase] binary int
    """
    dt = 1.0 / fps
    T = kp_positions.shape[0]

    left_ankle_idx = KEYPOINT_NAMES.index("left_ankle")
    right_ankle_idx = KEYPOINT_NAMES.index("right_ankle")
    left_foot_idx = KEYPOINT_NAMES.index("left_foot")
    right_foot_idx = KEYPOINT_NAMES.index("right_foot")

    # Auto-detect ground level from minimum foot heights (2nd percentile for robustness)
    all_foot_z = np.concatenate([
        kp_positions[:, left_ankle_idx, 2],
        kp_positions[:, right_ankle_idx, 2],
        kp_positions[:, left_foot_idx, 2],
        kp_positions[:, right_foot_idx, 2],
    ])
    ground_z = np.percentile(all_foot_z, 2)

    vel_threshold = 0.5       # m/s - below this is considered "still"
    ankle_height_abs = ground_z + 0.10   # ankle contact within 10cm of ground
    foot_height_abs = ground_z + 0.06    # toe contact within 6cm of ground

    print(f"  Foot contact detection: ground_z={ground_z:.3f}m, "
          f"ankle_thresh={ankle_height_abs:.3f}m, foot_thresh={foot_height_abs:.3f}m")

    results = {}
    for side, ankle_idx, foot_idx in [
        ("left", left_ankle_idx, left_foot_idx),
        ("right", right_ankle_idx, right_foot_idx),
    ]:
        ankle_pos = kp_positions[:, ankle_idx]  # (T, 3)
        foot_pos = kp_positions[:, foot_idx]    # (T, 3)

        # Compute velocities (m/s)
        ankle_vel = np.zeros_like(ankle_pos)
        ankle_vel[1:] = (ankle_pos[1:] - ankle_pos[:-1]) / dt
        ankle_speed = np.linalg.norm(ankle_vel, axis=-1)

        foot_vel = np.zeros_like(foot_pos)
        foot_vel[1:] = (foot_pos[1:] - foot_pos[:-1]) / dt
        foot_speed = np.linalg.norm(foot_vel, axis=-1)

        # Contact = low velocity AND near ground
        ankle_contact = (
            (ankle_speed < vel_threshold) & (ankle_pos[:, 2] < ankle_height_abs)
        ).astype(np.int32)
        foot_contact = (
            (foot_speed < vel_threshold) & (foot_pos[:, 2] < foot_height_abs)
        ).astype(np.int32)

        # First frame: copy from second (velocity at t=0 is undefined)
        if T > 1:
            ankle_contact[0] = ankle_contact[1]
            foot_contact[0] = foot_contact[1]

        results[f"{side}_foot_contacts"] = np.stack(
            [ankle_contact, foot_contact], axis=-1
        )  # (T, 2)

    return results["left_foot_contacts"], results["right_foot_contacts"]


# ============================================================================
# Main conversion
# ============================================================================

def convert_motion135_to_pyroki_keypoints(
    input_npz: str,
    output_npy: str,
    smpl_model_path: str,
    fps: int = 30,
):
    """Convert motion_135 NPZ to PyRoki keypoints .npy.

    Args:
        input_npz: path to motion_135 NPZ file
        output_npy: path to save keypoints .npy file
        smpl_model_path: path to directory containing smplx/ model files
        fps: motion frame rate
    """
    # ---- 1. Load motion_135 ----
    data = np.load(input_npz, allow_pickle=True)
    motion = data['motion_135']  # (T, 135)
    T = motion.shape[0]
    print(f"\n[1/9] Load motion_135: frames={T}, shape={motion.shape}")

    # ---- 2. Decode rot6d -> rotation matrices ----
    transl = motion[:, :3].astype(np.float64)                # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6).astype(np.float64)
    local_rotmat = rot6d_to_rotmat(rot6d)                    # (T, 22, 3, 3)
    print(f"[2/9] Decode rot6d: local_rotmat {local_rotmat.shape}")

    # Verify rotation matrices are valid (det ≈ 1)
    dets = np.linalg.det(local_rotmat.reshape(-1, 3, 3))
    print(f"  Rotation matrix determinants: mean={dets.mean():.6f}, "
          f"min={dets.min():.6f}, max={dets.max():.6f}")

    # ---- 3. Convert to axis-angle for SMPL FK ----
    aa = rotmat_to_axis_angle(local_rotmat)        # (T, 22, 3)
    root_orient = aa[:, 0, :]                      # (T, 3)
    pose_body = aa[:, 1:22, :].reshape(T, -1)      # (T, 63)
    print(f"[3/9] Axis-angle: root_orient {root_orient.shape}, pose_body {pose_body.shape}")

    # ---- 4. SMPL FK -> world-space joint positions (Y-up) ----
    print(f"[4/9] Running SMPL FK...")
    positions_yup = smpl_forward_kinematics(
        root_orient, pose_body, transl, smpl_model_path
    )  # (T, 22, 3)
    print(f"  Positions Y-up: {positions_yup.shape}, "
          f"root Y mean={positions_yup[:, 0, 1].mean():.3f}m")

    # ---- 5. World-space rotations via kinematic chain ----
    world_rotmat = compute_world_rotations(local_rotmat)  # (T, 22, 3, 3)
    print(f"[5/9] World rotations: {world_rotmat.shape}")

    # ---- 6. Y-up -> Z-up coordinate transform ----
    positions_zup, world_rotmat_zup = transform_y_up_to_z_up(
        positions_yup, world_rotmat
    )
    print(f"[6/9] Z-up transform: root Z mean={positions_zup[:, 0, 2].mean():.3f}m")

    # ---- 7. Extract 15 keypoints ----
    kp_positions = positions_zup[:, KEYPOINT_SMPL_INDICES, :].copy()       # (T, 15, 3)
    kp_orientations = world_rotmat_zup[:, KEYPOINT_SMPL_INDICES, :, :].copy()  # (T, 15, 3, 3)
    print(f"[7/9] Extract keypoints: {kp_positions.shape}")

    # ---- 8. Geometric surgery ----
    kp_positions, kp_orientations = apply_geometric_surgery(
        kp_positions, kp_orientations
    )
    print(f"[8/9] Geometric surgery applied")

    # ---- Detect foot contacts (before adding aux) ----
    left_contacts, right_contacts = detect_foot_contacts(kp_positions, fps)
    left_ratio = left_contacts.sum() / max(left_contacts.size, 1) * 100
    right_ratio = right_contacts.sum() / max(right_contacts.size, 1) * 100
    print(f"  Foot contacts: left={left_ratio:.1f}%, right={right_ratio:.1f}%")

    # ---- 9. Add auxiliary keypoints (15 -> 18) ----
    kp_positions, kp_orientations = add_auxiliary_keypoints(
        kp_positions, kp_orientations
    )
    print(f"[9/9] Add auxiliary keypoints: {kp_positions.shape}")

    # ---- Save output ----
    result = {
        'positions': kp_positions.astype(np.float64),             # (T, 18, 3)
        'orientations': kp_orientations.astype(np.float64),       # (T, 18, 3, 3)
        'left_foot_contacts': left_contacts.astype(np.int32),     # (T, 2)
        'right_foot_contacts': right_contacts.astype(np.int32),   # (T, 2)
    }
    np.save(output_npy, result)

    print(f"\nSaved: {output_npy}")
    print(f"  positions:           {result['positions'].shape}")
    print(f"  orientations:        {result['orientations'].shape}")
    print(f"  left_foot_contacts:  {result['left_foot_contacts'].shape}")
    print(f"  right_foot_contacts: {result['right_foot_contacts'].shape}")

    # ---- Sanity checks ----
    print(f"\nSanity checks:")
    print(f"  Position range: [{kp_positions.min():.3f}, {kp_positions.max():.3f}]")
    print(f"  Root height (Z, mean):  {kp_positions[:, 0, 2].mean():.3f}m")
    ankle_l_z = kp_positions[:, KEYPOINT_NAMES.index('left_ankle'), 2]
    ankle_r_z = kp_positions[:, KEYPOINT_NAMES.index('right_ankle'), 2]
    print(f"  L ankle Z (mean/min):   {ankle_l_z.mean():.3f}m / {ankle_l_z.min():.3f}m")
    print(f"  R ankle Z (mean/min):   {ankle_r_z.mean():.3f}m / {ankle_r_z.min():.3f}m")
    foot_l_z = kp_positions[:, KEYPOINT_NAMES.index('left_foot'), 2]
    foot_r_z = kp_positions[:, KEYPOINT_NAMES.index('right_foot'), 2]
    print(f"  L foot Z  (mean/min):   {foot_l_z.mean():.3f}m / {foot_l_z.min():.3f}m")
    print(f"  R foot Z  (mean/min):   {foot_r_z.mean():.3f}m / {foot_r_z.min():.3f}m")

    # Check rotation matrices are valid SO(3)
    R_flat = kp_orientations.reshape(-1, 3, 3)
    dets = np.linalg.det(R_flat)
    RtR = np.matmul(R_flat.transpose(0, 1, 2), R_flat.transpose(0, 2, 1))
    eye_err = np.abs(RtR - np.eye(3)[None]).max()
    print(f"  Rotation det: mean={dets.mean():.6f}, range=[{dets.min():.6f}, {dets.max():.6f}]")
    print(f"  R^T @ R - I max error: {eye_err:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert motion_135 to PyRoki keypoints for retargeting"
    )
    parser.add_argument("input", type=str, help="Path to motion_135 NPZ")
    parser.add_argument("output", type=str, help="Path to save keypoints .npy")
    parser.add_argument(
        "--smpl-model-path", type=str,
        default="checkpoints/smpl_models",
        help="Path to SMPL model directory (default: checkpoints/smpl_models)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Motion frame rate (default: 30)",
    )
    args = parser.parse_args()

    convert_motion135_to_pyroki_keypoints(
        args.input, args.output, args.smpl_model_path, args.fps
    )
