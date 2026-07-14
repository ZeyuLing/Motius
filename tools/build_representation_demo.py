#!/usr/bin/env python3
"""Build synchronized HumanML3D, SMPL mesh, and Unitree G1 mesh viewer data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.representation.rotation import matrix_to_axis_angle, rotation_6d_to_matrix
from motius.motion.retarget import GMRSMPLToG1Retargeter, GMR_Y_UP_FROM_Z_UP
from motius.motion.retarget.ardy_core import ARDY_CORE27_INDEX
from motius.motion.retarget.smpl_soma import (
    smpl_motion135_to_soma30,
)
from motius.motion.skeleton import SMPL22_NAMES, SMPL22_PARENTS
from motius.motion.skeleton.body_models import resolve_smpl_model_path


G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)

CORE27_PARENTS = np.asarray(
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        4,
        7,
        8,
        9,
        10,
        10,
        4,
        13,
        14,
        15,
        16,
        16,
        0,
        19,
        20,
        21,
        0,
        23,
        24,
        25,
    ],
    dtype=np.int64,
)


def _global_to_local_mats(global_rots: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.empty_like(global_rots, dtype=np.float32)
    for joint, parent in enumerate(np.asarray(parents).tolist()):
        if parent < 0:
            local[:, joint] = global_rots[:, joint]
        else:
            local[:, joint] = np.einsum(
                "tki,tkl->til",
                global_rots[:, parent],
                global_rots[:, joint],
            )
    return local


def _smpl_local_to_global(local_rots: np.ndarray) -> np.ndarray:
    global_rots = np.empty_like(local_rots, dtype=np.float32)
    for joint, parent in enumerate(np.asarray(SMPL22_PARENTS).tolist()):
        if parent < 0:
            global_rots[:, joint] = local_rots[:, joint]
        else:
            global_rots[:, joint] = global_rots[:, parent] @ local_rots[:, joint]
    return global_rots


def _motion135_local_rot_mats(motion: np.ndarray) -> np.ndarray:
    frames = len(motion)
    mats = rotation_6d_to_matrix(
        torch.from_numpy(motion[:, 3:135].reshape(-1, 6)),
        convention="row",
    ).reshape(frames, 22, 3, 3)
    return mats.cpu().numpy().astype(np.float32)


def _smpl_global_to_core27_global(smpl_global: np.ndarray) -> np.ndarray:
    smpl_index = {name: idx for idx, name in enumerate(SMPL22_NAMES)}
    core_to_smpl = {
        "Hips": "Pelvis",
        "Spine": "Spine1",
        "Spine1": "Spine1",
        "Spine2": "Spine2",
        "Spine3": "Spine3",
        "Neck": "Neck",
        "Head": "Head",
        "RightShoulder": "R_Collar",
        "RightArm": "R_Shoulder",
        "RightForeArm": "R_Elbow",
        "RightHand": "R_Wrist",
        "RightHandEnd": "R_Wrist",
        "RightHandThumb1": "R_Wrist",
        "LeftShoulder": "L_Collar",
        "LeftArm": "L_Shoulder",
        "LeftForeArm": "L_Elbow",
        "LeftHand": "L_Wrist",
        "LeftHandEnd": "L_Wrist",
        "LeftHandThumb1": "L_Wrist",
        "RightUpLeg": "R_Hip",
        "RightLeg": "R_Knee",
        "RightFoot": "R_Ankle",
        "RightToeBase": "R_Foot",
        "LeftUpLeg": "L_Hip",
        "LeftLeg": "L_Knee",
        "LeftFoot": "L_Ankle",
        "LeftToeBase": "L_Foot",
    }
    core_global = np.empty(smpl_global.shape[:1] + (27, 3, 3), dtype=np.float32)
    for core_name, smpl_name in core_to_smpl.items():
        core_global[:, ARDY_CORE27_INDEX[core_name]] = smpl_global[:, smpl_index[smpl_name]]
    return core_global


def _soma_skin_vertices(
    soma30_global_rots: np.ndarray,
    soma30_root_positions: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    from motius.models.kimodo.network.skeleton import SOMASkeleton30
    from motius.models.kimodo.network.skeleton.transforms import global_rots_to_local_rots

    skin_path = (
        ROOT
        / "motius/models/kimodo/network/assets/skeletons/somaskel77/skin_standard.npz"
    )
    if not skin_path.exists():
        raise FileNotFoundError(f"SOMA skin asset not found: {skin_path}")
    skin = np.load(skin_path)
    bind_vertices = np.asarray(skin["bind_vertices"], dtype=np.float32)
    faces = np.asarray(skin["faces"], dtype=np.uint32)
    bind = np.asarray(skin["bind_rig_transform"], dtype=np.float32)
    lbs_indices = np.asarray(skin["lbs_indices"], dtype=np.int64)
    lbs_weights = np.asarray(skin["lbs_weights"], dtype=np.float32)

    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    soma30 = SOMASkeleton30().to(device_t)
    global_t = torch.from_numpy(np.asarray(soma30_global_rots, dtype=np.float32)).to(device_t)
    root_t = torch.from_numpy(np.asarray(soma30_root_positions, dtype=np.float32)).to(device_t)
    with torch.no_grad():
        soma30_local = global_rots_to_local_rots(global_t, soma30)
        soma77_local = soma30.to_SOMASkeleton77(soma30_local)
        soma77 = soma30.somaskel77.to(device_t)
        soma77_global, soma77_joints, _ = soma77.fk(soma77_local, root_t)
        root_delta = root_t - soma77_joints[:, 0, :]
        soma77_joints = soma77_joints + root_delta[:, None, :]

    transforms = np.tile(np.eye(4, dtype=np.float32), (len(root_t), 77, 1, 1))
    transforms[:, :, :3, :3] = soma77_global.detach().cpu().numpy().astype(np.float32)
    transforms[:, :, :3, 3] = soma77_joints.detach().cpu().numpy().astype(np.float32)
    skin_transforms = transforms @ np.linalg.inv(bind)[None]
    vertex_h = np.concatenate([bind_vertices, np.ones((len(bind_vertices), 1), dtype=np.float32)], axis=1)
    gathered_t = skin_transforms[:, lbs_indices]
    transformed = np.einsum("tvkij,vj->tvki", gathered_t, vertex_h)
    vertices = (transformed[..., :3] * lbs_weights[None, :, :, None]).sum(axis=2).astype(np.float32)
    vertices[..., 1] -= float(vertices[..., 1].min())
    return vertices, faces


def _core_skin_vertices(
    core_local_rots: np.ndarray,
    root_positions: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    from motius.models.ardy.network.skeleton import CoreSkeleton27
    from motius.models.ardy.network.skeleton.kinematics import fk

    skin_path = (
        ROOT
        / "motius/models/ardy/network/assets/skeletons/cskel27/skin_standard.npz"
    )
    if not skin_path.exists():
        raise FileNotFoundError(f"Core skin asset not found: {skin_path}")
    skin = np.load(skin_path)
    bind_vertices = np.asarray(skin["bind_vertices"], dtype=np.float32)
    faces = np.asarray(skin["faces"], dtype=np.uint32)
    bind = np.asarray(skin["bind_rig_transform"], dtype=np.float32)
    lbs_indices = np.asarray(skin["lbs_indices"], dtype=np.int64)
    lbs_weights = np.asarray(skin["lbs_weights"], dtype=np.float32)

    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    skeleton = CoreSkeleton27().to(device_t)
    with torch.no_grad():
        global_rots, posed_joints, _ = fk(
            torch.from_numpy(np.asarray(core_local_rots, dtype=np.float32)).to(device_t),
            torch.from_numpy(np.asarray(root_positions, dtype=np.float32)).to(device_t),
            skeleton,
        )
    transforms = np.tile(np.eye(4, dtype=np.float32), (len(root_positions), 27, 1, 1))
    transforms[:, :, :3, :3] = global_rots.detach().cpu().numpy().astype(np.float32)
    transforms[:, :, :3, 3] = posed_joints.detach().cpu().numpy().astype(np.float32)
    skin_transforms = transforms @ np.linalg.inv(bind)[None]
    vertex_h = np.concatenate([bind_vertices, np.ones((len(bind_vertices), 1), dtype=np.float32)], axis=1)
    gathered_t = skin_transforms[:, lbs_indices]
    transformed = np.einsum("tvkij,vj->tvki", gathered_t, vertex_h)
    vertices = (transformed[..., :3] * lbs_weights[None, :, :, None]).sum(axis=2).astype(np.float32)
    vertices[..., 1] -= float(vertices[..., 1].min())
    return vertices, faces

def _linear_resample(values: np.ndarray, src_fps: float, dst_fps: float, frames: int) -> np.ndarray:
    src_times = np.arange(len(values), dtype=np.float64) / float(src_fps)
    dst_times = np.arange(frames, dtype=np.float64) / float(dst_fps)
    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    out = np.empty((frames, flat.shape[1]), dtype=np.float32)
    for channel in range(flat.shape[1]):
        out[:, channel] = np.interp(dst_times, src_times, flat[:, channel])
    return out.reshape((frames,) + values.shape[1:])


def _motion135_axis_angle(motion: np.ndarray) -> np.ndarray:
    frames = len(motion)
    matrices = rotation_6d_to_matrix(
        torch.from_numpy(motion[:, 3:135].reshape(-1, 6)), convention="row"
    ).reshape(frames, 22, 3, 3)
    return matrix_to_axis_angle(matrices).reshape(frames, 22, 3).cpu().numpy()


def _body_forward(
    joints: np.ndarray,
    *,
    left_hip: int,
    right_hip: int,
    left_shoulder: int,
    right_shoulder: int,
    reverse: bool = False,
) -> np.ndarray:
    first = np.asarray(joints[0], dtype=np.float64)
    across = 0.5 * (
        first[right_hip] - first[left_hip]
        + first[right_shoulder] - first[left_shoulder]
    )
    across[1] = 0.0
    norm = float(np.linalg.norm(across))
    if norm < 1e-8:
        raise ValueError("Cannot infer initial body heading from hips and shoulders.")
    across /= norm
    forward = np.cross(across, np.asarray([0.0, 1.0, 0.0]))
    if reverse:
        forward *= -1.0
    forward[1] = 0.0
    return (forward / np.linalg.norm(forward)).astype(np.float32)


def _heading_from_forward(forward: np.ndarray) -> np.ndarray:
    forward = np.asarray(forward, dtype=np.float64).copy()
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("Cannot canonicalize a zero-length body heading.")
    forward /= norm
    yaw = -float(np.arctan2(forward[0], forward[2]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )


def _canonical_heading(
    joints: np.ndarray,
    *,
    left_hip: int,
    right_hip: int,
    left_shoulder: int,
    right_shoulder: int,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    forward = _body_forward(
        joints,
        left_hip=left_hip,
        right_hip=right_hip,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        reverse=reverse,
    )
    heading = _heading_from_forward(forward)
    origin = np.asarray([joints[0, 0, 0], 0.0, joints[0, 0, 2]], dtype=np.float32)
    return heading, origin


def _apply_heading(points: np.ndarray, heading: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return ((np.asarray(points, dtype=np.float32) - origin) @ heading.T).astype(np.float32)


def _canonicalize_human_joints(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heading, origin = _canonical_heading(
        joints,
        left_hip=1,
        right_hip=2,
        left_shoulder=16,
        right_shoulder=17,
        reverse=True,
    )
    canonical = _apply_heading(joints, heading, origin)
    canonical[..., 1] -= float(canonical[..., 1].min())
    return canonical, heading


def _smpl_surface(
    axis_angle: np.ndarray,
    transl: np.ndarray,
    *,
    model_dir: Path,
    gender: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import smplx

    standard_model = model_dir / "smplh" / f"SMPLH_{gender.upper()}.pkl"
    resolved = (
        standard_model
        if standard_model.is_file()
        else resolve_smpl_model_path(model_dir, model_type="smplh", gender=gender)
    )
    if model_dir.is_file():
        body_model = smplx.SMPLH(str(resolved), gender=gender, use_pca=False)
    else:
        body_model = smplx.create(
            str(model_dir),
            model_type="smplh",
            gender=gender,
            ext=resolved.suffix.lstrip("."),
            use_pca=False,
        )
    frames = len(axis_angle)
    zeros_hand = torch.zeros((frames, 45), dtype=torch.float32)
    with torch.inference_mode():
        output = body_model(
            global_orient=torch.from_numpy(axis_angle[:, 0]).float(),
            body_pose=torch.from_numpy(axis_angle[:, 1:].reshape(frames, 63)).float(),
            left_hand_pose=zeros_hand,
            right_hand_pose=zeros_hand,
            transl=torch.from_numpy(np.asarray(transl, dtype=np.float32)),
            betas=torch.zeros((frames, int(body_model.num_betas)), dtype=torch.float32),
        )
    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    joints = output.joints[:, :22].detach().cpu().numpy().astype(np.float32)
    return vertices, joints, np.asarray(body_model.faces, dtype=np.uint32)


def _quantize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=(0, 1)).astype(np.float32)
    maximum = vertices.max(axis=(0, 1)).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    quantized = np.rint((vertices - minimum) / scale).clip(0, 65535).astype("<u2")
    return quantized, minimum, scale


def _quantize_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.empty_like(vertices, dtype=np.float32)
    for frame, frame_vertices in enumerate(vertices):
        frame_normals = np.zeros_like(frame_vertices, dtype=np.float32)
        triangles = frame_vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        np.add.at(frame_normals, faces[:, 0], face_normals)
        np.add.at(frame_normals, faces[:, 1], face_normals)
        np.add.at(frame_normals, faces[:, 2], face_normals)
        lengths = np.linalg.norm(frame_normals, axis=-1, keepdims=True)
        normals[frame] = frame_normals / np.maximum(lengths, 1e-8)
    return np.rint(normals * 127.0).clip(-127, 127).astype("i1")


def _g1_mesh_assets(
    qpos: np.ndarray,
    xml_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    geom_ids = np.where(
        (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_group == 1)
    )[0]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in G1_BODY_NAMES]
    frames = len(qpos)
    body_positions = np.empty((frames, len(body_ids), 3), dtype=np.float32)
    geom_positions = np.empty((frames, len(geom_ids), 3), dtype=np.float32)
    geom_rotations = np.empty((frames, len(geom_ids), 3, 3), dtype=np.float32)
    pelvis_forwards = np.empty((frames, 3), dtype=np.float32)

    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        body_positions[frame] = data.xpos[body_ids]
        geom_positions[frame] = data.geom_xpos[geom_ids]
        geom_rotations[frame] = data.geom_xmat[geom_ids].reshape(-1, 3, 3)
        pelvis_rotation = data.xmat[body_ids[0]].reshape(3, 3)
        pelvis_forwards[frame] = pelvis_rotation[:, 0]

    body_positions = body_positions @ GMR_Y_UP_FROM_Z_UP.T
    geom_positions = geom_positions @ GMR_Y_UP_FROM_Z_UP.T
    pelvis_forwards = pelvis_forwards @ GMR_Y_UP_FROM_Z_UP.T
    geom_rotations = (
        GMR_Y_UP_FROM_Z_UP[None, None]
        @ geom_rotations
        @ GMR_Y_UP_FROM_Z_UP.T[None, None]
    )
    heading = _heading_from_forward(pelvis_forwards[0])
    origin = np.asarray(
        [body_positions[0, 0, 0], 0.0, body_positions[0, 0, 2]], dtype=np.float32
    )
    body_positions = _apply_heading(body_positions, heading, origin)
    geom_positions = _apply_heading(geom_positions, heading, origin)
    pelvis_forwards = pelvis_forwards @ heading.T
    pelvis_forwards[:, 1] = 0.0
    pelvis_forwards /= np.maximum(np.linalg.norm(pelvis_forwards, axis=-1, keepdims=True), 1e-8)
    geom_rotations = heading[None, None] @ geom_rotations

    vertices_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    metadata: list[dict] = []
    vertex_offset = 0
    index_offset = 0
    geom_local_vertices: list[np.ndarray] = []
    for output_index, geom_id in enumerate(geom_ids):
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_begin = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_begin = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_begin : vertex_begin + vertex_count], dtype=np.float32
        ) @ GMR_Y_UP_FROM_Z_UP.T
        indices = np.asarray(
            model.mesh_face[face_begin : face_begin + face_count], dtype=np.uint32
        ).reshape(-1)
        vertices_parts.append(vertices)
        index_parts.append(indices)
        geom_local_vertices.append(vertices)
        metadata.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
                "vertex_offset": vertex_offset,
                "vertex_count": vertex_count,
                "index_offset": index_offset,
                "index_count": int(len(indices)),
                "color": np.round(model.geom_rgba[geom_id], 4).tolist(),
                "transform_index": output_index,
            }
        )
        vertex_offset += vertex_count
        index_offset += len(indices)

    floor = np.inf
    for geom_index, vertices in enumerate(geom_local_vertices):
        y_axes = geom_rotations[:, geom_index, 1, :]
        local_y = y_axes @ vertices.T
        floor = min(floor, float((local_y + geom_positions[:, geom_index, 1:2]).min()))
    geom_positions[..., 1] -= floor
    body_positions[..., 1] -= floor

    quaternions = Rotation.from_matrix(geom_rotations.reshape(-1, 3, 3)).as_quat()
    quaternions = quaternions.reshape(frames, len(geom_ids), 4).astype(np.float32)
    transforms = np.concatenate([geom_positions, quaternions], axis=-1).astype("<f4")
    return (
        np.concatenate(vertices_parts, axis=0).astype("<f4"),
        np.concatenate(index_parts, axis=0).astype("<u4"),
        transforms,
        metadata,
        pelvis_forwards,
    )


def _round_nested(values: np.ndarray, digits: int = 5):
    return np.round(values, digits).tolist()


def build(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = np.load(args.hml_fixture)
    hml_joints = np.asarray(fixture["joints"], dtype=np.float32)
    motion135 = np.asarray(np.load(args.motion135)["motion_135"], dtype=np.float32)
    frames = min(args.max_frames, len(motion135))
    motion135 = motion135[:frames]
    hml_joints = _linear_resample(hml_joints, args.hml_fps, args.fps, frames)
    hml_joints, _ = _canonicalize_human_joints(hml_joints)

    axis_angle = _motion135_axis_angle(motion135)
    smpl_vertices, smpl_joints, smpl_faces = _smpl_surface(
        axis_angle,
        motion135[:, :3],
        model_dir=args.smpl_model_dir,
        gender=args.gender,
    )
    smpl_heading, smpl_origin = _canonical_heading(
        smpl_joints,
        left_hip=1,
        right_hip=2,
        left_shoulder=16,
        right_shoulder=17,
        reverse=True,
    )
    smpl_vertices = _apply_heading(smpl_vertices, smpl_heading, smpl_origin)
    smpl_joints = _apply_heading(smpl_joints, smpl_heading, smpl_origin)
    smpl_floor = float(smpl_vertices[..., 1].min())
    smpl_vertices[..., 1] -= smpl_floor
    smpl_joints[..., 1] -= smpl_floor
    smpl_quantized, smpl_minimum, smpl_scale = _quantize_vertices(smpl_vertices)
    smpl_normals = _quantize_vertex_normals(smpl_vertices, smpl_faces)

    soma_result = smpl_motion135_to_soma30(
        motion135,
        assets_root=ROOT / "motius/models/kimodo/network/assets/skeletons",
    )
    soma_vertices, soma_faces = _soma_skin_vertices(
        soma_result["soma30_global_rots"],
        soma_result["soma30_joints"][:, 0, :],
        device="cpu",
    )
    soma_vertices = _apply_heading(soma_vertices, smpl_heading, smpl_origin)
    soma_vertices[..., 1] -= float(soma_vertices[..., 1].min())
    soma_quantized, soma_minimum, soma_scale = _quantize_vertices(soma_vertices)
    soma_normals = _quantize_vertex_normals(soma_vertices, soma_faces)

    smpl_local_rots = _motion135_local_rot_mats(motion135)
    smpl_global_rots = _smpl_local_to_global(smpl_local_rots)
    core_global_rots = _smpl_global_to_core27_global(smpl_global_rots)
    core_local_rots = _global_to_local_mats(core_global_rots, CORE27_PARENTS)
    core_vertices, core_faces = _core_skin_vertices(
        core_local_rots,
        motion135[:, :3],
        device="cpu",
    )
    core_vertices = _apply_heading(core_vertices, smpl_heading, smpl_origin)
    core_vertices[..., 1] -= float(core_vertices[..., 1].min())
    core_quantized, core_minimum, core_scale = _quantize_vertices(core_vertices)
    core_normals = _quantize_vertex_normals(core_vertices, core_faces)

    retargeter = GMRSMPLToG1Retargeter(
        tgt_fps=int(args.fps),
        smplx_model_dir=args.smpl_model_dir,
    )
    g1_result = retargeter.retarget_from_motion135(
        motion135,
        fps=args.fps,
        betas=np.zeros(16, dtype=np.float32),
        gender=args.gender,
    )
    g1_qpos = retargeter.to_mujoco_qpos(g1_result)
    (
        g1_vertices,
        g1_indices,
        g1_transforms,
        g1_geoms,
        g1_forwards,
    ) = _g1_mesh_assets(g1_qpos, retargeter.robot_xml)

    smpl_quantized.tofile(args.output_dir / "smpl_vertices.u16")
    smpl_normals.tofile(args.output_dir / "smpl_normals.i8")
    smpl_faces.astype("<u4").reshape(-1).tofile(args.output_dir / "smpl_indices.u32")
    soma_quantized.tofile(args.output_dir / "soma_vertices.u16")
    soma_normals.tofile(args.output_dir / "soma_normals.i8")
    soma_faces.astype("<u4").reshape(-1).tofile(args.output_dir / "soma_indices.u32")
    core_quantized.tofile(args.output_dir / "core_vertices.u16")
    core_normals.tofile(args.output_dir / "core_normals.i8")
    core_faces.astype("<u4").reshape(-1).tofile(args.output_dir / "core_indices.u32")
    g1_vertices.tofile(args.output_dir / "g1_mesh_vertices.f32")
    g1_indices.tofile(args.output_dir / "g1_mesh_indices.u32")
    g1_transforms.tofile(args.output_dir / "g1_geom_transforms.f32")

    hml_forward = _body_forward(
        hml_joints,
        left_hip=1,
        right_hip=2,
        left_shoulder=16,
        right_shoulder=17,
        reverse=True,
    )
    smpl_forward = _body_forward(
        smpl_joints,
        left_hip=1,
        right_hip=2,
        left_shoulder=16,
        right_shoulder=17,
        reverse=True,
    )
    g1_forward = g1_forwards[0]
    payload = {
        "case_id": args.case_id,
        "fps": args.fps,
        "frames": frames,
        "duration_seconds": round(frames / args.fps, 3),
        "representations": {
            "humanml3d": {
                "label": "HumanML3D-263",
                "parents": list(SMPL22_PARENTS),
                "positions": _round_nested(hml_joints),
                "initial_forward": _round_nested(hml_forward),
            },
            "smpl": {
                "label": "SMPL",
                "vertex_count": int(smpl_vertices.shape[1]),
                "index_count": int(smpl_faces.size),
                "vertices_file": "smpl_vertices.u16",
                "normals_file": "smpl_normals.i8",
                "indices_file": "smpl_indices.u32",
                "quantization_min": smpl_minimum.tolist(),
                "quantization_scale": smpl_scale.tolist(),
                "initial_forward": _round_nested(smpl_forward),
            },
            "soma": {
                "label": "SOMA-30",
                "vertex_count": int(soma_vertices.shape[1]),
                "index_count": int(soma_faces.size),
                "vertices_file": "soma_vertices.u16",
                "normals_file": "soma_normals.i8",
                "indices_file": "soma_indices.u32",
                "quantization_min": soma_minimum.tolist(),
                "quantization_scale": soma_scale.tolist(),
                "initial_forward": _round_nested(smpl_forward),
            },
            "core": {
                "label": "Core-27",
                "vertex_count": int(core_vertices.shape[1]),
                "index_count": int(core_faces.size),
                "vertices_file": "core_vertices.u16",
                "normals_file": "core_normals.i8",
                "indices_file": "core_indices.u32",
                "quantization_min": core_minimum.tolist(),
                "quantization_scale": core_scale.tolist(),
                "initial_forward": _round_nested(smpl_forward),
            },
            "g1": {
                "label": "Unitree G1",
                "vertex_count": int(len(g1_vertices)),
                "index_count": int(len(g1_indices)),
                "geom_count": len(g1_geoms),
                "vertices_file": "g1_mesh_vertices.f32",
                "indices_file": "g1_mesh_indices.u32",
                "transforms_file": "g1_geom_transforms.f32",
                "geoms": g1_geoms,
                "initial_forward": _round_nested(g1_forward),
                "forward_basis": "MuJoCo pelvis local +X axis",
            },
        },
        "provenance": {
            "humanml3d": str(args.hml_fixture),
            "smpl_motion135": str(args.motion135),
            "soma_route": "SMPL motion135 -> SOMA30 rotation transfer -> SOMA77 LBS mesh",
            "core_route": "SMPL motion135 global rotations -> Core-27 visual rotation bridge -> Core LBS mesh",
            "g1_route": "SMPL motion135 -> GMR IK -> G1 qpos -> MuJoCo mesh FK",
            "body_model": "local licensed SMPL-H parameters; only demo geometry is exported",
        },
    }
    json_path = args.output_dir / "data.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")))
    (args.output_dir / "data.js").write_text(
        "window.MOTIUS_REPRESENTATION_DEMO="
        + json.dumps(payload, separators=(",", ":"))
        + ";\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "fps": args.fps,
                "frames": frames,
                "viewer_data": "data.js",
                "assets": [
                    "smpl_vertices.u16",
                    "smpl_normals.i8",
                    "smpl_indices.u32",
                    "soma_vertices.u16",
                    "soma_normals.i8",
                    "soma_indices.u32",
                    "core_vertices.u16",
                    "core_normals.i8",
                    "core_indices.u32",
                    "g1_mesh_vertices.f32",
                    "g1_mesh_indices.u32",
                    "g1_geom_transforms.f32",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hml-fixture", type=Path, required=True)
    parser.add_argument("--motion135", type=Path, required=True)
    parser.add_argument("--smpl-model-dir", type=Path, default=Path("checkpoints/smpl_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/representation_demo/004822"))
    parser.add_argument("--case-id", default="004822")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hml-fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    return parser.parse_args()


if __name__ == "__main__":
    output = build(parse_args())
    print(output)
