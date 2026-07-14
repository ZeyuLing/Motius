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
from motius.motion.retarget.ardy_core import (
    ARDY_CORE27_NAMES,
    smpl22_joints_to_ardy_core27_joints,
)
from motius.motion.retarget.smpl_soma import (
    SOMA30_NAMES,
    SOMA30_PARENTS,
    SMPL22_TO_SOMA30,
)
from motius.motion.skeleton import SMPL22_PARENTS
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


def _smpl22_joints_to_soma30_preview(joints: np.ndarray) -> np.ndarray:
    value = np.asarray(joints, dtype=np.float32)
    if value.shape[-2:] != (22, 3):
        raise ValueError(f"SMPL joints must end in (22, 3), got {value.shape}")
    soma = np.empty(value.shape[:-2] + (len(SOMA30_NAMES), 3), dtype=np.float32)
    for smpl_index, soma_index in enumerate(SMPL22_TO_SOMA30.tolist()):
        soma[..., soma_index, :] = value[..., smpl_index, :]

    def extend(target: str, base: str, parent: str, scale: float) -> None:
        target_i = SOMA30_NAMES.index(target)
        base_i = SOMA30_NAMES.index(base)
        parent_i = SOMA30_NAMES.index(parent)
        soma[..., target_i, :] = soma[..., base_i, :] + (
            soma[..., base_i, :] - soma[..., parent_i, :]
        ) * scale

    head = soma[..., SOMA30_NAMES.index("Head"), :]
    neck = soma[..., SOMA30_NAMES.index("Neck2"), :]
    soma[..., SOMA30_NAMES.index("Jaw"), :] = head + np.asarray([0.0, -0.08, 0.06], dtype=np.float32)
    soma[..., SOMA30_NAMES.index("LeftEye"), :] = head + np.asarray([0.045, 0.035, 0.055], dtype=np.float32)
    soma[..., SOMA30_NAMES.index("RightEye"), :] = head + np.asarray([-0.045, 0.035, 0.055], dtype=np.float32)
    soma[..., SOMA30_NAMES.index("Neck2"), :] = neck
    extend("LeftHandThumbEnd", "LeftHand", "LeftForeArm", 0.20)
    extend("LeftHandMiddleEnd", "LeftHand", "LeftForeArm", 0.36)
    extend("RightHandThumbEnd", "RightHand", "RightForeArm", 0.20)
    extend("RightHandMiddleEnd", "RightHand", "RightForeArm", 0.36)
    return soma

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

    soma_joints = _smpl22_joints_to_soma30_preview(smpl_joints)
    soma_joints[..., 1] -= float(soma_joints[..., 1].min())
    core_joints = smpl22_joints_to_ardy_core27_joints(smpl_joints)
    core_joints[..., 1] -= float(core_joints[..., 1].min())

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
                "parents": SOMA30_PARENTS.astype(int).tolist(),
                "positions": _round_nested(soma_joints),
                "initial_forward": _round_nested(smpl_forward),
                "source_names": list(SOMA30_NAMES),
            },
            "core": {
                "label": "Core-27",
                "parents": CORE27_PARENTS.astype(int).tolist(),
                "positions": _round_nested(core_joints),
                "initial_forward": _round_nested(smpl_forward),
                "source_names": list(ARDY_CORE27_NAMES),
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
            "soma_route": "SMPL-22 joints -> named SOMA-30 preview bridge",
            "core_route": "SMPL-22 joints -> named Core-27 joint-position bridge",
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
