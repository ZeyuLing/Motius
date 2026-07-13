#!/usr/bin/env python3
"""Build synchronized HumanML3D, SMPL-22, and Unitree G1 viewer data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motius.motion.representation.rotation import matrix_to_axis_angle, rotation_6d_to_matrix
from motius.motion.retarget import GMRSMPLToG1Retargeter
from motius.motion.skeleton import SMPL22_PARENTS, smpl_to_joints


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


def _g1_fk(qpos: np.ndarray, xml_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in G1_BODY_NAMES]
    id_to_output = {body_id: idx for idx, body_id in enumerate(body_ids)}
    parents = []
    for body_id in body_ids:
        parent_id = int(model.body_parentid[body_id])
        while parent_id not in id_to_output and parent_id > 0:
            parent_id = int(model.body_parentid[parent_id])
        parents.append(id_to_output.get(parent_id, -1))

    positions = np.empty((len(qpos), len(body_ids), 3), dtype=np.float32)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[body_ids]
    return positions, parents, list(G1_BODY_NAMES)


def _viewer_coordinates(joints: np.ndarray, root: int, source_up: str) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32).copy()
    if source_up == "z":
        joints = joints[..., [0, 2, 1]]
        joints[..., 2] *= -1
    elif source_up != "y":
        raise ValueError(f"Unsupported up axis: {source_up}")
    joints[..., 0] -= joints[0, root, 0]
    joints[..., 2] -= joints[0, root, 2]
    joints[..., 1] -= float(joints[..., 1].min())
    return joints


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

    axis_angle = _motion135_axis_angle(motion135)
    smpl_joints = smpl_to_joints(
        axis_angle[:, 0],
        axis_angle[:, 1:].reshape(frames, 63),
        motion135[:, :3],
        betas=np.zeros(16, dtype=np.float32),
        gender=args.gender,
        model_type="smplh",
        model_path=args.smpl_model_dir,
    )

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
    g1_positions, g1_parents, g1_names = _g1_fk(g1_qpos, retargeter.robot_xml)

    hml_joints = _viewer_coordinates(hml_joints, root=0, source_up="y")
    smpl_joints = _viewer_coordinates(smpl_joints, root=0, source_up="y")
    g1_positions = _viewer_coordinates(g1_positions, root=0, source_up="z")

    payload = {
        "case_id": args.case_id,
        "caption": args.caption,
        "fps": args.fps,
        "frames": frames,
        "duration_seconds": round(frames / args.fps, 3),
        "representations": {
            "humanml3d": {
                "label": "HumanML3D-263",
                "detail": "decoded SMPL-22 joints",
                "parents": list(SMPL22_PARENTS),
                "positions": _round_nested(hml_joints),
            },
            "smpl": {
                "label": "SMPL motion135",
                "detail": "shape-aware SMPL-22 FK",
                "parents": list(SMPL22_PARENTS),
                "positions": _round_nested(smpl_joints),
            },
            "g1": {
                "label": "Unitree G1-38D",
                "detail": "GMR-retargeted robot FK",
                "parents": g1_parents,
                "names": g1_names,
                "positions": _round_nested(g1_positions),
            },
        },
        "provenance": {
            "humanml3d": str(args.hml_fixture),
            "smpl_motion135": str(args.motion135),
            "g1_route": "SMPL motion135 -> GMR IK -> G1 qpos -> MuJoCo FK",
            "body_model": "local licensed SMPL-H/SMPL-X files; not redistributed",
        },
    }
    json_path = args.output_dir / "data.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")))
    (args.output_dir / "data.js").write_text(
        "window.MOTIUS_REPRESENTATION_DEMO=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    np.savez_compressed(
        args.output_dir / "source_arrays.npz",
        hml_joints=hml_joints,
        smpl_joints=smpl_joints,
        g1_qpos=g1_qpos,
        g1_positions=g1_positions,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "caption": args.caption,
                "fps": args.fps,
                "frames": frames,
                "viewer_data": "data.js",
                "source_arrays": "source_arrays.npz",
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
    parser.add_argument(
        "--caption",
        default="A person walks forward at an average pace, swaying their arms and torso with swagger.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hml-fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    return parser.parse_args()


if __name__ == "__main__":
    output = build(parse_args())
    print(output)
