#!/usr/bin/env python3
"""Build a GT InterHuman skeleton versus SMPL mesh representation demo."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.representation.interhuman262 import (  # noqa: E402
    interhuman262_to_joints,
    joints_pair_to_interhuman262,
)
from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip  # noqa: E402
from motius.motion.skeleton import SMPL22_PARENTS  # noqa: E402
from render_motion135_smpl_demo import SMPLRenderer  # noqa: E402


def _load_raw_pair(data_root: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    joints = []
    rotations = []
    for person in ("person1", "person2"):
        path = data_root / "motions_processed" / person / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        motion = np.load(path).astype(np.float32)
        if motion.ndim != 2 or motion.shape[1] < 62 * 3 + 21 * 6:
            raise ValueError(
                f"{path} must have InterHuman raw processed shape (T, >=312), got {motion.shape}"
            )
        joints.append(motion[:, : 22 * 3].reshape(len(motion), 22, 3))
        rotations.append(motion[:, 62 * 3 : 62 * 3 + 21 * 6].reshape(len(motion), 21, 6))
    n = min(len(joints[0]), len(joints[1]))
    return (
        np.stack([value[:n] for value in joints], axis=1),
        np.stack([value[:n] for value in rotations], axis=1),
    )


def _fit_pair_vertices(
    joints: np.ndarray,
    renderer: SMPLRenderer,
    smpl_rest,
    *,
    model_dir: Path,
    device: str,
    fps: int,
    refine_iters: int,
) -> tuple[np.ndarray, float]:
    people = []
    errors = []
    for person in range(2):
        fit = retarget_hml263_clip(
            joints[:, person],
            smpl_rest=smpl_rest,
            model_dir=model_dir,
            device=device,
            source_fps=fps,
            target_fps=fps,
            refine_iters=refine_iters,
            floor_align=False,
            rotation_init="position_ik",
        )
        people.append(renderer.vertices(fit["global_orient"], fit["body_pose"], fit["transl"]))
        errors.append(float(np.asarray(fit["fit_mpjpe_mm"]).mean()))
    vertices = np.stack(people, axis=1).astype(np.float32)
    vertices[..., 1] -= float(vertices[..., 1].min())
    return vertices, float(np.mean(errors))


def _center_pair(points: np.ndarray, x_offset: float) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32).copy()
    ground = float(value[..., 1].min())
    value[..., 1] -= ground
    center = value[0].reshape(-1, 3).mean(axis=0)
    center[1] = 0.0
    value -= center
    value[..., 0] += x_offset
    return value


def _look_at(eye: np.ndarray, center: np.ndarray) -> np.ndarray:
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    forward = center - eye
    forward /= np.linalg.norm(forward) + 1e-9
    side = np.cross(forward, up)
    side /= np.linalg.norm(side) + 1e-9
    true_up = np.cross(side, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = side
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _capsule_between(
    start: np.ndarray, end: np.ndarray, radius: float, sections: int = 8
) -> trimesh.Trimesh | None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-5:
        return None
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    midpoint = (start + end) * 0.5
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction / length)
    transform[:3, 3] = midpoint
    mesh.apply_transform(transform)
    return mesh


def _skeleton_mesh(joints: np.ndarray, radius: float) -> trimesh.Trimesh:
    parts = []
    for joint, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            continue
        segment = _capsule_between(joints[parent], joints[joint], radius)
        if segment is not None:
            parts.append(segment)
    for joint in joints:
        sphere = trimesh.creation.uv_sphere(radius=radius * 1.55, count=[8, 8])
        sphere.apply_translation(joint)
        parts.append(sphere)
    return trimesh.util.concatenate(parts)


def render_interhuman_skeleton_smpl_mesh(
    joints: np.ndarray,
    smpl_vertices: np.ndarray,
    smpl_faces: np.ndarray,
    output: Path,
    *,
    fps: int,
    width: int,
    height: int,
    max_frames: int,
) -> list[np.ndarray]:
    skel = _center_pair(joints[:max_frames], -1.25)
    mesh = _center_pair(smpl_vertices[: len(skel)], 1.25)
    all_points = np.concatenate([skel.reshape(-1, 3), mesh.reshape(-1, 3)], axis=0)
    y_center = float(np.percentile(all_points[:, 1], 52))
    target = np.asarray([0.0, y_center, float(np.mean(all_points[:, 2]))], dtype=np.float32)
    span = np.ptp(all_points, axis=0)
    radius = max(float(span.max()), 2.2)
    dist = max(3.8, radius * 1.35)
    eye = target + np.asarray([0.72 * dist, 0.42 * dist, 1.05 * dist], dtype=np.float32)
    camera_pose = _look_at(eye, target)
    materials = {
        "skeleton_a": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.08, 0.42, 0.95, 1.0), metallicFactor=0.0, roughnessFactor=0.54
        ),
        "skeleton_b": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.02, 0.68, 0.63, 1.0), metallicFactor=0.0, roughnessFactor=0.56
        ),
        "mesh_a": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.20, 0.44, 0.92, 1.0), metallicFactor=0.02, roughnessFactor=0.58
        ),
        "mesh_b": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.94, 0.38, 0.25, 1.0), metallicFactor=0.02, roughnessFactor=0.58
        ),
        "floor": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.90, 0.91, 0.93, 1.0), metallicFactor=0.0, roughnessFactor=1.0
        ),
    }
    renderer = pyrender.OffscreenRenderer(width, height)
    frames: list[np.ndarray] = []
    try:
        for frame in range(len(skel)):
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.48, 0.48, 0.52])
            for person, mat_name in enumerate(("skeleton_a", "skeleton_b")):
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        _skeleton_mesh(skel[frame, person], 0.012),
                        material=materials[mat_name],
                    )
                )
            for person, mat_name in enumerate(("mesh_a", "mesh_b")):
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(mesh[frame, person], smpl_faces, process=False),
                        material=materials[mat_name],
                        smooth=True,
                    )
                )
            floor_size = max(4.4, radius * 1.3)
            floor = trimesh.creation.box(extents=(floor_size, 0.012, floor_size))
            floor.apply_translation([0.0, -0.006, target[2]])
            scene.add(pyrender.Mesh.from_trimesh(floor, material=materials["floor"], smooth=False))
            scene.add(pyrender.PerspectiveCamera(yfov=math.radians(36), aspectRatio=width / height), pose=camera_pose)
            for light_eye, intensity in [
                (target + np.asarray([2.6, 4.2, 3.0], dtype=np.float32), 4.0),
                (target + np.asarray([-3.0, 2.8, 1.4], dtype=np.float32), 1.9),
                (target + np.asarray([0.0, 3.3, -3.2], dtype=np.float32), 1.5),
            ]:
                scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                    pose=_look_at(light_eye, target),
                )
            color, _ = renderer.render(scene)
            frames.append(color)
    finally:
        renderer.delete()
    imageio.mimsave(output, frames, duration=1.0 / fps, loop=0)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/interhuman"))
    parser.add_argument("--sample-id", default="407")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("assets/motion/interhuman_representation_demo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--refine-iters", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--write-npz", action="store_true")
    parser.add_argument("--write-mp4", action="store_true")
    args = parser.parse_args()

    raw_joints, raw_rotations = _load_raw_pair(args.data_root, args.sample_id)
    interhuman = joints_pair_to_interhuman262(
        raw_joints,
        raw_rotations,
        feet_threshold=0.001,
        reference_frame=0,
        source_coordinates="interhuman_raw",
    )
    interhuman = interhuman[: args.frames]
    joints = interhuman262_to_joints(interhuman)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    renderer = SMPLRenderer(args.model_dir, args.device, args.width, args.height)
    smpl_rest = load_smpl_rest(args.model_dir, args.device)
    smpl_vertices, fit_mpjpe_mm = _fit_pair_vertices(
        joints,
        renderer,
        smpl_rest,
        model_dir=args.model_dir,
        device=args.device,
        fps=args.fps,
        refine_iters=args.refine_iters,
    )
    stem = f"interhuman_gt_{args.sample_id}_skeleton_smpl_mesh"
    gif = args.out_dir / f"{stem}.gif"
    frames = render_interhuman_skeleton_smpl_mesh(
        joints,
        smpl_vertices,
        renderer.faces,
        gif,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.frames,
    )
    meta = {
        "sample_id": args.sample_id,
        "source": "GT InterHuman motions_processed/person1+person2",
        "representation": "InterHuman-262 skeleton to SMPL mesh",
        "route": "raw InterHuman 492D -> joints_pair_to_interhuman262 -> position-IK SMPL mesh",
        "fps": args.fps,
        "frames": len(frames),
        "fit_mpjpe_mm": fit_mpjpe_mm,
        "gif": str(gif),
    }
    if args.write_npz:
        npz = args.out_dir / f"{stem}.npz"
        np.savez_compressed(
            npz,
            interhuman262=interhuman,
            interhuman_joints=joints,
            smpl_vertices=smpl_vertices,
            smpl_faces=renderer.faces,
            sample_id=args.sample_id,
            fps=args.fps,
            fit_mpjpe_mm=fit_mpjpe_mm,
        )
        meta["npz"] = str(npz)
    if args.write_mp4:
        mp4 = args.out_dir / f"{stem}.mp4"
        imageio.mimwrite(mp4, frames, fps=args.fps, quality=8, macro_block_size=1)
        meta["mp4"] = str(mp4)
    (args.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
