#!/usr/bin/env python3
"""Render a motion_135 NPZ as an SMPL mesh demo.

The tool accepts either a fully fitted NPZ with ``global_orient``, ``body_pose``
and ``transl`` arrays, or a compact NPZ containing only ``motion_135``
(``transl`` + 22 row-major 6D rotations). It writes an MP4 and an inline GIF
under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio.v2 as imageio
import numpy as np
import pyrender
import torch
import trimesh

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


def _rot6d_row_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major 6D rotations to axis-angle."""
    d6 = torch.as_tensor(rot6d[..., [0, 2, 4, 1, 3, 5]], dtype=torch.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot = torch.stack((b1, b2, b3), dim=-1)
    return _matrix_to_axis_angle(rot).cpu().numpy().astype(np.float32)


def _matrix_to_axis_angle(rot: torch.Tensor) -> torch.Tensor:
    cos_angle = ((rot.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    vec = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    denom = 2.0 * torch.sin(angle).unsqueeze(-1)
    axis = vec / denom.clamp(min=1e-6)
    small = angle.abs() < 1e-4
    axis = torch.where(small.unsqueeze(-1), torch.zeros_like(axis), axis)
    return axis * angle.unsqueeze(-1)


def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    meta = {"source": str(path), "keys": list(data.files)}
    if {"global_orient", "body_pose", "transl"}.issubset(data.files):
        global_orient = np.asarray(data["global_orient"], dtype=np.float32).reshape(-1, 3)
        body_pose = np.asarray(data["body_pose"], dtype=np.float32).reshape(len(global_orient), -1, 3)
        if body_pose.shape[1] < 21:
            raise ValueError(f"body_pose has {body_pose.shape[1]} joints, expected at least 21")
        transl = np.asarray(data["transl"], dtype=np.float32).reshape(len(global_orient), 3)
        return global_orient, body_pose[:, :21].reshape(len(global_orient), 63), transl, meta
    if "motion_135" not in data.files:
        raise KeyError(f"{path} must contain either SMPL params or motion_135")
    motion = np.asarray(data["motion_135"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 135:
        raise ValueError(f"motion_135 must have shape (T, >=135), got {motion.shape}")
    transl = motion[:, :3]
    aa = _rot6d_row_to_axis_angle(motion[:, 3:135].reshape(len(motion), 22, 6))
    return aa[:, 0], aa[:, 1:22].reshape(len(motion), 63), transl, meta


class SMPLRenderer:
    def __init__(self, model_dir: Path, device: str, width: int, height: int) -> None:
        import smplx

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = smplx.create(
            str(model_dir),
            model_type="smpl",
            gender="neutral",
            batch_size=1,
            use_pca=False,
        ).to(self.device)
        self.model.eval()
        self.faces = np.asarray(self.model.faces, dtype=np.int32)
        self.width = width
        self.height = height

    @torch.no_grad()
    def vertices(self, global_orient: np.ndarray, body_pose: np.ndarray, transl: np.ndarray) -> np.ndarray:
        verts = []
        for start in range(0, len(global_orient), 96):
            end = min(start + 96, len(global_orient))
            count = end - start
            body69 = np.zeros((count, 69), dtype=np.float32)
            body69[:, :63] = body_pose[start:end]
            self.model.batch_size = count
            res = self.model(
                betas=torch.zeros(count, 10, device=self.device),
                body_pose=torch.from_numpy(body69).to(self.device),
                global_orient=torch.from_numpy(global_orient[start:end]).to(self.device),
                transl=torch.from_numpy(transl[start:end]).to(self.device),
            )
            verts.append(res.vertices.detach().cpu().numpy())
        out = np.concatenate(verts, axis=0)
        out[..., 1] -= out[..., 1].min()
        return out

    @staticmethod
    def _look_at(eye: np.ndarray, center: np.ndarray) -> np.ndarray:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
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

    def render(self, verts: np.ndarray, fps: int, max_frames: int) -> list[np.ndarray]:
        verts = verts[: min(max_frames, len(verts))]
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        frames: list[np.ndarray] = []
        all_pts = verts.reshape(-1, 3)
        y_center = float(np.percentile(all_pts[:, 1], 54))
        span = np.ptp(all_pts[:, [0, 1, 2]], axis=0)
        radius = max(float(span[0]), float(span[1]), float(span[2]), 1.6)
        dist = max(2.7, radius * 1.35)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.25, 0.48, 0.92, 1.0),
            metallicFactor=0.03,
            roughnessFactor=0.58,
        )
        floor_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.88, 0.90, 0.93, 1.0),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        try:
            for frame_verts in verts:
                center = frame_verts.mean(axis=0)
                target = np.array([center[0], y_center, center[2]], dtype=np.float32)
                eye = target + np.array([0.82 * dist, 0.42 * dist, 1.0 * dist], dtype=np.float32)
                scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.42, 0.42, 0.46])
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(frame_verts, self.faces, process=False),
                        material=material,
                        smooth=True,
                    )
                )
                floor_size = max(3.2, radius * 1.35)
                floor = trimesh.creation.box(extents=(floor_size, 0.012, floor_size))
                floor.apply_translation([target[0], -0.006, target[2]])
                scene.add(pyrender.Mesh.from_trimesh(floor, material=floor_mat, smooth=False))
                scene.add(
                    pyrender.PerspectiveCamera(yfov=math.radians(38), aspectRatio=self.width / self.height),
                    pose=self._look_at(eye, target),
                )
                for light_eye, intensity in [
                    (target + np.array([2.5, 4.0, 3.0], dtype=np.float32), 3.8),
                    (target + np.array([-3.0, 2.5, 1.2], dtype=np.float32), 1.8),
                    (target + np.array([0.0, 3.2, -3.0], dtype=np.float32), 1.4),
                ]:
                    scene.add(
                        pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                        pose=self._look_at(light_eye, target),
                    )
                color, _ = renderer.render(scene)
                frames.append(color)
        finally:
            renderer.delete()
        if not frames:
            raise RuntimeError("render produced zero frames")
        return frames

    def render_pair(self, verts: np.ndarray, fps: int, max_frames: int) -> list[np.ndarray]:
        """Render two synchronized SMPL tracks shaped ``(T, 2, V, 3)``."""
        if verts.ndim != 4 or verts.shape[1] != 2:
            raise ValueError(f"pair vertices must have shape (T, 2, V, 3), got {verts.shape}")
        verts = verts[: min(max_frames, len(verts))]
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        frames: list[np.ndarray] = []
        all_pts = verts.reshape(-1, 3)
        y_center = float(np.percentile(all_pts[:, 1], 54))
        span = np.ptp(all_pts, axis=0)
        radius = max(float(span[0]), float(span[1]), float(span[2]), 1.9)
        dist = max(3.0, radius * 1.45)
        materials = [
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.15, 0.42, 0.92, 1.0), metallicFactor=0.03, roughnessFactor=0.58
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.95, 0.31, 0.25, 1.0), metallicFactor=0.03, roughnessFactor=0.58
            ),
        ]
        floor_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.88, 0.90, 0.93, 1.0), metallicFactor=0.0, roughnessFactor=1.0
        )
        try:
            for frame_verts in verts:
                center = frame_verts.reshape(-1, 3).mean(axis=0)
                target = np.array([center[0], y_center, center[2]], dtype=np.float32)
                eye = target + np.array([0.82 * dist, 0.42 * dist, 1.0 * dist], dtype=np.float32)
                scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.42, 0.42, 0.46])
                for person in range(2):
                    scene.add(
                        pyrender.Mesh.from_trimesh(
                            trimesh.Trimesh(frame_verts[person], self.faces, process=False),
                            material=materials[person],
                            smooth=True,
                        )
                    )
                floor_size = max(3.6, radius * 1.5)
                floor = trimesh.creation.box(extents=(floor_size, 0.012, floor_size))
                floor.apply_translation([target[0], -0.006, target[2]])
                scene.add(pyrender.Mesh.from_trimesh(floor, material=floor_mat, smooth=False))
                scene.add(
                    pyrender.PerspectiveCamera(yfov=math.radians(38), aspectRatio=self.width / self.height),
                    pose=self._look_at(eye, target),
                )
                for light_eye, intensity in [
                    (target + np.array([2.5, 4.0, 3.0], dtype=np.float32), 3.8),
                    (target + np.array([-3.0, 2.5, 1.2], dtype=np.float32), 1.8),
                    (target + np.array([0.0, 3.2, -3.0], dtype=np.float32), 1.4),
                ]:
                    scene.add(
                        pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                        pose=self._look_at(light_eye, target),
                    )
                color, _ = renderer.render(scene)
                frames.append(color)
        finally:
            renderer.delete()
        if not frames:
            raise RuntimeError("render produced zero frames")
        return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_orient, body_pose, transl, meta = load_motion(Path(args.input))
    renderer = SMPLRenderer(Path(args.model_dir), args.device, args.width, args.height)
    frames = renderer.render(renderer.vertices(global_orient, body_pose, transl), args.fps, args.max_frames)
    mp4 = out_dir / f"{args.name}.mp4"
    gif = out_dir / f"{args.name}_1024_30fps.gif"
    imageio.mimwrite(mp4, frames, fps=args.fps, quality=8, macro_block_size=1)
    imageio.mimwrite(gif, frames, fps=args.fps, loop=0)
    meta.update(
        {
            "mp4": str(mp4),
            "gif": str(gif),
            "frames": len(frames),
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
        }
    )
    (out_dir / f"{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
