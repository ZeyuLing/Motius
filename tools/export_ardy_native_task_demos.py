#!/usr/bin/env python3
"""Export truthful ARDY task demos in the checkpoint's native body mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.pipelines.ardy import ARDYPipeline  # noqa: E402
from tools.export_ardy_core_demo import core_lbs_vertices  # noqa: E402


def _load_text_features(path: Path):
    payload = np.load(path)
    return (
        str(payload["caption"]),
        payload["text_feat"],
        payload["text_pad_mask"],
    )


def _native_arrays(output: dict, frames: int):
    rotations = np.asarray(output["local_rot_mats"])[0, :frames]
    roots = np.asarray(output["root_positions"])[0, :frames]
    joints = np.asarray(output["posed_joints"])[0, :frames]
    return rotations, roots, joints


def _save_native(
    output: Path,
    rotations: np.ndarray,
    roots: np.ndarray,
    joints: np.ndarray,
    caption: str,
    fps: int,
    device: str,
    **extra,
) -> None:
    vertices, faces = core_lbs_vertices(rotations, roots, device=device)
    np.savez_compressed(
        output,
        core_joints=joints.astype(np.float32),
        core_vertices=vertices.astype(np.float32),
        core_faces=faces.astype(np.uint32),
        caption=caption,
        fps=fps,
        **extra,
    )


def export(args: argparse.Namespace) -> dict:
    caption, text_feat, text_mask = _load_text_features(args.text_features)
    pipe = ARDYPipeline.from_pretrained(
        str(args.checkpoint),
        bundle_kwargs={
            "device": args.device,
            "text_encoder": False,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sequential = pipe.infer_sequential_text_to_motion(
        [caption, caption],
        text_feat=text_feat,
        text_pad_mask=text_mask,
        num_denoising_steps=args.steps,
        seed=args.seed,
        return_numpy=True,
    )
    rotations = []
    roots = []
    joints = []
    segment_frames = []
    cursor = 0
    for segment in sequential["segments"]:
        current = len(segment["root_positions"][0])
        segment_frames.append([cursor, cursor + current])
        cursor += current
        rotation, root, joint = _native_arrays(segment, current)
        rotations.append(rotation)
        roots.append(root)
        joints.append(joint)
    sequential_path = args.output_dir / "ardy_sequential_native.npz"
    _save_native(
        sequential_path,
        np.concatenate(rotations),
        np.concatenate(roots),
        np.concatenate(joints),
        caption,
        int(pipe.fps),
        args.device,
        segment_frames=np.asarray(segment_frames, dtype=np.int32),
    )

    length = 80
    frame_indices = np.asarray([0, 20, 40, 60, 79], dtype=np.int64)
    root_2d = np.asarray(
        [
            [0.0, 0.0],
            [0.35, 0.15],
            [0.75, 0.35],
            [1.20, 0.30],
            [1.55, 0.0],
        ],
        dtype=np.float32,
    )
    heading = np.asarray([0.0, 0.15, 0.30, 0.12, 0.0], dtype=np.float32)
    constraint = pipe.root2d_constraint(
        frame_indices,
        root_2d,
        heading,
        device=pipe.device,
    )
    controlled = pipe.infer_kinematic_motion_control(
        [caption],
        [length],
        constraints=[constraint],
        text_feat=text_feat,
        text_pad_mask=text_mask,
        num_denoising_steps=args.steps,
        seed=args.seed + 1,
        return_numpy=True,
    )
    rotation, root, joint = _native_arrays(controlled, length)
    control_path = args.output_dir / "ardy_kinematic_native.npz"
    _save_native(
        control_path,
        rotation,
        root,
        joint,
        caption,
        int(pipe.fps),
        args.device,
        condition_frame_indices=frame_indices,
        condition_root_2d=root_2d,
        condition_heading=heading,
    )

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "caption": caption,
        "fps": int(pipe.fps),
        "text_features": str(args.text_features),
        "sequential": str(sequential_path),
        "kinematic": str(control_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))
