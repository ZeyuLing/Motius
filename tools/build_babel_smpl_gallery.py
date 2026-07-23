#!/usr/bin/env python3
"""Build the complete BABEL sequential-generation SMPL mesh comparison page."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from motius.motion.retarget._hml263_smpl_impl import (
    estimate_local_rotations,
    matrix_to_rot6d_rowmajor,
)
from motius.motion.retarget.hml263_smpl import load_smpl_rest
from smpl_gallery_assets import encode_motion135, write_chunked_manifest


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    accent: str


METHODS = (
    Method("gt", "GT", "#956000"),
    Method("flowmdm", "FlowMDM", "#315f9d"),
    Method("motionstreamer", "MotionStreamer", "#a5412e"),
    Method("motionlab", "MotionLab", "#287147"),
    Method("prism", "PRISM (epoch 26)", "#6d4ea2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--smpl-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-base-url", required=True)
    parser.add_argument("--body-model-url", default="smpl_model/")
    parser.add_argument(
        "--prism-directory",
        default="prism_epoch26_fixed360_cfg5_ar5_h20x8_resched",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--io-workers", type=int, default=32)
    return parser.parse_args()


def smpl_params_to_motion135(path: Path, max_frames: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        global_orient = np.asarray(source["global_orient"], dtype=np.float32).reshape(-1, 1, 3)
        body_pose = np.asarray(source["body_pose"], dtype=np.float32).reshape(len(global_orient), -1, 3)
        translation_key = "transl" if "transl" in source.files else "trans"
        translation = np.asarray(source[translation_key], dtype=np.float32).reshape(-1, 3)
    if min(len(global_orient), len(body_pose), len(translation)) < max_frames:
        raise ValueError(f"Short SMPL sequence in {path}: expected {max_frames} frames")
    axis_angle = np.concatenate((global_orient, body_pose[:, :21]), axis=1)[:max_frames]
    rotations = axis_angle_to_matrix(axis_angle.reshape(-1, 3)).reshape(len(axis_angle), 22, 3, 3)
    rotation6d = matrix_to_rotation_6d(rotations, convention="row").reshape(len(axis_angle), 132)
    motion = np.concatenate((translation[:len(axis_angle)], rotation6d), axis=1).astype(np.float32)
    if not np.isfinite(motion).all():
        raise ValueError(f"Non-finite SMPL parameters in {path}")
    return motion


def fit_joints(
    path: Path,
    max_frames: int,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Fit native SMPL-22 joints with the shared refine=0 position IK."""

    joints = np.asarray(np.load(path), dtype=np.float32).reshape(-1, 22, 3)
    if len(joints) < max_frames:
        raise ValueError(f"Short joints66 sequence in {path}: {len(joints)} < {max_frames}")
    joints = joints[:max_frames]
    if not np.isfinite(joints).all():
        raise ValueError(f"Non-finite joints66 values in {path}")
    local_rotations = estimate_local_rotations(joints, rest_joints, parents)
    translation = joints[:, 0] - rest_joints[0]
    rotation6d = matrix_to_rot6d_rowmajor(local_rotations).reshape(len(joints), 132)
    return np.concatenate((translation, rotation6d), axis=1).astype(np.float32)


def canonicalize_motion135(motion: np.ndarray) -> np.ndarray:
    """Place frame zero at the origin and align its SMPL forward axis to +Z."""

    value = np.asarray(motion, dtype=np.float32).copy()
    root_rotation = rotation_6d_to_matrix(
        value[:, 3:9], convention="row"
    ).astype(np.float32)
    forward = root_rotation[0] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    yaw = float(np.arctan2(forward[0], forward[2]))
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    canonicalizer = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )
    translation = value[:, :3]
    translation[:, 0] -= translation[0, 0]
    translation[:, 2] -= translation[0, 2]
    value[:, :3] = translation @ canonicalizer.T
    value[:, 3:9] = matrix_to_rotation_6d(
        canonicalizer[None] @ root_rotation, convention="row"
    )
    return value


def display_segments(segments: object) -> list[dict] | None:
    if not isinstance(segments, list):
        return None
    return [
        {
            "caption": str(segment["caption"]),
            "start_frame": int(segment["start_frame"]),
            "end_frame": int(segment["end_frame"]),
        }
        for segment in segments
    ]


def load_method_motion(
    root: Path,
    method: str,
    case_id: str,
    max_frames: int,
    prism_directory: str,
) -> np.ndarray:
    if method == "motionlab":
        path = root / "motionlab_f5_actiongroups_v4_smplfit" / "smpl" / f"{case_id}.npz"
        return smpl_params_to_motion135(path, max_frames)
    if method == "prism":
        path = root / prism_directory / "smplx" / f"{case_id}.npz"
        return smpl_params_to_motion135(path, max_frames)
    raise KeyError(method)


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    source = json.loads(args.source_manifest.expanduser().resolve().read_text())
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("leaderboard_smpl_gallery.html"), output / "index.html")
    _, rest_joints, parents = load_smpl_rest(
        args.smpl_model.expanduser().resolve(), "cpu", gender="neutral"
    )

    cases = [{
        "case_id": str(item["case_id"]),
        "sample_id": str(item.get("sample_id") or item["case_id"]),
        "segments": display_segments(item.get("segments")),
        "references": item.get("references"),
        "motions": {},
        "_frames": int(
            item.get("total_frames")
            or next(iter(item.get("motions", {}).values()))["display_frames"]
        ),
    } for item in source["cases"]]
    stride = max(1, args.stride)
    chunk_size = max(1, args.chunk_size)
    with (
        ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as executor,
        ProcessPoolExecutor(max_workers=max(1, args.io_workers)) as ik_executor,
    ):
        for start in range(0, len(cases), chunk_size):
            end = min(start + chunk_size, len(cases))
            chunk = cases[start:end]
            gt_loaded = ik_executor.map(
                fit_joints,
                [root / "references" / "joints66" / f"{item['case_id']}.npy" for item in chunk],
                [item["_frames"] for item in chunk],
                [rest_joints] * len(chunk),
                [parents] * len(chunk),
            )
            for method in METHODS:
                if method.key == "gt":
                    loaded = gt_loaded
                elif method.key in {"flowmdm", "motionstreamer"}:
                    directory = (
                        root / "flowmdm_seed42" / "joints66"
                        if method.key == "flowmdm"
                        else root / "motionstreamer_latest_seed42" / "joints66"
                    )
                    loaded = ik_executor.map(
                        fit_joints,
                        [directory / f"{item['case_id']}.npy" for item in chunk],
                        [item["_frames"] for item in chunk],
                        [rest_joints] * len(chunk),
                        [parents] * len(chunk),
                    )
                else:
                    loaded = executor.map(
                        lambda item, method=method: load_method_motion(
                            root,
                            method.key,
                            item["case_id"],
                            item["_frames"],
                            args.prism_directory,
                        ),
                        chunk,
                    )
                payload = bytearray()
                asset_name = f"{method.key}_{start // chunk_size:03d}.smpl"
                for item, motion in zip(chunk, loaded):
                    motion = canonicalize_motion135(motion)
                    encoded, descriptor = encode_motion135(motion, stride=stride)
                    byte_offset = len(payload)
                    descriptor.update({
                        "asset": f"assets/{asset_name}",
                        "translation_offset": byte_offset,
                        "rotation_offset": byte_offset + descriptor["translation_count"] * 2,
                        "fps": 30.0,
                    })
                    item["motions"][method.key] = descriptor
                    payload.extend(encoded)
                (assets / asset_name).write_bytes(payload)
            print(f"exported {end}/{len(cases)} sequential cases", flush=True)

    manifest = {
        "schema_version": 2,
        "representation": "smpl_motion135",
        "task": "sequential_motion_generation",
        "title": "Sequential Text-to-Motion · BABEL: SMPL Mesh Comparison",
        "protocol": source.get("protocol"),
        "provenance": {
            "flowmdm_visualization": "native joints66 fitted to the shared SMPL-22 body",
            "motionstreamer_visualization": "native joints66 fitted to the shared SMPL-22 body",
            "prism_checkpoint": args.prism_directory,
        },
        "population": len(cases),
        "asset_base_url": args.asset_base_url,
        "body_model_url": args.body_model_url,
        "reference_label": "Captioned subclips",
        "motion_methods": [method.__dict__ for method in METHODS],
        "cases": cases,
    }
    for item in cases:
        item.pop("_frames")
    write_chunked_manifest(
        output,
        manifest,
        chunk_size=chunk_size,
    )
    print(json.dumps({"output": str(output), "cases": len(cases), "methods": len(METHODS)}, indent=2))


if __name__ == "__main__":
    main()
