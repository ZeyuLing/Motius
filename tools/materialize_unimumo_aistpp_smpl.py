#!/usr/bin/env python3
"""Materialize quality-gated SMPL previews from UniMuMo AIST++ outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.retarget.hml263_smpl import (
    load_smpl_rest,
    retarget_hml263_clip,
    validate_smpl_motion_integrity,
)
from motius.motion.retarget._hml263_smpl_impl import (
    estimate_local_rotations,
    matrix_to_rot6d_rowmajor,
    recover_from_ric,
    resample_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "body_models" / "smpl",
    )
    parser.add_argument(
        "--web-rig-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "leaderboards"
            / "hf_space_music_to_dance"
            / "cases"
            / "smpl_model"
        ),
        help="Public SMPL web rig used by --web-rig-only.",
    )
    parser.add_argument(
        "--web-rig-only",
        action="store_true",
        help="Fit rotations to the public web rig without licensed SMPL parameters.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-fit-mpjpe-mm", type=float, default=50.0)
    parser.add_argument("--max-rotation-jump-p99-deg", type=float, default=90.0)
    parser.add_argument("--max-mesh-edge-ratio-p99", type=float, default=2.0)
    parser.add_argument("--min-mesh-edge-ratio-p01", type=float, default=0.2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def materialize_case(source: Path, destination: Path, *, args, smpl_rest) -> dict:
    with np.load(source, allow_pickle=False) as payload:
        motion_key = (
            "input_humanml3d_263"
            if "input_humanml3d_263" in payload.files
            else "humanml3d_263"
        )
        hml263 = np.asarray(payload[motion_key], dtype=np.float32)
        source_fps = (
            float(np.asarray(payload["fps"]).item())
            if "fps" in payload.files
            else 60.0
        )
    converted = retarget_hml263_clip(
        hml263,
        smpl_rest=smpl_rest,
        device=args.device,
        source_fps=source_fps,
        target_fps=args.target_fps,
        gender="neutral",
        floor_align=True,
        refine_iters=0,
        rotation_init="position_ik",
        temporal_twist_stabilization=True,
        compute_mesh_metrics=True,
    )
    errors = np.asarray(converted["fit_mpjpe_mm"], dtype=np.float32)
    mean_error = float(errors.mean())
    if not np.isfinite(errors).all() or mean_error > args.max_fit_mpjpe_mm:
        raise RuntimeError(
            f"SMPL fit MPJPE {mean_error:.2f} mm exceeds "
            f"{args.max_fit_mpjpe_mm:.2f} mm for {source}"
        )
    integrity = dict(converted["mesh_integrity"])
    validate_smpl_motion_integrity(
        integrity,
        max_rotation_jump_p99_deg=args.max_rotation_jump_p99_deg,
        max_mesh_edge_ratio_p99=args.max_mesh_edge_ratio_p99,
        min_mesh_edge_ratio_p01=args.min_mesh_edge_ratio_p01,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}-{time.time_ns()}.tmp.npz"
    )
    try:
        np.savez_compressed(
            temporary,
            motion_135=np.asarray(converted["motion_135"], dtype=np.float32),
            global_orient=np.asarray(converted["global_orient"], dtype=np.float32),
            body_pose=np.asarray(converted["body_pose"], dtype=np.float32),
            transl=np.asarray(converted["transl"], dtype=np.float32),
            target_joints=np.asarray(converted["target_joints"], dtype=np.float32),
            fitted_joints=np.asarray(converted["fitted_joints"], dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_framerate=np.asarray(args.target_fps, dtype=np.float32),
            fit_mpjpe_mm=errors,
            rotation_jump_deg_p99=np.asarray(
                integrity["rotation_jump_deg_p99"], dtype=np.float32
            ),
            rotation_jump_deg_max=np.asarray(
                integrity["rotation_jump_deg_max"], dtype=np.float32
            ),
            mesh_edge_ratio_p01=np.asarray(
                integrity["mesh_edge_ratio_p01"], dtype=np.float32
            ),
            mesh_edge_ratio_p99=np.asarray(
                integrity["mesh_edge_ratio_p99"], dtype=np.float32
            ),
            mesh_edge_ratio_max=np.asarray(
                integrity["mesh_edge_ratio_max"], dtype=np.float32
            ),
            rotation_init=np.asarray(str(converted["rotation_init"])),
            source_unimumo_hml263=np.asarray(str(source)),
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": len(errors),
        "fit_mpjpe_mm_mean": mean_error,
        "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
        **{key: float(value) for key, value in integrity.items()},
    }


def load_web_rig(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
    joints = np.fromfile(path / "joints.f32", dtype="<f4").reshape(-1, 3)
    parents = np.asarray(metadata["parents"], dtype=np.int64)
    if len(joints) < 22 or len(parents) < 22:
        raise ValueError("SMPL web rig must contain at least 22 joints")
    return joints[:22].astype(np.float32), parents[:22]


def materialize_web_rig_case(
    source: Path,
    destination: Path,
    *,
    args,
    web_rig: tuple[np.ndarray, np.ndarray],
) -> dict:
    with np.load(source, allow_pickle=False) as payload:
        motion_key = (
            "input_humanml3d_263"
            if "input_humanml3d_263" in payload.files
            else "humanml3d_263"
        )
        hml263 = np.asarray(payload[motion_key], dtype=np.float32)
        source_fps = (
            float(np.asarray(payload["fps"]).item())
            if "fps" in payload.files
            else 60.0
        )
    joints, parents = web_rig
    target = resample_linear(
        recover_from_ric(hml263), source_fps, args.target_fps
    )
    target = np.asarray(target, dtype=np.float32)
    target[..., 1] -= target[..., 1].min()
    rotations = estimate_local_rotations(
        target,
        joints,
        parents,
        orientation_mode="bone",
        parent_ref_weight=0.25,
        temporal_twist_stabilization=True,
    )
    translation = target[:, 0] - joints[0]
    motion135 = np.concatenate(
        (
            translation,
            matrix_to_rot6d_rowmajor(rotations).reshape(len(target), 132),
        ),
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(motion135).all():
        raise RuntimeError(f"Web-rig IK produced non-finite values for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}-{time.time_ns()}.tmp.npz"
    )
    try:
        np.savez_compressed(
            temporary,
            motion_135=motion135,
            target_joints=target,
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_framerate=np.asarray(args.target_fps, dtype=np.float32),
            rotation_init=np.asarray("position_ik_public_web_rig"),
            source_unimumo_hml263=np.asarray(str(source)),
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": len(motion135),
        "target_floor_height": float(target[..., 1].min()),
    }


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.expanduser().resolve().glob("*.npz"))
    smpl_rest = None
    web_rig = None
    if args.web_rig_only:
        web_rig = load_web_rig(args.web_rig_dir.expanduser().resolve())
    else:
        smpl_rest = load_smpl_rest(
            args.smpl_model_dir.expanduser(), args.device, gender="neutral"
        )
    records = []
    failed = 0
    for index, source in enumerate(sources):
        if index % args.num_shards != args.shard_index:
            continue
        destination = args.output_dir.expanduser().resolve() / source.name
        if destination.is_file() and not args.overwrite:
            records.append({"case_id": source.stem, "status": "skipped"})
            continue
        try:
            if web_rig is not None:
                stats = materialize_web_rig_case(
                    source, destination, args=args, web_rig=web_rig
                )
            else:
                stats = materialize_case(
                    source, destination, args=args, smpl_rest=smpl_rest
                )
            record = {"case_id": source.stem, "status": "generated", **stats}
        except Exception as exc:  # noqa: BLE001
            failed += 1
            record = {"case_id": source.stem, "status": "failed", "error": str(exc)}
        records.append(record)
        print(json.dumps(record), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "source_representation": "UniMuMo HumanML3D-263",
        "target_representation": "neutral SMPL motion135",
        "source_fps": 60.0,
        "target_fps": args.target_fps,
        "rotation_init": (
            "position_ik_public_web_rig"
            if args.web_rig_only
            else "position_ik_temporal_twist_stabilized"
        ),
        "refine_iters": 0,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": records,
    }
    summary_path = (
        args.output_dir / f"materialization_shard_{args.shard_index:02d}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
