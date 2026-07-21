#!/usr/bin/env python3
"""Materialize quality-gated SMPL previews from TM2D AIST++ outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.retarget.hml263_smpl import (
    load_smpl_rest,
    retarget_hml263_clip,
    validate_smpl_motion_integrity,
)
from motius.motion.representation.aistpp import aistpp_smpl24_to_smpl22_joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "body_models" / "smpl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-fit-mpjpe-mm", type=float, default=50.0)
    parser.add_argument("--max-rotation-jump-p99-deg", type=float, default=90.0)
    parser.add_argument("--max-mesh-edge-ratio-p99", type=float, default=1.8)
    parser.add_argument("--min-mesh-edge-ratio-p01", type=float, default=0.2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ids", default="", help="Comma-separated AIST++ case IDs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def materialize_case(
    source: Path,
    destination: Path,
    *,
    smpl_rest,
    device: str,
    target_fps: float,
    max_fit_mpjpe_mm: float = 50.0,
    max_rotation_jump_p99_deg: float = 90.0,
    max_mesh_edge_ratio_p99: float = 1.8,
    min_mesh_edge_ratio_p01: float = 0.2,
) -> dict[str, float]:
    with np.load(source, allow_pickle=False) as payload:
        if "joints" not in payload.files:
            raise KeyError(f"joints not found in {source}")
        joints = aistpp_smpl24_to_smpl22_joints(payload["joints"])
        source_fps = float(np.asarray(payload["fps"]).item())
    converted = retarget_hml263_clip(
        joints,
        smpl_rest=smpl_rest,
        device=device,
        source_fps=source_fps,
        target_fps=target_fps,
        gender="neutral",
        floor_align=True,
        refine_iters=0,
        rotation_init="position_ik",
        compute_mesh_metrics=True,
    )
    errors = np.asarray(converted["fit_mpjpe_mm"], dtype=np.float32)
    mean_error = float(errors.mean())
    if not np.isfinite(errors).all() or mean_error > max_fit_mpjpe_mm:
        raise RuntimeError(
            f"SMPL fit MPJPE {mean_error:.2f} mm exceeds "
            f"{max_fit_mpjpe_mm:.2f} mm for {source}"
        )
    integrity = dict(converted.get("mesh_integrity") or {})
    try:
        validate_smpl_motion_integrity(
            integrity,
            max_rotation_jump_p99_deg=max_rotation_jump_p99_deg,
            max_mesh_edge_ratio_p99=max_mesh_edge_ratio_p99,
            min_mesh_edge_ratio_p01=min_mesh_edge_ratio_p01,
        )
    except ValueError as exc:
        raise RuntimeError(f"SMPL mesh integrity gate failed for {source}: {exc}") from exc

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
            mocap_framerate=np.asarray(target_fps, dtype=np.float32),
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
            source_tm2d287=np.asarray(str(source)),
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": float(len(errors)),
        "fit_mpjpe_mm_mean": mean_error,
        "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
        **{key: float(value) for key, value in integrity.items()},
    }


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.expanduser().resolve().glob("*.npz"))
    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    smpl_rest = load_smpl_rest(
        args.smpl_model_dir.expanduser(), args.device, gender="neutral"
    )
    records = []
    failed = 0
    for index, source in enumerate(sources):
        if index % args.num_shards != args.shard_index:
            continue
        if selected and source.stem not in selected:
            continue
        destination = args.output_dir.expanduser().resolve() / f"{source.stem}.npz"
        if destination.is_file() and not args.overwrite:
            records.append({"case_id": source.stem, "status": "skipped"})
            continue
        try:
            stats = materialize_case(
                source,
                destination,
                smpl_rest=smpl_rest,
                device=args.device,
                target_fps=args.target_fps,
                max_fit_mpjpe_mm=args.max_fit_mpjpe_mm,
                max_rotation_jump_p99_deg=args.max_rotation_jump_p99_deg,
                max_mesh_edge_ratio_p99=args.max_mesh_edge_ratio_p99,
                min_mesh_edge_ratio_p01=args.min_mesh_edge_ratio_p01,
            )
            records.append({"case_id": source.stem, "status": "generated", **stats})
            print(json.dumps(records[-1]), flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            records.append({"case_id": source.stem, "status": "failed", "error": str(exc)})
            print(json.dumps(records[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "source_representation": "TM2D HumanML-24 287D joints",
        "target_representation": "neutral SMPL motion135",
        "source_fps": 60.0,
        "target_fps": args.target_fps,
        "rotation_init": "position_ik_temporal_twist_stabilized",
        "refine_iters": 0,
        "quality_gate": {
            "max_fit_mpjpe_mm": args.max_fit_mpjpe_mm,
            "max_rotation_jump_p99_deg": args.max_rotation_jump_p99_deg,
            "max_mesh_edge_ratio_p99": args.max_mesh_edge_ratio_p99,
            "min_mesh_edge_ratio_p01": args.min_mesh_edge_ratio_p01,
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": records,
    }
    (args.output_dir / f"materialization_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
