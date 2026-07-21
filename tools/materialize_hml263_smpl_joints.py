#!/usr/bin/env python3
"""Retarget sequential HML263 predictions to canonical SMPL-22 joints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import canonicalize_smpl22_joints
from motius.motion.retarget.hml263_smpl import (
    load_smpl_rest,
    retarget_hml263_clip,
    validate_smpl_motion_integrity,
)


def _valid_materialization(smpl_path: Path, joints_path: Path) -> bool:
    """Return true only for a complete, readable SMPL/joints output pair."""

    try:
        with np.load(smpl_path, allow_pickle=False) as payload:
            required = (
                "motion_135",
                "global_orient",
                "body_pose",
                "transl",
                "fit_mpjpe_mm",
                "rotation_jump_deg_p99",
                "mesh_edge_ratio_p99",
            )
            arrays = {key: np.asarray(payload[key]) for key in required}
        frames = len(arrays["transl"])
        if frames < 2 or arrays["motion_135"].shape != (frames, 135):
            return False
        if arrays["global_orient"].shape != (frames, 3):
            return False
        if arrays["body_pose"].shape != (frames, 63):
            return False
        if arrays["fit_mpjpe_mm"].size != frames:
            return False
        if not all(np.isfinite(value).all() for value in arrays.values()):
            return False
        joints = np.asarray(np.load(joints_path, allow_pickle=False))
        return joints.shape == (frames, 66) and bool(np.isfinite(joints).all())
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument(
        "--ids-file",
        type=Path,
        help=(
            "Newline-delimited motion ids. Output lengths are derived from each "
            "HML263 clip and the source/target frame rates."
        ),
    )
    parser.add_argument("--hml263-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "body_models" / "smpl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument(
        "--max-fit-mpjpe-mm",
        type=float,
        default=50.0,
        help="Reject a clip when its mean SMPL joint-fit error exceeds this value.",
    )
    parser.add_argument("--max-rotation-jump-p99-deg", type=float, default=90.0)
    parser.add_argument("--max-mesh-edge-ratio-p99", type=float, default=2.0)
    parser.add_argument("--min-mesh-edge-ratio-p01", type=float, default=0.2)
    parser.add_argument(
        "--rotation-init",
        default="position_ik",
        choices=("hml263_end_effectors", "position_ik", "hml263_init"),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ids", default="", help="Comma-separated case ids.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def _load_cases(args: argparse.Namespace, hml263_dir: Path) -> tuple[str, list[dict]]:
    if args.manifest is not None:
        manifest = json.loads(args.manifest.resolve().read_text())
        return str(manifest.get("protocol") or args.manifest.stem), list(
            manifest.get("cases", [])
        )

    case_ids = [
        line.strip()
        for line in args.ids_file.resolve().read_text().splitlines()
        if line.strip()
    ]
    cases = []
    for index, case_id in enumerate(case_ids):
        if index % args.num_shards != args.shard_index:
            cases.append({"case_id": case_id, "total_frames": 0})
            continue
        # The source is loaded once inside materialize_case. Deferring the
        # derived target length avoids a separate small-file read for every
        # item before GPU work can begin.
        cases.append({"case_id": case_id, "total_frames": None})
    return f"ids:{args.ids_file.resolve()}", cases


def materialize_case(
    source_path: Path,
    smpl_path: Path,
    joints_path: Path,
    *,
    smpl_rest,
    expected_frames: Optional[int],
    device: str,
    source_fps: float,
    target_fps: float,
    refine_iters: int,
    refine_lr: float,
    rotation_init: str,
    max_fit_mpjpe_mm: float = 50.0,
    max_rotation_jump_p99_deg: float = 90.0,
    max_mesh_edge_ratio_p99: float = 2.0,
    min_mesh_edge_ratio_p01: float = 0.2,
) -> dict[str, float]:
    features = np.asarray(np.load(source_path), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 263:
        raise ValueError(f"Expected [T, 263] in {source_path}, got {features.shape}")
    if expected_frames is None:
        expected_frames = max(
            2, int(round(len(features) * target_fps / source_fps))
        )
    converted = retarget_hml263_clip(
        features,
        smpl_rest=smpl_rest,
        device=device,
        gender="neutral",
        source_fps=source_fps,
        target_fps=target_fps,
        target_len=expected_frames,
        floor_align=True,
        refine_iters=refine_iters,
        refine_lr=refine_lr,
        rotation_init=rotation_init,
        compute_mesh_metrics=True,
    )
    joints = canonicalize_smpl22_joints(converted["fitted_joints"])
    if joints.shape != (expected_frames, 22, 3):
        raise RuntimeError(
            f"Retargeted {source_path} to {joints.shape}, expected "
            f"({expected_frames}, 22, 3)"
        )
    if not np.isfinite(joints).all():
        raise RuntimeError(f"Retargeted joints contain non-finite values: {source_path}")

    errors = np.asarray(converted["fit_mpjpe_mm"], dtype=np.float64)
    if errors.size != expected_frames or not np.isfinite(errors).all():
        raise RuntimeError(
            f"Invalid per-frame SMPL fit errors for {source_path}: "
            f"shape={errors.shape}, finite={bool(np.isfinite(errors).all())}"
        )
    mean_error = float(errors.mean())
    if mean_error > float(max_fit_mpjpe_mm):
        raise RuntimeError(
            f"SMPL fit MPJPE {mean_error:.2f} mm exceeds "
            f"the {float(max_fit_mpjpe_mm):.2f} mm quality gate: {source_path}"
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
        raise RuntimeError(f"SMPL mesh integrity gate failed for {source_path}: {exc}") from exc

    smpl_path.parent.mkdir(parents=True, exist_ok=True)
    joints_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    smpl_tmp = smpl_path.with_name(f".{smpl_path.stem}.{token}.tmp.npz")
    joints_tmp = joints_path.with_name(f".{joints_path.stem}.{token}.tmp.npy")
    try:
        np.savez_compressed(
            smpl_tmp,
            motion_135=np.asarray(converted["motion_135"], dtype=np.float32),
            global_orient=np.asarray(converted["global_orient"], dtype=np.float32),
            body_pose=np.asarray(converted["body_pose"], dtype=np.float32),
            transl=np.asarray(converted["transl"], dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_framerate=np.asarray(target_fps, dtype=np.float32),
            fit_mpjpe_mm=np.asarray(converted["fit_mpjpe_mm"], dtype=np.float32),
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
            source_hml263=np.asarray(str(source_path)),
        )
        np.save(joints_tmp, joints.reshape(expected_frames, 66).astype(np.float32))
        os.replace(smpl_tmp, smpl_path)
        os.replace(joints_tmp, joints_path)
    finally:
        smpl_tmp.unlink(missing_ok=True)
        joints_tmp.unlink(missing_ok=True)
    return {
        "fit_mpjpe_mm_mean": mean_error,
        "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
        "fit_mpjpe_mm_max": float(errors.max()),
        **{key: float(value) for key, value in integrity.items()},
    }


def main() -> None:
    args = parse_args()
    hml263_dir = args.hml263_dir.resolve()
    protocol, cases = _load_cases(args, hml263_dir)
    output_dir = args.output_dir.resolve()
    smpl_dir = output_dir / "smpl"
    joints_dir = output_dir / "joints66"
    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    # Keep the caller-visible mount path. Resolving an absolute symlink can
    # produce a host-only CEPH alias that is not mounted inside GPU containers.
    smpl_rest = load_smpl_rest(
        args.smpl_model_dir.expanduser(), args.device, gender="neutral"
    )

    started = time.time()
    generated = skipped = missing = failed = 0
    fit_means: list[float] = []
    failures: list[dict[str, str]] = []
    for index, case in enumerate(cases):
        if index % args.num_shards != args.shard_index:
            continue
        case_id = str(case["case_id"])
        if selected and case_id not in selected:
            continue
        source_path = hml263_dir / f"{case_id}.npy"
        smpl_path = smpl_dir / f"{case_id}.npz"
        joints_path = joints_dir / f"{case_id}.npy"
        if not source_path.is_file():
            missing += 1
            failures.append({"case_id": case_id, "error": "missing HML263 source"})
            continue
        if args.skip_existing and not args.overwrite and _valid_materialization(
            smpl_path, joints_path
        ):
            skipped += 1
            continue
        try:
            stats = materialize_case(
                source_path,
                smpl_path,
                joints_path,
                smpl_rest=smpl_rest,
                expected_frames=case["total_frames"],
                device=args.device,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                refine_iters=args.refine_iters,
                refine_lr=args.refine_lr,
                rotation_init=args.rotation_init,
                max_fit_mpjpe_mm=args.max_fit_mpjpe_mm,
                max_rotation_jump_p99_deg=args.max_rotation_jump_p99_deg,
                max_mesh_edge_ratio_p99=args.max_mesh_edge_ratio_p99,
                min_mesh_edge_ratio_p01=args.min_mesh_edge_ratio_p01,
            )
            fit_means.append(stats["fit_mpjpe_mm_mean"])
            generated += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"case_id": case_id, "error": str(exc)})
            print(f"[failed] {case_id}: {exc}", flush=True)
        if (generated + skipped + failed) % 10 == 0:
            print(
                f"[progress] shard={args.shard_index}/{args.num_shards} "
                f"generated={generated} skipped={skipped} failed={failed}",
                flush=True,
            )

    summary = {
        "protocol": protocol,
        "source_representation": "HumanML3D-263",
        "target_representation": "SMPL-22 joints66",
        "rotation_init": args.rotation_init,
        "refine_iters": args.refine_iters,
        "max_fit_mpjpe_mm": args.max_fit_mpjpe_mm,
        "max_rotation_jump_p99_deg": args.max_rotation_jump_p99_deg,
        "max_mesh_edge_ratio_p99": args.max_mesh_edge_ratio_p99,
        "min_mesh_edge_ratio_p01": args.min_mesh_edge_ratio_p01,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "generated": generated,
        "skipped": skipped,
        "missing": missing,
        "failed": failed,
        "fit_mpjpe_mm_mean": float(np.mean(fit_means)) if fit_means else None,
        "fit_mpjpe_mm_max_case_mean": float(np.max(fit_means)) if fit_means else None,
        "elapsed_seconds": time.time() - started,
        "failures": failures[:100],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"materialization_shard_{args.shard_index:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
