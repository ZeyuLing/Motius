#!/usr/bin/env python3
"""Retarget sequential HML263 predictions to canonical SMPL-22 joints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import canonicalize_smpl22_joints
from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
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
    parser.add_argument("--refine-iters", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument(
        "--max-fit-mpjpe-mm",
        type=float,
        default=50.0,
        help="Reject a clip when its mean SMPL joint-fit error exceeds this value.",
    )
    parser.add_argument(
        "--rotation-init",
        default="hml263_end_effectors",
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


def materialize_case(
    source_path: Path,
    smpl_path: Path,
    joints_path: Path,
    *,
    smpl_rest,
    expected_frames: int,
    device: str,
    source_fps: float,
    target_fps: float,
    refine_iters: int,
    refine_lr: float,
    rotation_init: str,
    max_fit_mpjpe_mm: float = 50.0,
) -> dict[str, float]:
    features = np.asarray(np.load(source_path), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 263:
        raise ValueError(f"Expected [T, 263] in {source_path}, got {features.shape}")
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

    smpl_path.parent.mkdir(parents=True, exist_ok=True)
    joints_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        smpl_path,
        global_orient=np.asarray(converted["global_orient"], dtype=np.float32),
        body_pose=np.asarray(converted["body_pose"], dtype=np.float32),
        transl=np.asarray(converted["transl"], dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(target_fps, dtype=np.float32),
        fit_mpjpe_mm=np.asarray(converted["fit_mpjpe_mm"], dtype=np.float32),
        rotation_init=np.asarray(str(converted["rotation_init"])),
        source_hml263=np.asarray(str(source_path)),
    )
    np.save(joints_path, joints.reshape(expected_frames, 66).astype(np.float32))
    return {
        "fit_mpjpe_mm_mean": mean_error,
        "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
        "fit_mpjpe_mm_max": float(errors.max()),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    hml263_dir = args.hml263_dir.resolve()
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
    cases = manifest.get("cases", [])
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
        if (
            args.skip_existing
            and not args.overwrite
            and smpl_path.is_file()
            and joints_path.is_file()
        ):
            skipped += 1
            continue
        try:
            stats = materialize_case(
                source_path,
                smpl_path,
                joints_path,
                smpl_rest=smpl_rest,
                expected_frames=int(case["total_frames"]),
                device=args.device,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                refine_iters=args.refine_iters,
                refine_lr=args.refine_lr,
                rotation_init=args.rotation_init,
                max_fit_mpjpe_mm=args.max_fit_mpjpe_mm,
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
        "protocol": manifest.get("protocol"),
        "source_representation": "HumanML3D-263",
        "target_representation": "SMPL-22 joints66",
        "rotation_init": args.rotation_init,
        "refine_iters": args.refine_iters,
        "max_fit_mpjpe_mm": args.max_fit_mpjpe_mm,
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
