#!/usr/bin/env python3
"""Materialize canonical joints66 from SMPL-family BABEL predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import canonicalize_smpl22_joints, smpl_to_joints


SUPPORTED_PROTOCOLS = {
    "babel-official-val-shortmerge30-llm-joints66-actiongroups-v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smplx-model", required=True, type=Path)
    parser.add_argument(
        "--model-type", choices=("smpl", "smplh", "smplx"), default="smplx"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scalar_string(value, default: str) -> str:
    array = np.asarray(value)
    if not array.size:
        return default
    return str(array.item() if array.shape == () else array.reshape(-1)[0])


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards.")
    manifest = json.loads(args.manifest.resolve().read_text())
    if manifest.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol {manifest.get('protocol')!r}.")
    predictions_dir = args.predictions_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.smplx_model.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_ids = set(args.ids)
    generated = skipped = missing = 0
    for index, case in enumerate(manifest.get("cases", [])):
        if index % args.num_shards != args.shard_index:
            continue
        case_id = str(case["case_id"])
        if requested_ids and case_id not in requested_ids:
            continue
        source_path = predictions_dir / f"{case_id}.npz"
        output_path = output_dir / f"{case_id}.npy"
        if not source_path.is_file():
            if args.skip_missing:
                missing += 1
                continue
            raise FileNotFoundError(source_path)
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        with np.load(source_path, allow_pickle=False) as source:
            global_orient = np.asarray(source["global_orient"], dtype=np.float32)
            body_pose = np.asarray(source["body_pose"], dtype=np.float32)
            transl_key = "transl" if "transl" in source else "trans"
            transl = np.asarray(source[transl_key], dtype=np.float32)
            betas = np.asarray(source["betas"], dtype=np.float32) if "betas" in source else None
            gender = _scalar_string(source["gender"], "neutral") if "gender" in source else "neutral"
        expected = int(case["total_frames"])
        if not (len(global_orient) == len(body_pose) == len(transl) == expected):
            raise ValueError(
                f"{case_id} has SMPL-X lengths "
                f"{len(global_orient)}/{len(body_pose)}/{len(transl)}, expected {expected}."
            )
        joints = smpl_to_joints(
            global_orient,
            body_pose,
            transl,
            betas=betas,
            gender=gender,
            model_type=args.model_type,
            model_path=model_path,
        )
        joints = canonicalize_smpl22_joints(joints).reshape(expected, 66)
        np.save(output_path, joints.astype(np.float32))
        generated += 1
        print(f"[{index + 1}/{len(manifest['cases'])}] materialized {case_id}", flush=True)
    print(json.dumps({"generated": generated, "skipped": skipped, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
