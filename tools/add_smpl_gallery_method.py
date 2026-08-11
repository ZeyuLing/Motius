#!/usr/bin/env python3
"""Add one packed motion135 method to a chunked SMPL gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from smpl_gallery_assets import encode_motion135, load_motion135, motion_path


INTEGRITY_FIELDS = (
    "fit_mpjpe_mm",
    "rotation_jump_deg_p99",
    "rotation_jump_deg_max",
    "mesh_edge_ratio_p01",
    "mesh_edge_ratio_p99",
    "mesh_edge_ratio_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--accent", default="#176b48")
    parser.add_argument("--insert-after", default="gt")
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def _scalar_metadata(path: Path) -> dict[str, float]:
    if path.suffix != ".npz":
        return {}
    metadata = {}
    with np.load(path, allow_pickle=False) as payload:
        for field in INTEGRITY_FIELDS:
            if field not in payload.files:
                continue
            value = np.asarray(payload[field], dtype=np.float32)
            if field == "fit_mpjpe_mm":
                metadata["fit_mpjpe_mm_mean"] = float(value.mean())
            elif value.size == 1:
                metadata[field] = float(value.item())
    return metadata


def _descriptor_config(manifest: dict) -> tuple[list[dict], int, str]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain inline case metadata")
    config = manifest.get("case_descriptor_chunks")
    if not isinstance(config, dict):
        raise ValueError("manifest must use case_descriptor_chunks")
    chunk_size = int(config["size"])
    template = str(config["path"])
    if chunk_size < 1:
        raise ValueError("descriptor chunk size must be positive")
    return cases, chunk_size, template


def add_method(
    manifest: dict,
    *,
    manifest_root: Path,
    method_key: str,
    method_label: str,
    accent: str,
    insert_after: str,
    motion_dir: Path,
    output_dir: Path,
    fps: float,
    stride: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict:
    methods = manifest.get("motion_methods")
    if not isinstance(methods, list):
        raise ValueError("manifest has no motion_methods")
    if any(method.get("key") == method_key for method in methods):
        raise ValueError(f"motion method {method_key!r} already exists")
    entry = {"key": method_key, "label": method_label, "accent": accent}
    insertion = next(
        (
            index + 1
            for index, method in enumerate(methods)
            if method.get("key") == insert_after
        ),
        len(methods),
    )
    methods.insert(insertion, entry)

    cases, chunk_size, template = _descriptor_config(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_count = 0
    case_count = 0
    fit_values = []
    for start in range(0, len(cases), chunk_size):
        chunk_index = start // chunk_size
        if chunk_index % num_shards != shard_index:
            continue
        relative = Path(template.format(chunk=f"{chunk_index:03d}"))
        payload = json.loads((manifest_root / relative).read_text(encoding="utf-8"))
        if int(payload.get("start", -1)) != start:
            raise ValueError(f"descriptor chunk {relative} has the wrong start")
        records = payload.get("motions")
        expected = cases[start : start + chunk_size]
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError(f"descriptor chunk {relative} has the wrong motion count")

        asset = f"assets/{method_key}_{chunk_index:03d}.smpl"
        packed = bytearray()
        for case, motions in zip(expected, records):
            case_id = str(case.get("case_id") or case.get("sample_id"))
            source = motion_path(motion_dir, case_id)
            motion = load_motion135(source)
            encoded, descriptor = encode_motion135(motion, stride=max(1, stride))
            metadata = _scalar_metadata(source)
            descriptor.update(
                {
                    "asset": asset,
                    "translation_offset": len(packed),
                    "rotation_offset": len(packed)
                    + descriptor["translation_count"] * 2,
                    "fps": float(fps),
                    **metadata,
                }
            )
            motions[method_key] = descriptor
            packed.extend(encoded)
            if "fit_mpjpe_mm_mean" in metadata:
                fit_values.append(metadata["fit_mpjpe_mm_mean"])

        destination = output_dir / asset
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(packed)
        asset_count += 1
        case_count += len(records)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    if shard_index == 0:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return {
        "method": method_key,
        "cases": case_count,
        "assets": asset_count,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "fit_mpjpe_mm_mean": float(np.mean(fit_values)) if fit_values else None,
        "output": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = add_method(
        manifest,
        manifest_root=manifest_path.parent,
        method_key=args.method_key,
        method_label=args.method_label,
        accent=args.accent,
        insert_after=args.insert_after,
        motion_dir=args.motion_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        fps=args.fps,
        stride=args.stride,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
