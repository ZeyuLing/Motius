#!/usr/bin/env python3
"""Replace one method's packed motion135 assets in an existing gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


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


def replace_method_assets(
    manifest: dict,
    *,
    method_key: str,
    motion_dir: Path,
    output_dir: Path,
    descriptor_root: Path | None = None,
) -> dict:
    methods = {method["key"] for method in manifest.get("motion_methods", [])}
    if method_key not in methods:
        raise KeyError(f"motion method {method_key!r} is not declared in manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain inline cases")

    motion_records: list[dict]
    descriptor_chunks: list[tuple[Path, dict]] = []
    chunk_config = manifest.get("case_descriptor_chunks")
    if chunk_config:
        if descriptor_root is None:
            raise ValueError("descriptor_root is required for a chunked manifest")
        chunk_size = int(chunk_config["size"])
        path_template = str(chunk_config["path"])
        motion_records = []
        for start in range(0, len(cases), chunk_size):
            relative = Path(
                path_template.format(chunk=f"{start // chunk_size:03d}")
            )
            payload = json.loads((descriptor_root / relative).read_text())
            if int(payload.get("start", -1)) != start:
                raise ValueError(f"descriptor chunk {relative} has the wrong start")
            values = payload.get("motions")
            if not isinstance(values, list):
                raise ValueError(f"descriptor chunk {relative} has no motions list")
            motion_records.extend(values)
            descriptor_chunks.append((relative, payload))
        if len(motion_records) != len(cases):
            raise ValueError(
                f"loaded {len(motion_records)} descriptors for {len(cases)} cases"
            )
    else:
        motion_records = [case.get("motions", {}) for case in cases]

    groups: dict[str, list[tuple[dict, dict]]] = {}
    for case, motions in zip(cases, motion_records):
        descriptor = motions.get(method_key)
        if not isinstance(descriptor, dict) or "asset" not in descriptor:
            raise KeyError(f"case {case.get('case_id')!r} has no {method_key!r} motion")
        groups.setdefault(str(descriptor["asset"]), []).append((case, motions))

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for asset, grouped_records in groups.items():
        packed = bytearray()
        for case, motions in grouped_records:
            case_id = str(case.get("case_id") or case.get("sample_id"))
            old = motions[method_key]
            source = motion_path(motion_dir, case_id)
            expected_frames = int(old.get("display_frames") or old["frames"])
            motion = load_motion135(source, max_frames=expected_frames)
            if len(motion) != expected_frames:
                raise ValueError(
                    f"{case_id}: replacement has {len(motion)} frames, "
                    f"expected {expected_frames}"
                )
            encoded, descriptor = encode_motion135(
                motion, stride=max(1, int(old.get("stride", 1)))
            )
            descriptor.update(
                {
                    "asset": asset,
                    "translation_offset": len(packed),
                    "rotation_offset": len(packed)
                    + descriptor["translation_count"] * 2,
                    "fps": float(old.get("fps", 30.0)),
                    **_scalar_metadata(source),
                }
            )
            motions[method_key] = descriptor
            packed.extend(encoded)
        destination = output_dir / asset
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(packed)

    for relative, payload in descriptor_chunks:
        start = int(payload["start"])
        payload["motions"] = motion_records[start : start + len(payload["motions"])]
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "method": method_key,
        "cases": len(cases),
        "assets": len(groups),
        "output": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    summary = replace_method_assets(
        manifest,
        method_key=args.method_key,
        motion_dir=args.motion_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        descriptor_root=args.manifest.expanduser().resolve().parent,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
