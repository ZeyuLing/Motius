#!/usr/bin/env python3
"""Build the public BrokenAMASS motion-repair SMPL comparison gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from tools.smpl_gallery_assets import encode_motion135


Z_UP_TO_Y_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)

METHODS = (
    ("gt", "Clean GT", "#087d72", "motion"),
    ("corrupted", "Corrupted input", "#a5412e", "motion"),
    ("mogendit", "MoGenDiT", "#315f9d", "motion_fix"),
    ("stablemotion", "StableMotion", "#956000", "motion_fix"),
    ("motioncanvas", "MotionCanvas", "#7a4e98", "motion_fix"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--corrupted", required=True, type=Path)
    parser.add_argument("--mogendit", required=True, type=Path)
    parser.add_argument("--stablemotion", required=True, type=Path)
    parser.add_argument("--motioncanvas", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-base-url", required=True)
    parser.add_argument("--descriptor-base-url", required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def load_packed(path: Path) -> dict:
    # NumPy 2 pickles use numpy._core; keep benchmark artifacts readable on
    # the NumPy 1.x runtime shipped by several supported environments.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    value = np.load(path, allow_pickle=True)
    if value.shape != () or value.dtype != object:
        raise ValueError(f"{path} is not a packed motion-repair dictionary")
    result = value.item()
    if not isinstance(result, dict):
        raise TypeError(f"{path} did not contain a dictionary")
    return result


def motion135(record: dict, length: int) -> np.ndarray:
    pose = np.asarray(record["poses"], dtype=np.float32)[:length].reshape(length, -1, 3)
    translation = np.asarray(record["trans"], dtype=np.float32)[:length]
    if pose.shape[1] < 22 or translation.shape != (length, 3):
        raise ValueError(
            f"invalid SMPL record: pose={pose.shape}, translation={translation.shape}"
        )
    rotations = np.asarray(
        axis_angle_to_matrix(pose[:, :22].reshape(-1, 3)),
        dtype=np.float32,
    ).reshape(length, 22, 3, 3)
    rotations = rotations.copy()
    rotations[:, 0] = Z_UP_TO_Y_UP @ rotations[:, 0]
    translation = translation @ Z_UP_TO_Y_UP.T
    rotation6d = np.asarray(
        matrix_to_rotation_6d(rotations, convention="row"),
        dtype=np.float32,
    ).reshape(length, 132)
    return np.ascontiguousarray(
        np.concatenate((translation, rotation6d), axis=-1),
        dtype=np.float32,
    )


def corruption_map(path: Path) -> dict[str, list[str]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        result[str(item["relpath"])] = [
            str(augmentation["type"])
            for augmentation in item.get("recipe", {}).get("augmentations", [])
        ]
    return result


def build(args: argparse.Namespace) -> dict:
    packed = {
        "gt": load_packed(args.gt),
        "corrupted": load_packed(args.corrupted),
        "mogendit": load_packed(args.mogendit),
        "stablemotion": load_packed(args.stablemotion),
        "motioncanvas": load_packed(args.motioncanvas),
    }
    reference = packed["gt"]
    case_keys = [str(value) for value in reference["case_keys"]]
    lengths = np.asarray(reference["lengths"], dtype=np.int32)
    if len(case_keys) != 299 or lengths.shape != (299,):
        raise ValueError(
            f"expected 299 paired cases, got {len(case_keys)} keys and {lengths.shape}"
        )
    for method, value in packed.items():
        if [str(item) for item in value["case_keys"]] != case_keys:
            raise ValueError(f"{method} case order does not match Clean GT")
        if not np.array_equal(np.asarray(value["lengths"], dtype=np.int32), lengths):
            raise ValueError(f"{method} lengths do not match Clean GT")

    corruptions = corruption_map(args.dataset_manifest)
    output = args.output_dir
    assets = output / "assets"
    descriptors = output / "descriptors"
    assets.mkdir(parents=True, exist_ok=True)
    descriptors.mkdir(parents=True, exist_ok=True)
    cases = []
    chunk_size = max(1, int(args.chunk_size))

    for start in range(0, len(case_keys), chunk_size):
        stop = min(start + chunk_size, len(case_keys))
        chunk_index = start // chunk_size
        records = [dict() for _ in range(start, stop)]
        for method_key, _label, _accent, record_key in METHODS:
            asset = f"assets/{method_key}_{chunk_index:03d}.smpl"
            payload = bytearray()
            for offset, case_index in enumerate(range(start, stop)):
                source = packed[method_key][record_key][case_index]
                encoded, descriptor = encode_motion135(
                    motion135(source, int(lengths[case_index])),
                    stride=1,
                )
                descriptor.update(
                    {
                        "asset": asset,
                        "translation_offset": len(payload),
                        "rotation_offset": (
                            len(payload) + descriptor["translation_count"] * 2
                        ),
                        "fps": float(args.fps),
                    }
                )
                records[offset][method_key] = descriptor
                payload.extend(encoded)
            (output / asset).write_bytes(payload)
        (descriptors / f"{chunk_index:03d}.json").write_text(
            json.dumps(
                {"start": start, "motions": records},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    starts = np.asarray(reference.get("case_starts", np.zeros(299)), dtype=np.int32)
    for index, case_key in enumerate(case_keys):
        kinds = corruptions.get(case_key, [])
        cases.append(
            {
                "case_id": f"repair_{index:03d}",
                "sample_id": case_key.rsplit("/", 1)[-1],
                "case_key": case_key,
                "references": [
                    "Corruptions: " + ", ".join(kinds) if kinds else "Corrupted motion",
                    (
                        f"Frames {int(starts[index])}–"
                        f"{int(starts[index] + lengths[index] - 1)}"
                    ),
                ],
                "segments": None,
                "outputs": None,
            }
        )

    manifest = {
        "schema_version": 3,
        "representation": "smpl_motion135",
        "task": "motion_repair",
        "title": "Motion Repair · BrokenAMASS SMPL Mesh Comparison",
        "protocol": "BrokenAMASS Z-up v2 pair-validated · fixed 100-frame crops",
        "population": len(cases),
        "asset_base_url": args.asset_base_url.rstrip("/") + "/",
        "body_model_url": "smpl_model/",
        "reference_label": "Corruption recipe",
        "motion_methods": [
            {"key": key, "label": label, "accent": accent}
            for key, label, accent, _record_key in METHODS
        ],
        "cases": cases,
        "case_descriptor_chunks": {
            "size": chunk_size,
            "path": "descriptors/{chunk}.json",
            "base_url": args.descriptor_base_url.rstrip("/") + "/",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "cases": len(cases),
        "methods": len(METHODS),
        "asset_files": len(list(assets.glob("*.smpl"))),
        "descriptor_files": len(list(descriptors.glob("*.json"))),
        "output": str(output),
    }


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
