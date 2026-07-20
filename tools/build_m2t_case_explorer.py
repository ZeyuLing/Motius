#!/usr/bin/env python3
"""Build the complete, lazy-loaded HumanML3D M2T case explorer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.retarget._hml263_smpl_impl import recover_from_ric


METHODS = (
    ("tm2t", "TM2T", "#087d72"),
    ("motiongpt", "MotionGPT", "#315f9d"),
    ("motiongpt3", "MotionGPT3", "#a5412e"),
    ("vermo", "VerMo", "#956000"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("outputs/m2t/humanml3d/protocol_manifest.json"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/m2t/humanml3d/full_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/leaderboards/hf_space_m2t_humanml3d/cases"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--asset-base-url",
        help="Optional public base URL used by the browser instead of local assets/.",
    )
    return parser.parse_args()


def _prediction(root: Path, method: str, sample_id: str) -> tuple[str, str]:
    path = root / method / "predictions" / f"{sample_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = str(json.loads(path.read_text(encoding="utf-8"))["prediction"]).strip()
    return sample_id, value


def _quantize(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = joints.min(axis=(0, 1)).astype(np.float32)
    maximum = joints.max(axis=(0, 1)).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    values = np.rint((joints - minimum) / scale).clip(0, 65535).astype("<u2")
    return values, minimum, scale


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_manifest.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else Path(protocol["data_root"]).expanduser().resolve()
    )
    prediction_root = args.prediction_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)

    chunk_size = max(1, int(args.chunk_size))
    unique_ids = list(dict.fromkeys(str(item["sample_id"]) for item in protocol["samples"]))
    predictions = {}
    with ThreadPoolExecutor(max_workers=64) as executor:
        for method, _label, _color in METHODS:
            predictions[method] = dict(
                executor.map(
                    lambda sample_id, method=method: _prediction(
                        prediction_root, method, sample_id
                    ),
                    unique_ids,
                )
            )
            print(f"loaded {method}: {len(predictions[method])} predictions", flush=True)

    def load_joints(sample: dict) -> np.ndarray:
        source = np.load(data_root / sample["motion_path"], mmap_mode="r")
        start = int(sample["start_frame"])
        end = min(int(sample["end_frame"]), start + 196)
        clip = np.asarray(source[start:end], dtype=np.float32)
        joints = recover_from_ric(clip, 22).astype(np.float32)
        joints[..., 1] -= float(joints[..., 1].min())
        return joints

    cases = []
    chunk_values: list[np.ndarray] = []
    chunk_index = 0
    chunk_offset = 0
    occurrences: dict[str, int] = {}

    def flush_chunk() -> None:
        nonlocal chunk_values, chunk_index, chunk_offset
        if not chunk_values:
            return
        path = assets_dir / f"motions_{chunk_index:03d}.u16"
        np.concatenate(chunk_values).astype("<u2", copy=False).tofile(path)
        chunk_values = []
        chunk_index += 1
        chunk_offset = 0

    with ThreadPoolExecutor(max_workers=32) as executor:
        all_joints = executor.map(load_joints, protocol["samples"])
        iterable = zip(protocol["samples"], all_joints)
        for index, (sample, joints) in enumerate(iterable):
            if index and index % chunk_size == 0:
                flush_chunk()
            sample_id = str(sample["sample_id"])
            occurrence = occurrences.get(sample_id, 0)
            occurrences[sample_id] = occurrence + 1
            quantized, minimum, scale = _quantize(joints)
            flat = quantized.reshape(-1)
            chunk_values.append(flat)
            cases.append(
                {
                    "index": index,
                    "case_key": f"{sample_id}#{occurrence}",
                    "sample_id": sample_id,
                    "source_id": sample["source_id"],
                    "frames": int(len(joints)),
                    "fps": 20.0,
                    "asset": f"assets/motions_{chunk_index:03d}.u16",
                    "offset": int(chunk_offset),
                    "count": int(flat.size),
                    "minimum": minimum.tolist(),
                    "scale": scale.tolist(),
                    "references": [
                        item["caption"] for item in sample["captions"][:3]
                    ],
                    "outputs": {
                        method: predictions[method][sample_id]
                        for method, _label, _color in METHODS
                    },
                }
            )
            chunk_offset += int(flat.size)
            if (index + 1) % 250 == 0:
                print(f"exported {index + 1}/{len(protocol['samples'])}", flush=True)
    flush_chunk()

    shutil.copy2(
        Path(__file__).with_name("leaderboard_case_explorer.html"),
        output_dir / "index.html",
    )
    manifest = {
        "schema_version": 1,
        "task": "m2t",
        "title": "HumanML3D M2T Case Explorer",
        "protocol": protocol["protocol"],
        "population": len(cases),
        "asset_base_url": args.asset_base_url,
        "motion_label": "Ground-truth motion",
        "methods": [
            {"key": key, "label": label, "accent": accent}
            for key, label, accent in METHODS
        ],
        "parents": [-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 12, 16, 17, 18, 19],
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "cases": len(cases),
                "chunks": chunk_index,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
