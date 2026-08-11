#!/usr/bin/env python3
"""Build the all-case M2T comparison with a true SMPL input-motion preview."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip
from smpl_gallery_assets import encode_motion135, load_motion135


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--t2m-gt-motion135", required=True, type=Path)
    parser.add_argument("--smpl-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-base-url", required=True)
    parser.add_argument("--body-model-url", default="smpl_model/")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_manifest.expanduser().resolve().read_text())
    protocol = json.loads(args.protocol_manifest.expanduser().resolve().read_text())
    samples = protocol["samples"]
    if len(samples) != len(source["cases"]):
        raise ValueError("M2T source and protocol manifests do not have the same population")
    for old, sample in zip(source["cases"], samples):
        if str(old["sample_id"]) != str(sample["sample_id"]):
            raise ValueError(f"M2T order mismatch: {old['sample_id']} != {sample['sample_id']}")

    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("leaderboard_smpl_gallery.html"), output / "index.html")
    t2m_gt = args.t2m_gt_motion135.expanduser().resolve()
    t2m_names = set(os.listdir(t2m_gt))
    data_root = Path(protocol["data_root"]).expanduser().resolve()
    smpl_rest = load_smpl_rest(args.smpl_model.expanduser().resolve(), "cpu", gender="neutral")

    def load_sample(sample: dict) -> tuple[np.ndarray, str]:
        target_frames = max(1, round((int(sample["end_frame"]) - int(sample["start_frame"])) * 1.5))
        direct_name = f"{sample['sample_id']}.npz"
        source_name = f"{sample['source_id']}.npz"
        if direct_name in t2m_names:
            return load_motion135(t2m_gt / direct_name, max_frames=target_frames), "official_t2m_gt_motion135"
        if int(sample["start_frame"]) == 0 and source_name in t2m_names:
            return load_motion135(t2m_gt / source_name, max_frames=target_frames), "official_t2m_gt_motion135"
        features = np.asarray(np.load(data_root / sample["motion_path"]), dtype=np.float32)
        features = features[int(sample["start_frame"]):int(sample["end_frame"])]
        converted = retarget_hml263_clip(
            features,
            smpl_rest=smpl_rest,
            device="cpu",
            gender="neutral",
            source_fps=20.0,
            target_fps=30.0,
            target_len=target_frames,
            rotation_init="position_ik",
            floor_align=True,
            refine_iters=0,
        )
        return np.asarray(converted["motion_135"], dtype=np.float32), "position_ik_from_official_hml263"

    cases = []
    stride = max(1, args.stride)
    chunk_size = max(1, args.chunk_size)
    with ThreadPoolExecutor(max_workers=64) as executor:
        for start in range(0, len(samples), chunk_size):
            end = min(start + chunk_size, len(samples))
            payload = bytearray()
            asset_name = f"gt_{start // chunk_size:03d}.smpl"
            loaded = executor.map(load_sample, samples[start:end])
            for old, sample, (motion, source_kind) in zip(
                source["cases"][start:end], samples[start:end], loaded
            ):
                encoded, descriptor = encode_motion135(motion, stride=stride)
                byte_offset = len(payload)
                descriptor.update({
                    "asset": f"assets/{asset_name}",
                    "translation_offset": byte_offset,
                    "rotation_offset": byte_offset + descriptor["translation_count"] * 2,
                    "fps": 30.0,
                    "source": source_kind,
                })
                payload.extend(encoded)
                cases.append({
                    "case_id": str(old.get("case_key") or old["sample_id"]),
                    "case_key": old.get("case_key"),
                    "sample_id": str(old["sample_id"]),
                    "references": old.get("references"),
                    "outputs": old.get("outputs"),
                    "motions": {"gt": descriptor},
                })
            (assets / asset_name).write_bytes(payload)
            print(f"exported {end}/{len(samples)} M2T cases", flush=True)

    manifest = {
        "schema_version": 2,
        "representation": "smpl_motion135",
        "task": "motion_to_text",
        "title": "Motion-to-Text · HumanML3D: SMPL Mesh and Caption Comparison",
        "protocol": source.get("protocol"),
        "population": len(cases),
        "asset_base_url": args.asset_base_url,
        "body_model_url": args.body_model_url,
        "reference_label": "Reference captions",
        "motion_methods": [{"key": "gt", "label": "Input SMPL Motion", "accent": "#956000"}],
        "output_methods": source["methods"],
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "cases": len(cases), "methods": len(source["methods"])}, indent=2))


if __name__ == "__main__":
    main()
