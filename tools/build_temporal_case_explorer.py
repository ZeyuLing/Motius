#!/usr/bin/env python3
"""Build all-case Temporal Motion Completion galleries with SMPL meshes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from smpl_gallery_assets import encode_motion135, write_chunked_manifest


SETTINGS = (
    ("start_1f", "Prediction: first frame"),
    ("pre20", "Prediction: first 20%"),
    ("pre20_uncond", "Prediction: first 20%, motion-only"),
    ("both_1f", "In-betweening: first and last frame"),
    ("mid80", "In-betweening: middle 80%"),
    ("mid80_uncond", "In-betweening: middle 80%, motion-only"),
    ("adaptive_keyframes", "Keyframe: adaptive sparse frames"),
    ("adaptive_keyframes_uncond", "Keyframe: adaptive sparse frames, motion-only"),
)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    accent: str


METHODS = (
    Method("gt", "GT", "#956000"),
    Method("condmdi", "CondMDI", "#315f9d"),
    Method("flowmdm", "FlowMDM", "#a5412e"),
    Method("kimodo", "KIMODO", "#6d4ea2"),
    Method("maskcontrol", "MaskControl", "#d95f02"),
    Method("motionlab", "MotionLab", "#287147"),
    Method("omnicontrol", "OmniControl", "#b34b8c"),
    Method("ours", "MotionCanvas", "#087d72"),
)

KEYFRAME_METHOD_KEYS = {
    "gt",
    "condmdi",
    "kimodo",
    "maskcontrol",
    "motionlab",
    "omnicontrol",
    "ours",
}


def condition_intervals(
    setting: str,
    length: int,
    *,
    case_id: str = "",
    keyframes: dict[str, dict] | None = None,
) -> list[list[int]]:
    """Return half-open frame intervals that are supplied to the generator."""

    frames = max(1, int(length))
    mode = setting.removesuffix("_uncond")
    if mode == "start_1f":
        return [[0, 1]]
    if mode == "both_1f":
        return [[0, 1]] if frames == 1 else [[0, 1], [frames - 1, frames]]
    if mode == "pre20":
        count = max(1, int(round(frames * 0.2)))
        return [[0, min(count, frames)]]
    if mode == "mid80":
        count = min(frames, max(1, int(round(frames * 0.1))))
        if count * 2 >= frames:
            return [[0, frames]]
        return [[0, count], [frames - count, frames]]
    if mode == "adaptive_keyframes":
        if keyframes is None or case_id not in keyframes:
            raise KeyError(f"No adaptive keyframes for case {case_id!r}")
        record = keyframes[case_id]
        source_length = max(1, int(record.get("T", frames)))
        indices = record.get("keyframe_indices")
        if not isinstance(indices, list):
            raise ValueError(f"Invalid adaptive keyframes for case {case_id!r}")
        if source_length == frames:
            mapped = [int(index) for index in indices]
        else:
            denominator = max(1, source_length - 1)
            mapped = [round(int(index) * (frames - 1) / denominator) for index in indices]
        return [
            [index, index + 1]
            for index in sorted({max(0, min(frames - 1, value)) for value in mapped})
        ]
    raise ValueError(f"Unsupported temporal setting: {setting}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-base-url", required=True)
    parser.add_argument("--keyframe-file", type=Path)
    parser.add_argument("--body-model-url", default="../smpl_model/")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--io-workers", type=int, default=64)
    parser.add_argument(
        "--settings",
        nargs="*",
        choices=[value for value, _ in SETTINGS],
        default=[value for value, _ in SETTINGS],
    )
    return parser.parse_args()


def load_motion(
    temporal_root: Path,
    setting: str,
    method: str,
    case_id: str,
    sample_index: int,
    max_frames: int,
) -> np.ndarray:
    setting_root = temporal_root / f"temporal_{setting}"
    if method == "ours":
        direct = setting_root / method / "eval_npz" / f"{sample_index:05d}.npz"
        shard_manifest = setting_root / method / "eval_npz_shards.json"
        if direct.is_file():
            path = direct
        elif shard_manifest.is_file():
            shard_spec = json.loads(shard_manifest.read_text())
            outputs_root = next(
                (parent for parent in temporal_root.parents if parent.name == "outputs"),
                None,
            )
            repository_root = outputs_root.parent if outputs_root else Path.cwd()
            directories = [
                value if value.is_absolute() else repository_root / value
                for value in map(Path, shard_spec["directories"])
            ]
            path = directories[sample_index % len(directories)] / f"{sample_index:05d}.npz"
        else:
            path = direct
        key = "motion_135"
    elif method == "gt":
        path = setting_root / "condmdi" / "eval_npz" / f"{case_id}.npz"
        key = "gt_motion_135"
    else:
        path = setting_root / method / "eval_npz" / f"{case_id}.npz"
        key = "motion_135"
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"{key!r} not found in {path}: {payload.files}")
        motion = np.asarray(payload[key], dtype=np.float32)[:max_frames, :135]
    if motion.ndim != 2 or motion.shape[1] != 135 or not np.isfinite(motion).all():
        raise ValueError(f"Invalid motion_135 in {path}: {motion.shape}")
    return np.ascontiguousarray(motion)


def load_source_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    spec = manifest.get("case_descriptor_chunks")
    if not spec:
        return manifest
    size = max(1, int(spec["size"]))
    for chunk in range((len(manifest["cases"]) + size - 1) // size):
        chunk_path = path.parent / spec["path"].replace("{chunk}", f"{chunk:03d}")
        payload = json.loads(chunk_path.read_text())
        for offset, motions in enumerate(payload["motions"]):
            manifest["cases"][payload["start"] + offset]["motions"] = motions
    return manifest


def display_references(references: object, setting_label: str) -> object:
    if not isinstance(references, list):
        return references
    values = [
        value
        for value in references
        if not (isinstance(value, str) and value.startswith("Condition: "))
    ]
    return [*values, f"Condition: {setting_label}"]


def build_setting(args: argparse.Namespace, setting: str, setting_label: str) -> None:
    source_path = args.source_root.expanduser().resolve() / setting / "manifest.json"
    if not source_path.is_file() and setting.startswith("adaptive_keyframes"):
        source_path = args.source_root.expanduser().resolve() / "start_1f" / "manifest.json"
    source_manifest = load_source_manifest(source_path)
    keyframes = None
    if setting.startswith("adaptive_keyframes"):
        if args.keyframe_file is None:
            raise ValueError("Adaptive keyframe galleries require --keyframe-file")
        keyframe_payload = json.loads(args.keyframe_file.expanduser().resolve().read_text())
        keyframes = keyframe_payload.get("data_list", keyframe_payload)
    methods = tuple(
        method
        for method in METHODS
        if not setting.startswith("adaptive_keyframes") or method.key in KEYFRAME_METHOD_KEYS
    )
    output_dir = args.output_dir.expanduser().resolve() / setting
    assets_dir = output_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("leaderboard_smpl_gallery.html"), output_dir / "index.html")

    cases = []
    for item in source_manifest["cases"]:
        descriptor = next(iter(item["motions"].values()))
        duration = float(descriptor["display_frames"]) / float(descriptor["fps"])
        max_frames = max(1, round(duration * 30.0))
        cases.append(
            {
                "case_id": str(item["case_id"]),
                "sample_id": str(item.get("sample_id") or item["case_id"]),
                "references": display_references(item.get("references"), setting_label),
                "condition_intervals": condition_intervals(
                    setting,
                    max_frames,
                    case_id=str(item["case_id"]),
                    keyframes=keyframes,
                ),
                "motions": {},
                "_max_frames": max_frames,
            }
        )

    stride = max(1, args.stride)
    chunk_size = max(1, args.chunk_size)
    with ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as executor:
        for start in range(0, len(cases), chunk_size):
            end = min(start + chunk_size, len(cases))
            chunk = cases[start:end]
            futures = {
                (method.key, index): executor.submit(
                    load_motion,
                    args.temporal_root.expanduser().resolve(),
                    setting,
                    method.key,
                    item["case_id"],
                    start + index,
                    item["_max_frames"],
                )
                for method in methods
                for index, item in enumerate(chunk)
            }
            for method in methods:
                payload = bytearray()
                asset_name = f"{method.key}_{start // chunk_size:03d}.smpl"
                for index, item in enumerate(chunk):
                    motion = futures[(method.key, index)].result()
                    encoded, descriptor = encode_motion135(motion, stride=stride)
                    byte_offset = len(payload)
                    descriptor.update(
                        {
                            "asset": f"assets/{asset_name}",
                            "translation_offset": byte_offset,
                            "rotation_offset": byte_offset
                            + descriptor["translation_count"] * 2,
                            "fps": 30.0,
                        }
                    )
                    item["motions"][method.key] = descriptor
                    payload.extend(encoded)
                (assets_dir / asset_name).write_bytes(payload)
            print(f"{setting}: exported {end}/{len(cases)} cases", flush=True)

    manifest = {
        "schema_version": 2,
        "representation": "smpl_motion135",
        "task": "temporal_condition_generation",
        "title": f"Temporal Motion Completion · HumanML3D: {setting_label}",
        "protocol": source_manifest.get("protocol"),
        "population": len(cases),
        "asset_base_url": f"{args.asset_base_url.rstrip('/')}/{setting}/",
        "body_model_url": args.body_model_url,
        "reference_label": "Text and temporal condition",
        "condition_legend": {
            "conditioned": {"label": "Condition frame", "color": "#d95f02"},
            "generated": {"label": "Generated frame"},
        },
        "display_canonicalization": {
            "ground": "per_clip_global_smpl_mesh_minimum",
            "transform": "single_rigid_vertical_translation",
        },
        "motion_methods": [method.__dict__ for method in methods],
        "cases": cases,
    }
    for item in manifest["cases"]:
        item.pop("_max_frames")
    write_chunked_manifest(output_dir, manifest, chunk_size=chunk_size)


def main() -> None:
    args = parse_args()
    labels = dict(SETTINGS)
    for setting in args.settings:
        build_setting(args, setting, labels[setting])


if __name__ == "__main__":
    main()
