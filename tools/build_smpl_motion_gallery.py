#!/usr/bin/env python3
"""Pack aligned motion135 directories for an all-case SMPL mesh gallery."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smpl_gallery_assets import (
    encode_joint_positions,
    encode_motion135,
    load_joint_positions,
    load_motion135,
    resample_joint_positions,
)


ACCENTS = (
    "#087d72", "#315f9d", "#a5412e", "#956000", "#6d4ea2",
    "#287147", "#9f3f72", "#46646f", "#b35c16", "#345d2d",
)


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    directory: Path
    accent: str
    suffix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--motion", required=True, action="append", metavar="KEY=LABEL=DIR")
    parser.add_argument("--skeleton", action="append", default=[], metavar="KEY=LABEL=DIR")
    parser.add_argument("--skeleton-fps", type=float, default=60.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-base-url")
    parser.add_argument("--body-model-url", default="smpl_model/")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--io-workers", type=int, default=64)
    parser.add_argument("--title", default="Motius SMPL Mesh Comparison")
    return parser.parse_args()


def parse_sources(values: list[str], *, suffixes=(".npz", ".npy")) -> list[Source]:
    sources = []
    for index, value in enumerate(values):
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected KEY=LABEL=DIR, got {value!r}")
        key, label, raw_path = parts
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(path)
        suffix = next((value for value in suffixes if next(path.glob(f"*{value}"), None)), None)
        if suffix is None:
            raise FileNotFoundError(f"No supported files {suffixes} under {path}")
        sources.append(Source(key, label, path, ACCENTS[index % len(ACCENTS)], suffix))
    return sources


def load_motion_record(path: Path, *, max_frames: int) -> tuple[np.ndarray, float | None]:
    motion = load_motion135(path, max_frames=max_frames)
    fit_mean = None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "fit_mpjpe_mm" in payload.files:
                errors = np.asarray(payload["fit_mpjpe_mm"], dtype=np.float32)[: len(motion)]
                if errors.size and np.isfinite(errors).all():
                    fit_mean = float(errors.mean())
    return motion, fit_mean


def load_skeleton_record(
    path: Path,
    *,
    source_fps: float,
    target_fps: float,
    target_frames: int,
) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "fps" in payload.files:
                source_fps = float(np.asarray(payload["fps"]).item())
    return resample_joint_positions(
        load_joint_positions(path),
        source_fps=source_fps,
        target_fps=target_fps,
        target_frames=target_frames,
    )


def main() -> None:
    args = parse_args()
    source_manifest = json.loads(args.source_manifest.expanduser().resolve().read_text())
    sources = parse_sources(args.motion)
    skeleton_sources = parse_sources(
        args.skeleton, suffixes=(".npz", ".npy", ".json")
    )
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("leaderboard_smpl_gallery.html"), output / "index.html")

    cases = []
    for item in source_manifest["cases"]:
        if item.get("motions"):
            source_descriptor = next(iter(item["motions"].values()))
            source_frames = int(source_descriptor.get("display_frames") or source_descriptor["frames"])
            source_fps = float(source_descriptor.get("fps") or args.fps)
        else:
            source_frames = int(item.get("display_frames") or item.get("frames") or 1)
            source_fps = float(item.get("fps") or args.fps)
        cases.append({
            "case_id": str(item.get("case_id") or item.get("sample_id")),
            "sample_id": str(item.get("sample_id") or item.get("case_id")),
            "case_key": item.get("case_key"),
            "references": item.get("references"),
            "segments": item.get("segments"),
            "outputs": item.get("outputs"),
            "audio": item.get("audio"),
            "audio_start_seconds": item.get("audio_start_seconds"),
            "audio_end_seconds": item.get("audio_end_seconds"),
            "motions": {},
            "skeletons": {},
            "_max_frames": max(1, round(source_frames * float(args.fps) / source_fps)),
        })

    stride = max(1, args.stride)
    chunk_size = max(1, args.chunk_size)
    workers = max(1, args.io_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(cases), chunk_size):
            end = min(start + chunk_size, len(cases))
            chunk = cases[start:end]
            futures = {
                (source.key, index): executor.submit(
                    load_motion_record,
                    source.directory / f"{item['case_id']}{source.suffix}",
                    max_frames=item["_max_frames"],
                )
                for source in sources
                for index, item in enumerate(chunk)
            }
            for source in sources:
                payload = bytearray()
                asset_name = f"{source.key}_{start // chunk_size:03d}.smpl"
                for index, item in enumerate(chunk):
                    motion, fit_mean = futures[(source.key, index)].result()
                    encoded, descriptor = encode_motion135(motion, stride=stride)
                    byte_offset = len(payload)
                    descriptor.update({
                        "asset": f"assets/{asset_name}",
                        "translation_offset": byte_offset,
                        "rotation_offset": byte_offset + descriptor["translation_count"] * 2,
                        "fps": float(args.fps),
                    })
                    if fit_mean is not None:
                        descriptor["fit_mpjpe_mm_mean"] = fit_mean
                    item["motions"][source.key] = descriptor
                    payload.extend(encoded)
                (assets / asset_name).write_bytes(payload)
            skeleton_futures = {
                (source.key, index): executor.submit(
                    load_skeleton_record,
                    source.directory / f"{item['case_id']}{source.suffix}",
                    source_fps=float(args.skeleton_fps),
                    target_fps=float(args.fps),
                    target_frames=item["_max_frames"],
                )
                for source in skeleton_sources
                for index, item in enumerate(chunk)
            }
            for source in skeleton_sources:
                payload = bytearray()
                asset_name = f"{source.key}_skeleton_{start // chunk_size:03d}.joints"
                for index, item in enumerate(chunk):
                    joints = skeleton_futures[(source.key, index)].result()
                    encoded, descriptor = encode_joint_positions(joints)
                    descriptor.update({
                        "asset": f"assets/{asset_name}",
                        "position_offset": len(payload),
                        "fps": float(args.fps),
                        "representation": "aistpp_smpl24_joints",
                    })
                    item["skeletons"][source.key] = descriptor
                    payload.extend(encoded)
                (assets / asset_name).write_bytes(payload)
            print(f"exported {end}/{len(cases)} cases", flush=True)

    manifest = {
        "schema_version": 2,
        "representation": "smpl_motion135",
        "task": source_manifest.get("task", "motion_generation"),
        "title": args.title,
        "protocol": source_manifest.get("protocol"),
        "population": len(cases),
        "asset_base_url": args.asset_base_url,
        "body_model_url": args.body_model_url,
        "reference_label": source_manifest.get("reference_label", "Input caption"),
        "motion_methods": [source.__dict__ | {"directory": None} for source in sources],
        "skeleton_methods": [
            source.__dict__ | {"directory": None} for source in skeleton_sources
        ],
        "cases": cases,
    }
    for item in manifest["cases"]:
        item.pop("_max_frames", None)
        if not item["skeletons"]:
            item.pop("skeletons")
        for key in ("audio", "audio_start_seconds", "audio_end_seconds"):
            if item.get(key) is None:
                item.pop(key, None)
    for method in manifest["motion_methods"] + manifest["skeleton_methods"]:
        method.pop("directory", None)
        method.pop("suffix", None)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"output": str(output), "cases": len(cases), "methods": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
