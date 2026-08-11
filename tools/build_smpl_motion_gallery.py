#!/usr/bin/env python3
"""Pack aligned motion135 directories for an all-case SMPL mesh gallery."""

from __future__ import annotations

import argparse
import json
import shutil
import time
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
    write_chunked_manifest,
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
    parser.add_argument(
        "--motion-asset-base",
        action="append",
        default=[],
        metavar="KEY=URL",
        help="Override the public asset base for selected motion methods.",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed chunk checkpoints in the output directory.",
    )
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


def parse_asset_bases(values: list[str], source_keys: set[str]) -> dict[str, str]:
    bases = {}
    for value in values:
        key, separator, base = value.partition("=")
        if not separator or not key or not base:
            raise ValueError(f"Expected KEY=URL, got {value!r}")
        if key not in source_keys:
            raise KeyError(f"Unknown motion source in asset override: {key}")
        bases[key] = base.rstrip("/") + "/"
    return bases


def load_motion_record(
    path: Path, *, max_frames: int | None
) -> tuple[np.ndarray, float | None]:
    motion = load_motion135(path, max_frames=max_frames)
    fit_mean = None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "fit_mpjpe_mm" in payload.files:
                errors = np.asarray(payload["fit_mpjpe_mm"], dtype=np.float32)[: len(motion)]
                if errors.size and np.isfinite(errors).all():
                    fit_mean = float(errors.mean())
    return motion, fit_mean


def load_motion_record_with_retry(
    path: Path, *, max_frames: int | None, attempts: int = 4
) -> tuple[np.ndarray, float | None]:
    """Retry transient shared-filesystem reads while preserving useful errors."""

    for attempt in range(attempts):
        try:
            return load_motion_record(path, max_frames=max_frames)
        except (OSError, EOFError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Failed to read motion after {attempts} attempts: {path}") from error
            time.sleep(0.25 * (2**attempt))
    raise AssertionError("unreachable")


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
    motion_asset_bases = parse_asset_bases(
        args.motion_asset_base, {source.key for source in sources}
    )
    skeleton_sources = parse_sources(
        args.skeleton, suffixes=(".npz", ".npy", ".json")
    )
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    checkpoints = output / ".build_chunks"
    if assets.exists() and not args.resume:
        shutil.rmtree(assets)
    if checkpoints.exists() and not args.resume:
        shutil.rmtree(checkpoints)
    assets.mkdir(parents=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
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
        requested_frames = (
            max(1, round(source_frames * float(args.fps) / source_fps))
            if item.get("motions")
            or item.get("display_frames")
            or item.get("frames")
            else None
        )
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
            "_max_frames": requested_frames,
        })

    stride = max(1, args.stride)
    chunk_size = max(1, args.chunk_size)
    workers = max(1, args.io_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(cases), chunk_size):
            end = min(start + chunk_size, len(cases))
            chunk = cases[start:end]
            chunk_index = start // chunk_size
            checkpoint_path = checkpoints / f"{chunk_index:03d}.json"
            expected_assets = [
                assets / f"{source.key}_{chunk_index:03d}.smpl" for source in sources
            ] + [
                assets / f"{source.key}_skeleton_{chunk_index:03d}.joints"
                for source in skeleton_sources
            ]
            if args.resume and checkpoint_path.is_file() and all(
                path.is_file() for path in expected_assets
            ):
                restored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if len(restored) != len(chunk):
                    raise ValueError(f"Invalid chunk checkpoint: {checkpoint_path}")
                cases[start:end] = restored
                print(f"resumed {end}/{len(cases)} cases", flush=True)
                continue
            futures = {
                (source.key, index): executor.submit(
                    load_motion_record_with_retry,
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
                    asset = f"assets/{asset_name}"
                    if source.key in motion_asset_bases:
                        asset = f"{motion_asset_bases[source.key]}{asset_name}"
                    descriptor.update({
                        "asset": asset,
                        "translation_offset": byte_offset,
                        "rotation_offset": byte_offset + descriptor["translation_count"] * 2,
                        "fps": float(args.fps),
                    })
                    if fit_mean is not None:
                        descriptor["fit_mpjpe_mm_mean"] = fit_mean
                    item["motions"][source.key] = descriptor
                    payload.extend(encoded)
                if source.key in motion_asset_bases:
                    # Give same-origin fallback assets a distinct Xet object.
                    # The viewer only addresses descriptor ranges, so this
                    # trailing marker is intentionally outside readable data.
                    payload.extend(b"MOTIUS")
                (assets / asset_name).write_bytes(payload)
            target_frames = [
                item["_max_frames"]
                or len(futures[(sources[0].key, index)].result()[0])
                for index, item in enumerate(chunk)
            ]
            skeleton_futures = {
                (source.key, index): executor.submit(
                    load_skeleton_record,
                    source.directory / f"{item['case_id']}{source.suffix}",
                    source_fps=float(args.skeleton_fps),
                    target_fps=float(args.fps),
                    target_frames=target_frames[index],
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
            checkpoint_tmp = checkpoint_path.with_suffix(".json.tmp")
            checkpoint_tmp.write_text(
                json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            checkpoint_tmp.replace(checkpoint_path)
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
    write_chunked_manifest(output, manifest, chunk_size=chunk_size)
    shutil.rmtree(checkpoints)
    print(json.dumps({"output": str(output), "cases": len(cases), "methods": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
