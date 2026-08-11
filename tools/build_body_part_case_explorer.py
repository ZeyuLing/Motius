#!/usr/bin/env python3
"""Pack aligned body-part control outputs into a Three.js case explorer."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smpl_gallery_assets import encode_motion135, write_chunked_manifest


ACCENTS = (
    "#087d72",
    "#315f9d",
    "#a5412e",
    "#956000",
    "#6d4ea2",
    "#287147",
    "#9f3f72",
)


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    directory: Path
    accent: str
    indexed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", metavar="KEY=LABEL=DIR")
    parser.add_argument(
        "--indexed-source",
        action="append",
        default=[],
        metavar="KEY=LABEL=DIR",
        help="Source saved as 00000.npz, 00001.npz, ... in benchmark order.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-base-url", required=True)
    parser.add_argument("--body-model-url", required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--setting-label", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--io-workers", type=int, default=256)
    return parser.parse_args()


def parse_sources(
    values: list[str], *, indexed: bool = False, accent_offset: int = 0
) -> list[Source]:
    sources = []
    for index, value in enumerate(values):
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected KEY=LABEL=DIR, got {value!r}")
        key, label, raw_path = parts
        directory = Path(raw_path).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        sources.append(
            Source(
                key,
                label,
                directory,
                ACCENTS[(index + 1 + accent_offset) % len(ACCENTS)],
                indexed=indexed,
            )
        )
    return sources


def condition_intervals_from_mask(mask: np.ndarray) -> list[list[int]]:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError(f"Expected a (T,D) source mask, got {value.shape}")
    known = np.any(value < 0.5, axis=1)
    intervals = []
    start = None
    for frame, active in enumerate(np.append(known, False)):
        if active and start is None:
            start = frame
        elif not active and start is not None:
            intervals.append([start, frame])
            start = None
    return intervals


def condition_atoms_from_mask(mask: np.ndarray) -> dict:
    """Decode the benchmark's [22x6 rotation, 22x3 position] mask layout."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.shape[1] != 198:
        raise ValueError(f"Expected a (T,198) source mask, got {value.shape}")
    known = value < 0.5
    rotation = known[:, :132].reshape(len(value), 22, 6)
    position = known[:, 132:].reshape(len(value), 22, 3)
    rotation_joints = np.flatnonzero(rotation.any(axis=(0, 2))).tolist()
    position_joints = np.flatnonzero(position.any(axis=(0, 2))).tolist()
    position_axes = np.flatnonzero(position.any(axis=(0, 1))).tolist()
    return {
        "rotation_joint_indices": rotation_joints,
        "position_joint_indices": position_joints,
        "position_axes": ["xyz"[axis] for axis in position_axes],
    }


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = {"motion_135", "gt_motion_135", "src_mask"}
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        return {
            "motion": np.asarray(payload["motion_135"], dtype=np.float32),
            "gt": np.asarray(payload["gt_motion_135"], dtype=np.float32),
            "mask": np.asarray(payload["src_mask"], dtype=np.float32),
            "caption": str(payload["caption"]) if "caption" in payload.files else None,
        }


def source_path(source: Source, case_id: str, case_index: int) -> Path:
    filename = f"{case_index:05d}.npz" if source.indexed else f"{case_id}.npz"
    return source.directory / filename


def main() -> None:
    args = parse_args()
    sources = [
        *parse_sources(args.source),
        *parse_sources(
            args.indexed_source,
            indexed=True,
            accent_offset=len(args.source),
        ),
    ]
    source_manifest = json.loads(args.source_manifest.expanduser().resolve().read_text())
    source_cases = {
        str(item.get("case_id") or item.get("sample_id")): item
        for item in source_manifest["cases"]
    }
    named_populations = [
        {path.stem for path in source.directory.glob("*.npz")}
        for source in sources
        if not source.indexed
    ]
    case_ids = [
        case_id
        for case_id in source_cases
        if all(case_id in values for values in named_populations)
    ]
    for source in (value for value in sources if value.indexed):
        filenames = {path.name for path in source.directory.glob("*.npz")}
        missing = [
            index
            for index in range(len(case_ids))
            if f"{index:05d}.npz" not in filenames
        ]
        if missing:
            raise FileNotFoundError(
                f"{source.key} is missing {len(missing)} indexed cases; "
                f"first missing index is {missing[0]}"
            )
    if not case_ids:
        raise RuntimeError("No common body-part cases were found")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    assets = output / "assets"
    assets.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("leaderboard_smpl_gallery.html"), output / "index.html")

    methods = [
        {"key": "gt", "label": "GT", "accent": ACCENTS[0]},
        *[
            {"key": source.key, "label": source.label, "accent": source.accent}
            for source in sources
        ],
    ]
    cases = []
    chunk_size = max(1, args.chunk_size)
    stride = max(1, args.stride)
    with ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as executor:
        for start in range(0, len(case_ids), chunk_size):
            selected = case_ids[start : start + chunk_size]
            futures = {
                (source.key, case_id): executor.submit(
                    load_npz, source_path(source, case_id, start + index)
                )
                for source in sources
                for index, case_id in enumerate(selected)
            }
            records = {
                key: future.result() for key, future in futures.items()
            }
            chunk_cases = []
            first = sources[0]
            for case_id in selected:
                primary = records[(first.key, case_id)]
                frames = min(len(primary["gt"]), len(primary["mask"]))
                condition = condition_atoms_from_mask(primary["mask"][:frames])
                chunk_cases.append(
                    {
                        "case_id": case_id,
                        "sample_id": case_id,
                        "references": [source_cases[case_id]["references"][0]],
                        "condition": {
                            "label": args.setting_label,
                            **condition,
                        },
                        "condition_intervals": condition_intervals_from_mask(
                            primary["mask"][:frames]
                        ),
                        "motions": {},
                    }
                )

            for source in (value for value in sources if value.indexed):
                for case_id in selected:
                    expected = source_cases[case_id]["references"][0]
                    actual = records[(source.key, case_id)]["caption"]
                    if actual is not None and actual != expected:
                        raise ValueError(
                            f"{source.key} benchmark order mismatch for {case_id}: "
                            f"{actual!r} != {expected!r}"
                        )

            for method in methods:
                payload = bytearray()
                asset_name = f"{method['key']}_{start // chunk_size:03d}.smpl"
                for case, case_id in zip(chunk_cases, selected):
                    primary = records[(first.key, case_id)]
                    motion = (
                        primary["gt"]
                        if method["key"] == "gt"
                        else records[(method["key"], case_id)]["motion"]
                    )
                    encoded, descriptor = encode_motion135(motion, stride=stride)
                    offset = len(payload)
                    descriptor.update(
                        {
                            "asset": f"assets/{asset_name}",
                            "translation_offset": offset,
                            "rotation_offset": offset + descriptor["translation_count"] * 2,
                            "fps": float(args.fps),
                        }
                    )
                    case["motions"][method["key"]] = descriptor
                    payload.extend(encoded)
                (assets / asset_name).write_bytes(payload)
            cases.extend(chunk_cases)
            print(f"exported {len(cases)}/{len(case_ids)} cases", flush=True)

    manifest = {
        "schema_version": 3,
        "representation": "smpl_motion135",
        "task": "part_level_motion_control",
        "title": "HumanML3D Part-Level Motion Control Comparison",
        "protocol": {
            "setting_id": args.setting_id,
            "setting": args.setting_label,
            "population": len(cases),
            "coverage": "GT plus persisted baseline inference artifacts",
        },
        "population": len(cases),
        "asset_base_url": args.asset_base_url,
        "body_model_url": args.body_model_url,
        "reference_label": "Selected caption and control setting",
        "condition_legend": {
            "conditioned": {"label": "Supplied frame", "color": "#d95f02"},
            "generated": {"label": "Generated frame", "color": "#66736d"},
            "marker": {"label": "Controlled body part", "color": "#d95f02"},
        },
        "motion_methods": methods,
        "cases": cases,
    }
    write_chunked_manifest(output, manifest, chunk_size=chunk_size)
    print(json.dumps({"output": str(output), "cases": len(cases), "methods": len(methods)}))


if __name__ == "__main__":
    main()
