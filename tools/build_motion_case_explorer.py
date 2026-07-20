#!/usr/bin/env python3
"""Build a complete lazy-loaded case explorer from aligned motion directories."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.retarget._hml263_smpl_impl import recover_from_ric


@dataclass(frozen=True)
class MotionSource:
    key: str
    label: str
    directory: Path
    accent: str


ACCENTS = (
    "#087d72",
    "#315f9d",
    "#a5412e",
    "#956000",
    "#6d4ea2",
    "#287147",
    "#9f3f72",
    "#46646f",
    "#b35c16",
    "#345d2d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--motion",
        action="append",
        required=True,
        metavar="KEY=LABEL=DIR",
        help="Aligned motion source; repeat once per displayed method.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-representation",
        choices=("joints66", "hml263"),
        default="joints66",
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--io-workers", type=int, default=64)
    parser.add_argument("--title", default="Motius All-Case Comparison")
    parser.add_argument(
        "--asset-base-url",
        help="Optional public base URL used by the browser instead of local assets/.",
    )
    return parser.parse_args()


def parse_sources(values: list[str]) -> list[MotionSource]:
    sources = []
    for index, value in enumerate(values):
        parts = value.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"--motion expects KEY=LABEL=DIR, got {value!r}.")
        key, label, raw_directory = parts
        directory = Path(raw_directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        sources.append(MotionSource(key, label, directory, ACCENTS[index % len(ACCENTS)]))
    if len({source.key for source in sources}) != len(sources):
        raise ValueError("Motion source keys must be unique.")
    return sources


def load_cases(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("cases"), list):
        return payload, payload["cases"]
    if isinstance(payload.get("data_list"), dict):
        cases = []
        for case_id, value in payload["data_list"].items():
            item = dict(value)
            item.setdefault("case_id", case_id)
            item.setdefault("sample_id", case_id)
            item["references"] = [item.get("selected_caption") or item.get("caption") or ""]
            cases.append(item)
        return payload, cases
    raise ValueError(f"Unsupported case manifest schema: {path}")


def motion_path(source: MotionSource, case_id: str) -> Path:
    npy = source.directory / f"{case_id}.npy"
    if npy.is_file():
        return npy
    npz = source.directory / f"{case_id}.npz"
    if npz.is_file():
        return npz
    raise FileNotFoundError(f"No motion for {case_id!r} under {source.directory}")


def load_array(
    source: MotionSource,
    case_id: str,
    representation: str,
    stride: int,
) -> tuple[np.ndarray, int]:
    path = motion_path(source, case_id)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        for key in ("joints", "joints66", "motion", "motion263", "arr_0"):
            if key in loaded.files:
                value = np.asarray(loaded[key], dtype=np.float32)
                break
        else:
            raise ValueError(f"No supported motion array in {path}: {loaded.files}")
    else:
        value = np.asarray(loaded, dtype=np.float32)
    if representation == "hml263":
        if value.ndim != 2 or value.shape[1] != 263:
            raise ValueError(f"Expected (T,263) HML motion in {path}, got {value.shape}")
        value = recover_from_ric(value, 22)
    elif value.ndim == 2 and value.shape[1] == 66:
        value = value.reshape(-1, 22, 3)
    if value.ndim != 3 or value.shape[1:] != (22, 3):
        raise ValueError(f"Expected (T,22,3) joints in {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite joints in {path}")
    display_frames = len(value)
    value = np.ascontiguousarray(value[::stride], dtype=np.float32)
    value[..., 1] -= float(value[..., 1].min())
    return value, display_frames


def quantize(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = joints.min(axis=(0, 1)).astype(np.float32)
    maximum = joints.max(axis=(0, 1)).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    values = np.rint((joints - minimum) / scale).clip(0, 65535).astype("<u2")
    return values, minimum, scale


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    source_manifest, source_cases = load_cases(manifest_path)
    sources = parse_sources(args.motion)
    output_dir = args.output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).with_name("leaderboard_case_explorer.html"),
        output_dir / "index.html",
    )

    stride = max(1, int(args.stride))
    chunk_size = max(1, int(args.chunk_size))
    fps = float(args.fps or source_manifest.get("fps") or 20.0)
    cases = []
    for source_case in source_cases:
        case_id = str(source_case.get("case_id") or source_case.get("sample_id"))
        segments = source_case.get("segments")
        references = source_case.get("references")
        cases.append(
            {
                "case_id": case_id,
                "sample_id": str(source_case.get("sample_id") or case_id),
                "segments": segments,
                "references": references,
                "motions": {},
            }
        )

    workers = max(1, int(args.io_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(0, len(cases), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(cases))
            chunk_cases = cases[chunk_start:chunk_end]
            for source in sources:
                loaded = executor.map(
                    lambda item, source=source: load_array(
                        source,
                        item["case_id"],
                        args.input_representation,
                        stride,
                    ),
                    chunk_cases,
                )
                values = []
                offset = 0
                asset_name = f"{source.key}_{chunk_start // chunk_size:03d}.u16"
                for item, (joints, display_frames) in zip(chunk_cases, loaded):
                    quantized, minimum, scale = quantize(joints)
                    flat = quantized.reshape(-1)
                    values.append(flat)
                    item["motions"][source.key] = {
                        "asset": f"assets/{asset_name}",
                        "offset": int(offset),
                        "count": int(flat.size),
                        "frames": int(len(joints)),
                        "display_frames": int(display_frames),
                        "fps": fps,
                        "stride": stride,
                        "minimum": minimum.tolist(),
                        "scale": scale.tolist(),
                    }
                    offset += int(flat.size)
                np.concatenate(values).astype("<u2", copy=False).tofile(
                    assets_dir / asset_name
                )
            print(f"exported {chunk_end}/{len(cases)} cases", flush=True)

    payload = {
        "schema_version": 1,
        "task": "motion_generation",
        "title": args.title,
        "protocol": source_manifest.get("protocol") or source_manifest.get("caption_protocol"),
        "population": len(cases),
        "asset_base_url": args.asset_base_url,
        "reference_label": "Condition sequence" if any(item.get("segments") for item in cases) else "Input caption",
        "output_label": "Displayed methods",
        "motion_methods": [
            {"key": source.key, "label": source.label, "accent": source.accent}
            for source in sources
        ],
        "parents": [-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 12, 16, 17, 18, 19],
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "cases": len(cases),
                "methods": len(sources),
                "stride": stride,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
