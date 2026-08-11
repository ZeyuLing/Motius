#!/usr/bin/env python3
"""Build the official D2M-GAN AIST++ leaderboard and SMPL/audio viewer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from smpl_gallery_assets import encode_motion135, load_motion135


PAPER_ROWS = (
    ("Dance2Music", 0.835, 0.824),
    ("Foley", 0.741, 0.694),
    ("CMT", 0.855, 0.835),
    ("D2M-GAN", 0.882, 0.847),
    ("CDCD", 0.939, 0.907),
    ("UniMuMo paper", 0.930, 0.884),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--motion135-dir", required=True, type=Path)
    parser.add_argument("--generated-audio-dir", required=True, type=Path)
    parser.add_argument("--reference-audio-dir", required=True, type=Path)
    parser.add_argument(
        "--body-model-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "leaderboards"
            / "hf_space_music_to_dance"
            / "cases"
            / "smpl_model"
        ),
        help="SMPL Web rig copied into the self-contained viewer package.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="Optional deployable copy including generated and reference audio.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_cases(
    manifest: dict,
    metrics: dict,
    motion135_dir: Path,
    output: Path,
) -> dict:
    metric_by_id = {item["case_id"]: item for item in metrics["cases"]}
    cases = []
    asset_root = output / "cases" / "motions"
    if asset_root.exists():
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    for item in manifest["cases"]:
        case_id = str(item["case_id"])
        case_metrics = metric_by_id.get(case_id)
        if case_metrics is None:
            raise KeyError(f"Missing beat metrics for {case_id}")
        source = motion135_dir / f"{case_id}.npz"
        if not source.is_file():
            source = motion135_dir / f"{case_id}.npy"
        motion = load_motion135(source)
        encoded, descriptor = encode_motion135(motion, stride=1)
        motion_asset = asset_root / f"{case_id}.smpl"
        motion_asset.write_bytes(encoded)
        descriptor.update(
            {
                "asset": f"motions/{case_id}.smpl",
                "fps": 30.0,
                "translation_offset": 0,
                "rotation_offset": descriptor["translation_count"] * 2,
            }
        )
        music_id = str(item["music_id"])
        start = float(item["reference_start_seconds"])
        cases.append(
            {
                "case_id": case_id,
                "sample_id": case_id,
                "references": [music_id, f"{start:.0f}-{start + 2:.0f} s"],
                "motion": descriptor,
                "audio_tracks": [
                    {
                        "key": "reference",
                        "label": "Ground truth music",
                        "asset": f"audio/reference/{music_id}.mp3",
                        "start_seconds": start,
                        "end_seconds": start + 2.0,
                    },
                    {
                        "key": "unimumo",
                        "label": "UniMuMo",
                        "asset": f"audio/generated/{case_id}.wav",
                        "start_seconds": 0.0,
                        "end_seconds": 2.0,
                    },
                ],
                "beat_metrics": {
                    "reference_beats": case_metrics["reference_beat_bins"],
                    "generated_beats": case_metrics["generated_beat_bins"],
                    "matched_beats": case_metrics["hit_beat_bins"],
                    "coverage": case_metrics["beat_count_ratio"],
                    "hit": case_metrics["beat_hit_rate"],
                },
            }
        )
    if len(cases) != int(metrics["n_samples"]):
        raise ValueError(
            f"Built {len(cases)} cases for metrics population {metrics['n_samples']}"
        )
    return {
        "schema_version": 2,
        "task": "dance_to_music",
        "title": "AIST++ Dance-to-Music Audio Comparison",
        "protocol": metrics["dataset"],
        "population": len(cases),
        "representation": "smpl_motion135",
        "body_model_url": "smpl_model/",
        "cases": cases,
    }


def build_results(metrics: dict) -> dict:
    paper_rows = [
        {
            "method": method,
            "version": "paper Table 2",
            "source": "paper",
            "beat_count_ratio": ratio,
            "beat_hit_rate": hit,
        }
        for method, ratio, hit in PAPER_ROWS
    ]
    return {
        "schema_version": 2,
        "task": "dance_to_music",
        "dataset": metrics["dataset"],
        "population": metrics["n_samples"],
        "updated": "2026-07-22",
        "protocol": {
            "aggregation": metrics["aggregation"],
            "beat_detection": metrics["protocol"],
            "generation": {
                "guidance_scale": 3.0,
                "temperature": 1.0,
                "top_k": 250,
                "prompt": "none",
            },
            "coverage_note": metrics["coverage_note"],
        },
        "rows": [
            {
                "method": "GT",
                "version": "AIST++ reference music",
                "source": "reference",
                "reference": True,
                "beat_count_ratio": 1.0,
                "beat_hit_rate": 1.0,
            },
            {
                "method": "UniMuMo",
                "version": "Motius reproduction, official checkpoint, CFG 3",
                "source": "motius",
                "reference": False,
                "checkpoint": "https://huggingface.co/ZeyuLing/Motius-UniMuMo",
                "model_card": (
                    "https://github.com/ZeyuLing/Motius/blob/main/"
                    "docs/model_zoo/unimumo.md"
                ),
                "paper": "https://arxiv.org/abs/2410.04534",
                "beat_count_ratio": metrics["beat_count_ratio"],
                "beat_hit_rate": metrics["beat_hit_rate"],
            },
        ],
        "paper_rows": paper_rows,
    }


def copy_audio(
    package_root: Path,
    manifest: dict,
    generated_audio_dir: Path,
    reference_audio_dir: Path,
) -> None:
    generated_root = package_root / "cases" / "audio" / "generated"
    reference_root = package_root / "cases" / "audio" / "reference"
    generated_root.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)
    music_ids = set()
    for item in manifest["cases"]:
        case_id = str(item["case_id"])
        music_ids.add(str(item["music_id"]))
        source = generated_audio_dir / f"{case_id}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, generated_root / source.name)
    for music_id in sorted(music_ids):
        source = reference_audio_dir / f"{music_id}.mp3"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, reference_root / source.name)


def build_package(
    source_root: Path,
    package_root: Path,
    manifest: dict,
    generated_audio_dir: Path,
    reference_audio_dir: Path,
) -> None:
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "README.md", package_root / "README.md")
    shutil.copy2(source_root / "index.html", package_root / "index.html")
    shutil.copy2(
        source_root / "dance_to_music_results.json",
        package_root / "dance_to_music_results.json",
    )
    shutil.copytree(source_root / "cases", package_root / "cases", dirs_exist_ok=True)
    copy_audio(package_root, manifest, generated_audio_dir, reference_audio_dir)


def copy_body_model(body_model_dir: Path, output: Path) -> None:
    if not (body_model_dir / "model.json").is_file():
        raise FileNotFoundError(
            f"SMPL Web rig is incomplete or missing: {body_model_dir}"
        )
    shutil.copytree(
        body_model_dir,
        output / "cases" / "smpl_model",
        dirs_exist_ok=True,
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    case_manifest = build_cases(manifest, metrics, args.motion135_dir, args.output)
    results = build_results(metrics)
    copy_body_model(args.body_model_dir, args.output)
    write_json(args.output / "cases" / "manifest.json", case_manifest)
    write_json(args.output / "dance_to_music_results.json", results)
    if args.package_dir is not None:
        build_package(
            args.output,
            args.package_dir,
            manifest,
            args.generated_audio_dir,
            args.reference_audio_dir,
        )
    print(
        json.dumps(
            {
                "cases": len(case_manifest["cases"]),
                "output": str(args.output),
                "package": str(args.package_dir) if args.package_dir else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
