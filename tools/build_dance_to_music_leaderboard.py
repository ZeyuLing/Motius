#!/usr/bin/env python3
"""Build the AIST++ dance-to-music leaderboard and audio comparison manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


M2D_CASE_BASE = (
    "https://huggingface.co/spaces/ZeyuLing/"
    "music-to-dance-aistpp-leaderboard/resolve/main/cases/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music-to-dance-manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--generated-audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="Optional deployable copy with generated WAV files.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_cases(source: dict, metrics: dict) -> dict:
    metric_by_id = {item["case_id"]: item for item in metrics["cases"]}
    cases = []
    for item in source["cases"]:
        case_id = item["case_id"]
        if case_id not in metric_by_id:
            continue
        case_metrics = metric_by_id[case_id]
        motion = dict(item["motions"]["gt"])
        motion["asset"] = M2D_CASE_BASE + motion["asset"]
        duration = min(10.0, motion["display_frames"] / float(motion["fps"]))
        references = list(item.get("references") or [])
        cases.append(
            {
                "case_id": case_id,
                "sample_id": case_id,
                "references": references,
                "motion": motion,
                "audio_tracks": [
                    {
                        "key": "reference",
                        "label": "Ground truth music",
                        "asset": M2D_CASE_BASE + item["audio"],
                        "start_seconds": float(item.get("audio_start_seconds", 0.0)),
                        "end_seconds": float(item.get("audio_start_seconds", 0.0))
                        + duration,
                    },
                    {
                        "key": "unimumo",
                        "label": "UniMuMo",
                        "asset": f"audio/generated/{case_id}.wav",
                        "start_seconds": 0.0,
                        "end_seconds": duration,
                    },
                ],
                "beat_metrics": {
                    "reference_beats": case_metrics["reference_beats"],
                    "generated_beats": case_metrics["generated_beats"],
                    "matched_beats": case_metrics["matched_beats"],
                    "coverage": case_metrics["generated_beats"]
                    / max(case_metrics["reference_beats"], 1),
                    "hit": case_metrics["matched_beats"]
                    / max(case_metrics["reference_beats"], 1),
                },
            }
        )
    return {
        "schema_version": 1,
        "task": "dance_to_music",
        "title": "AIST++ Dance-to-Music Audio Comparison",
        "protocol": metrics["dataset"],
        "population": len(cases),
        "representation": "smpl_motion135",
        "body_model_url": M2D_CASE_BASE + "smpl_model/",
        "cases": cases,
    }


def build_results(metrics: dict) -> dict:
    return {
        "schema_version": 1,
        "task": "dance_to_music",
        "dataset": metrics["dataset"],
        "population": metrics["n_samples"],
        "updated": "2026-07-22",
        "protocol": {
            "aggregation": metrics["aggregation"],
            "beat_tolerance_seconds": metrics["tolerance_seconds"],
            "generation": {
                "guidance_scale": 3.0,
                "temperature": 1.0,
                "top_k": 250,
            },
            "note": metrics["protocol_note"],
        },
        "rows": [
            {
                "method": "GT",
                "version": "AIST++ reference music",
                "reference": True,
                "beats_coverage": 1.0,
                "beats_hit": 1.0,
            },
            {
                "method": "UniMuMo",
                "version": "official checkpoint, CFG 3, Motius common40",
                "reference": False,
                "checkpoint": "https://huggingface.co/ZeyuLing/Motius-UniMuMo",
                "model_card": (
                    "https://github.com/ZeyuLing/Motius/blob/main/"
                    "docs/model_zoo/unimumo.md"
                ),
                "paper": "https://arxiv.org/abs/2410.04534",
                "beats_coverage": metrics["beats_coverage"],
                "beats_hit": metrics["beats_hit"],
            },
        ],
        "paper_protocol_rows": [
            {
                "method": "UniMuMo paper",
                "protocol": "D2M-GAN 2-second AIST++ split",
                "beats_coverage": 0.93,
                "beats_hit": 0.884,
            }
        ],
    }


def build_package(
    source_root: Path,
    package_root: Path,
    generated_audio_dir: Path,
    case_ids: list[str],
) -> None:
    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "index.html", package_root / "index.html")
    shutil.copy2(
        source_root / "dance_to_music_results.json",
        package_root / "dance_to_music_results.json",
    )
    shutil.copytree(source_root / "cases", package_root / "cases", dirs_exist_ok=True)
    audio_root = package_root / "cases" / "audio" / "generated"
    audio_root.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        source = generated_audio_dir / f"{case_id}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, audio_root / source.name)


def main() -> None:
    args = parse_args()
    source = json.loads(args.music_to_dance_manifest.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    case_manifest = build_cases(source, metrics)
    results = build_results(metrics)
    write_json(args.output / "cases" / "manifest.json", case_manifest)
    write_json(args.output / "dance_to_music_results.json", results)
    if args.package_dir is not None:
        build_package(
            args.output,
            args.package_dir,
            args.generated_audio_dir,
            [item["case_id"] for item in case_manifest["cases"]],
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
