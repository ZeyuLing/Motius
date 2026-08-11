#!/usr/bin/env python3
"""Evaluate UniMuMo with the exact D2M-GAN beat-score formula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.dance_to_music import (
    D2MGAN_SAMPLE_RATE,
    aggregate_d2mgan_beat_scores,
    d2mgan_beat_bins,
    d2mgan_beat_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--reference-audio-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_audio(path: Path) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=D2MGAN_SAMPLE_RATE, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prediction_root = args.prediction_root or args.manifest.parent
    references: dict[str, np.ndarray] = {}
    rows = []
    scores = []
    for case in manifest["cases"]:
        music_id = str(case["music_id"])
        if music_id not in references:
            references[music_id] = load_audio(
                args.reference_audio_root / f"{music_id}.mp3"
            )
        start = int(
            round(float(case["reference_start_seconds"]) * D2MGAN_SAMPLE_RATE)
        )
        samples = 2 * D2MGAN_SAMPLE_RATE
        reference = references[music_id][start : start + samples]
        generated = load_audio(prediction_root / str(case["generated_audio"]))[:samples]
        if len(reference) != samples or len(generated) < D2MGAN_SAMPLE_RATE:
            raise ValueError(f"Invalid protocol audio length for {case['case_id']}")
        reference_bins = d2mgan_beat_bins(reference)
        generated_bins = d2mgan_beat_bins(generated)
        score = d2mgan_beat_score(reference_bins, generated_bins)
        scores.append(score)
        rows.append(
            {
                "case_id": case["case_id"],
                "beat_count_ratio": score.beat_count_ratio,
                "beat_hit_rate": score.beat_hit_rate,
                "reference_beat_bins": score.reference_beat_bins,
                "generated_beat_bins": score.generated_beat_bins,
                "hit_beat_bins": score.hit_beat_bins,
            }
        )
    aggregate = aggregate_d2mgan_beat_scores(scores)
    result = {
        "schema_version": 2,
        "task": "dance_to_music",
        "method": manifest.get("method", "UniMuMo"),
        "dataset": manifest["dataset"],
        "n_samples": aggregate["n_samples"],
        "beats_coverage": aggregate["beat_count_ratio"],
        "beat_count_ratio": aggregate["beat_count_ratio"],
        "beats_hit": aggregate["beat_hit_rate"],
        "beat_hit_rate": aggregate["beat_hit_rate"],
        "aggregation": "macro average over two-second clips",
        "protocol": "D2M-GAN Beats_Scores (22.05 kHz onset detection, one-second bins)",
        "coverage_note": (
            "The paper's Beats Coverage is an unbounded generated/reference "
            "beat-count ratio. Values above 100% indicate excess beats."
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
