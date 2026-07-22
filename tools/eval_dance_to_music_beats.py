#!/usr/bin/env python3
"""Evaluate generated music against AIST++ reference beats."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import librosa
import numpy as np


DEFAULT_AUDIO_BASE_URL = (
    "https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/"
    "cases/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reference-audio-root", type=Path)
    parser.add_argument("--audio-cache", required=True, type=Path)
    parser.add_argument("--audio-base-url", default=DEFAULT_AUDIO_BASE_URL)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-seconds", type=float, default=0.1)
    return parser.parse_args()


def cached_audio(args: argparse.Namespace, relative_path: str) -> Path:
    relative = Path(relative_path)
    if args.reference_audio_root is not None:
        source = args.reference_audio_root / relative
        if source.is_file():
            return source
    destination = args.audio_cache / relative.name
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = args.audio_base_url.rstrip("/") + "/" + relative.as_posix()
        urllib.request.urlretrieve(url, destination)
    return destination


def load_audio(path: Path, target_sample_rate: int = 32_000) -> np.ndarray:
    try:
        waveform, _ = librosa.load(path, sr=target_sample_rate, mono=True)
        return np.asarray(waveform, dtype=np.float32)
    except Exception as librosa_error:  # noqa: BLE001
        try:
            import av

            chunks = []
            with av.open(str(path)) as container:
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(
                    format="fltp", layout="mono", rate=target_sample_rate
                )
                for frame in container.decode(stream):
                    for converted in resampler.resample(frame):
                        chunks.append(converted.to_ndarray().reshape(-1))
                for converted in resampler.resample(None):
                    chunks.append(converted.to_ndarray().reshape(-1))
            if not chunks:
                raise ValueError(f"No audio frames decoded from {path}")
            return np.concatenate(chunks).astype(np.float32)
        except Exception as av_error:  # noqa: BLE001
            raise RuntimeError(
                f"Could not decode {path} with librosa or PyAV: "
                f"{librosa_error}; {av_error}"
            ) from av_error


def audio_beats(
    waveform: np.ndarray,
    start: float,
    duration: float,
    sample_rate: int = 32_000,
) -> np.ndarray:
    first = max(0, int(round(start * sample_rate)))
    last = min(len(waveform), first + int(round(duration * sample_rate)))
    waveform = waveform[first:last]
    _, frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, units="frames")
    return librosa.frames_to_time(frames, sr=sample_rate)


def unique_beat_matches(
    reference: np.ndarray, prediction: np.ndarray, tolerance: float
) -> int:
    candidates = sorted(
        (
            (abs(float(gt - pred)), gt_index, pred_index)
            for gt_index, gt in enumerate(reference)
            for pred_index, pred in enumerate(prediction)
            if abs(float(gt - pred)) <= tolerance
        ),
        key=lambda item: item[0],
    )
    used_gt: set[int] = set()
    used_prediction: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_prediction:
            used_gt.add(gt_index)
            used_prediction.add(pred_index)
    return len(used_gt)


def main() -> None:
    args = parse_args()
    cases = json.loads(args.case_manifest.read_text(encoding="utf-8"))["cases"]
    rows = []
    total_reference = 0
    total_prediction = 0
    total_matches = 0
    reference_waveforms: dict[Path, np.ndarray] = {}
    for case in cases:
        case_id = str(case.get("case_id") or case.get("sample_id"))
        prediction_path = args.predictions / f"{case_id}.wav"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        start = max(0.0, float(case.get("audio_start_seconds") or 0.0))
        end = float(case.get("audio_end_seconds") or start + 10.0)
        duration = min(10.0, end - start)
        reference_path = cached_audio(args, str(case["audio"]))
        if reference_path not in reference_waveforms:
            reference_waveforms[reference_path] = load_audio(reference_path)
        reference = audio_beats(reference_waveforms[reference_path], start, duration)
        prediction = audio_beats(load_audio(prediction_path), 0.0, duration)
        matches = unique_beat_matches(reference, prediction, args.tolerance_seconds)
        total_reference += len(reference)
        total_prediction += len(prediction)
        total_matches += matches
        rows.append(
            {
                "case_id": case_id,
                "reference_beats": len(reference),
                "generated_beats": len(prediction),
                "matched_beats": matches,
            }
        )
    if total_reference == 0:
        raise ValueError("No reference beats were detected")
    result = {
        "schema_version": 1,
        "task": "dance_to_music",
        "method": "UniMuMo",
        "dataset": "AIST++ shared crossmodal 40-case package",
        "n_samples": len(rows),
        "beats_coverage": total_prediction / total_reference,
        "beats_hit": total_matches / total_reference,
        "tolerance_seconds": args.tolerance_seconds,
        "aggregation": "micro average over reference beats",
        "protocol_note": (
            "Motius common-case diagnostic. UniMuMo paper Table 2 uses the "
            "D2M-GAN 2-second split and must not be compared directly."
        ),
        "reference_beats": total_reference,
        "generated_beats": total_prediction,
        "matched_beats": total_matches,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
