#!/usr/bin/env python3
"""Build motion-length AIST++ MP3 clips for the web comparison gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlretrieve

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.models.bailando.audio import extract_bailando_audio_features


DEFAULT_AUDIO_BASE_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/audio/mp3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-base-url", default=DEFAULT_AUDIO_BASE_URL)
    parser.add_argument(
        "--feature-root",
        type=Path,
        help="Optional official 7.5 fps feature directory for time-zero validation.",
    )
    parser.add_argument("--bitrate", default="128k")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _audio_durations(manifest: dict) -> dict[str, float]:
    durations: dict[str, set[float]] = {}
    for case in manifest["cases"]:
        if not case.get("audio"):
            continue
        music_id = Path(case["audio"]).stem
        start = float(case.get("audio_start_seconds", 0.0))
        end = float(case["audio_end_seconds"])
        durations.setdefault(music_id, set()).add(round(end - start, 6))
    resolved = {}
    for music_id, values in durations.items():
        if len(values) != 1:
            raise ValueError(f"Inconsistent clip durations for {music_id}: {sorted(values)}")
        resolved[music_id] = values.pop()
    return resolved


def main() -> None:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError(
                "ffmpeg or the optional imageio-ffmpeg package is required"
            ) from exc
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    durations = _audio_durations(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with tempfile.TemporaryDirectory(prefix="motius-aistpp-audio-") as temp_dir:
        temp = Path(temp_dir)
        for index, (music_id, duration) in enumerate(sorted(durations.items()), start=1):
            output = args.output_dir / f"{music_id}.mp3"
            source_url = f"{args.audio_base_url.rstrip('/')}/{music_id}.mp3"
            if args.overwrite or not output.is_file():
                source = temp / f"{music_id}.mp3"
                urlretrieve(source_url, source)
                subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(source),
                        "-t",
                        f"{duration:.6f}",
                        "-vn",
                        "-ac",
                        "2",
                        "-ar",
                        "44100",
                        "-b:a",
                        args.bitrate,
                        str(output),
                    ],
                    check=True,
                )
            payload = output.read_bytes()
            record = {
                "music_id": music_id,
                "duration_seconds": duration,
                "source_url": source_url,
                "path": f"audio/{output.name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if args.feature_root is not None:
                raw = subprocess.check_output(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(output),
                        "-f",
                        "f32le",
                        "-ac",
                        "1",
                        "-ar",
                        "3840",
                        "pipe:1",
                    ]
                )
                waveform = np.frombuffer(raw, dtype="<f4")
                extracted = extract_bailando_audio_features(
                    waveform, sample_rate=3_840
                )
                official_path = args.feature_root / f"{music_id}.json"
                official = np.asarray(
                    json.loads(official_path.read_text(encoding="utf-8"))["music_array"],
                    dtype=np.float32,
                )
                length = min(len(extracted), len(official))
                correlation = np.corrcoef(
                    extracted[:length, :53].reshape(-1),
                    official[:length, :53].reshape(-1),
                )[0, 1]
                record["feature_prefix_correlation"] = float(correlation)
            records.append(record)
            print(f"[{index}/{len(durations)}] {music_id}: {duration:.3f}s", flush=True)

    audit = {
        "schema_version": 1,
        "source": "AIST Dance Video Database v1.0.0 official MP3 release",
        "source_page": "https://aistdancedb.ongaaccel.jp/database_download/",
        "alignment": "Each clip starts at music time 0 and ends at its gallery motion duration.",
        "feature_alignment_validation": (
            "Correlation compares continuous channels 0:53 against the official "
            "Bailando 7.5 fps feature prefix."
            if args.feature_root is not None
            else None
        ),
        "files": records,
    }
    audit_path = args.output_dir.parent / "audio_manifest.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
