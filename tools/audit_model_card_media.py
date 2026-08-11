#!/usr/bin/env python3
"""Audit public Model Card videos against the Motius media contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENTS = ROOT / "docs/model_zoo/video_attachments.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/model_card_video_uploads"
RENDER_PROFILE = "motius-threejs-floor-v1"
MIN_CONTAINER_FPS = 19.0
MIN_VISUAL_CHANGE_FPS = 5.0


@dataclass(frozen=True)
class MediaProbe:
    frames: int
    duration_seconds: float
    container_fps: float
    visual_change_fps: float
    has_audio: bool
    audio_duration_seconds: float
    black_fraction: float


def _audio_required(source: str) -> bool:
    path = Path(source)
    package = path.parts[2] if len(path.parts) > 2 else ""
    name = path.name.lower()
    return package in {"bailando", "edge", "tm2d", "unimumo"} and (
        "aistpp" in name or "dance_to_music" in name
    )


def _has_audio(path: Path) -> bool:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _audio_duration_seconds(path: Path) -> float:
    sample_rate = 16_000
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return 0.0
    return len(result.stdout) / (sample_rate * 2)


def _frame_hashes(path: Path) -> list[str]:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "framemd5",
        "-",
    ]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in result.stdout.splitlines()
        if line and not line.startswith("#")
    ]


def _probe(path: Path) -> MediaProbe:
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    hashes = _frame_hashes(path)
    if frames != len(hashes):
        raise RuntimeError(
            f"decoded frame count mismatch: {frames} != {len(hashes)}"
        )
    changes = 1 + sum(
        left != right
        for left, right in zip(hashes, hashes[1:])
    )
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    first_frame = np.frombuffer(next(reader), dtype=np.uint8).reshape(
        metadata["size"][1],
        metadata["size"][0],
        3,
    )
    reader.close()
    black_fraction = float(np.mean(np.max(first_frame, axis=-1) < 8))
    has_audio = _has_audio(path)
    return MediaProbe(
        frames=frames,
        duration_seconds=duration,
        container_fps=frames / duration,
        visual_change_fps=changes / duration,
        has_audio=has_audio,
        audio_duration_seconds=(
            _audio_duration_seconds(path)
            if has_audio
            else 0.0
        ),
        black_fraction=black_fraction,
    )


def _video_path(source: str, output_root: Path) -> Path:
    return output_root / Path(source).with_suffix(".mp4")


def _audit_one(source: str, output_root: Path) -> tuple[MediaProbe, dict]:
    video = _video_path(source, output_root)
    metadata_path = video.with_suffix(".render.json")
    if not video.is_file():
        raise RuntimeError(f"missing unified MP4: {video}")
    if not metadata_path.is_file():
        raise RuntimeError(f"missing render metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("render_backend") != "threejs":
        raise RuntimeError("render_backend must be threejs")
    if metadata.get("render_profile") != RENDER_PROFILE:
        raise RuntimeError(f"render_profile must be {RENDER_PROFILE}")
    if metadata.get("frame_step") != 1:
        raise RuntimeError("frame_step must be 1")
    if not metadata.get("floor"):
        raise RuntimeError("unified floor metadata is missing")
    if not metadata.get("representation"):
        raise RuntimeError("display representation is missing")

    probe = _probe(video)
    if metadata.get("frames") != probe.frames:
        raise RuntimeError(
            "render metadata frame count does not match decoded video"
        )
    if probe.container_fps < MIN_CONTAINER_FPS:
        raise RuntimeError(
            f"container frame rate is only {probe.container_fps:.2f} fps"
        )
    if probe.visual_change_fps < MIN_VISUAL_CHANGE_FPS:
        raise RuntimeError(
            f"visual content changes at only "
            f"{probe.visual_change_fps:.2f} fps"
        )
    if probe.black_fraction > 0.15:
        raise RuntimeError(
            f"first frame is {100 * probe.black_fraction:.1f}% black"
        )
    if _audio_required(source):
        if not probe.has_audio:
            raise RuntimeError("audio-task preview has no audio stream")
        audio_gap = abs(
            probe.audio_duration_seconds - probe.duration_seconds
        )
        if audio_gap > 0.25:
            raise RuntimeError(
                "audio/video duration mismatch: "
                f"{probe.audio_duration_seconds:.2f}s audio versus "
                f"{probe.duration_seconds:.2f}s video"
            )
    return probe, metadata


def audit(output_root: Path, sources: list[str]) -> tuple[int, list[str]]:
    passed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {
            executor.submit(_audit_one, source, output_root): source
            for source in sources
        }
        for future in as_completed(pending):
            source = pending[future]
            try:
                probe, metadata = future.result()
                passed += 1
                print(
                    f"ok {source}: {probe.frames} frames, "
                    f"{probe.container_fps:.2f} fps, "
                    f"{probe.visual_change_fps:.2f} visual changes/s, "
                    f"audio={'yes' if probe.has_audio else 'no'}, "
                    f"{metadata['representation']}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"{source}: {exc}")
                print(f"failed {source}: {exc}", flush=True)
    return passed, sorted(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--match")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    payload = json.loads(ATTACHMENTS.read_text(encoding="utf-8"))
    sources = sorted(payload["videos"])
    if args.match:
        sources = [source for source in sources if args.match in source]
    passed, failures = audit(args.output_root, sources)
    print(f"{passed}/{len(sources)} media previews satisfy the contract")
    if failures:
        print("\n".join(failures))
        if not args.allow_failures:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
