#!/usr/bin/env python3
"""Stage selected licensed 3DPW clips for a local-only comparison gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/3DPW/test_imageFiles.tar",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=ROOT / "data/3DPW/sequenceFiles/test",
    )
    parser.add_argument(
        "--viewer-dir",
        type=Path,
        default=ROOT
        / (
            "outputs/visualization/monocular_capture/3dpw_test/"
            "gem_smpl/interactive"
        ),
    )
    parser.add_argument(
        "--cached-video-root",
        type=Path,
        default=ROOT
        / (
            "outputs/evaluation/monocular_capture/3dpw_test/"
            "gem_x/_official_runs"
        ),
        help="Optional existing browser-compatible videos from a licensed run.",
    )
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def video_properties(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def expected_frames(annotation_dir: Path, sequence: str) -> int:
    with (annotation_dir / f"{sequence}.pkl").open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")
    if str(payload["sequence"]) != sequence:
        raise ValueError(f"Annotation sequence mismatch for {sequence}.")
    return len(np.asarray(payload["img_frame_ids"]))


def extract_selected(
    archive: Path,
    staging: Path,
    expected: dict[str, int],
) -> None:
    counts = {sequence: 0 for sequence in expected}
    with tarfile.open(archive, "r") as source:
        for member in source:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            sequence = next((value for value in expected if value in parts), None)
            if sequence is None:
                continue
            name = parts[-1]
            if not name.startswith("image_") or not name.endswith(".jpg"):
                continue
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Unable to read {member.name}.")
            destination = staging / sequence / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)
            counts[sequence] += 1
    for sequence, frames in expected.items():
        names = sorted(path.name for path in (staging / sequence).glob("image_*.jpg"))
        wanted = [f"image_{index:05d}.jpg" for index in range(frames)]
        if counts[sequence] != frames or names != wanted:
            raise RuntimeError(
                f"{sequence}: extracted {counts[sequence]} frames; expected {frames}."
            )


def mux_clip(images: Path, output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:g}",
            "-start_number",
            "0",
            "-i",
            str(images / "image_%05d.jpg"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(output),
        ],
        check=True,
    )


def transcode_cached_video(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            "scale=960:-2:force_original_aspect_ratio=decrease",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-tune",
            "fastdecode",
            "-crf",
            "27",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def mux_selected_from_tar(
    archive: Path,
    outputs: dict[str, Path],
    expected: dict[str, int],
    fps: float,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    members: dict[str, dict[int, bytes]] = {
        sequence: {} for sequence in outputs
    }
    with tarfile.open(archive, "r|") as source:
        for member in source:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            sequence = next((value for value in outputs if value in parts), None)
            if sequence is None:
                continue
            name = parts[-1]
            if not name.startswith("image_") or not name.endswith(".jpg"):
                continue
            try:
                index = int(name[6:11])
            except ValueError as exc:
                raise ValueError(f"Unexpected frame name: {member.name}") from exc
            if index in members[sequence]:
                raise RuntimeError(f"Duplicate frame {sequence}/{name}.")
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Unable to read {member.name}.")
            members[sequence][index] = extracted.read()
    for sequence, frames in expected.items():
        if sorted(members[sequence]) != list(range(frames)):
            raise RuntimeError(
                f"{sequence}: indexed {len(members[sequence])} frames; "
                f"expected contiguous 0..{frames - 1}."
            )
    print(
        "Indexed "
        + ", ".join(f"{sequence}={len(value)}" for sequence, value in members.items()),
        flush=True,
    )

    for sequence, output in outputs.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "image2pipe",
                    "-framerate",
                    f"{fps:g}",
                    "-vcodec",
                    "mjpeg",
                    "-i",
                    "pipe:0",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    "-y",
                    str(output),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin pipe is unavailable.")
            for index in range(expected[sequence]):
                process.stdin.write(members[sequence][index])
                if (index + 1) % 300 == 0:
                    print(
                        f"{sequence}: streamed {index + 1}/{expected[sequence]}",
                        flush=True,
                    )
            process.stdin.close()
            stderr = (
                process.stderr.read().decode(errors="replace")
                if process.stderr
                else ""
            )
            return_code = process.wait()
        except Exception:
            if process.poll() is None:
                process.kill()
            raise
        if return_code != 0:
            raise RuntimeError(
                f"{sequence}: ffmpeg exited {return_code}: {stderr[-2000:]}"
            )
        print(f"{sequence}: encoded {expected[sequence]} frames", flush=True)


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    annotations = args.annotation_dir.expanduser().resolve()
    viewer = args.viewer_dir.expanduser().resolve()
    viewer.relative_to((ROOT / "outputs").resolve())
    if not archive.is_file():
        raise FileNotFoundError(archive)
    manifest_path = viewer / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    available = {case["case_id"] for case in manifest["cases"]}
    requested = list(dict.fromkeys(args.sequence))
    missing = set(requested) - available
    if missing:
        raise ValueError(f"Sequences are absent from the viewer: {sorted(missing)}")

    expected = {
        sequence: expected_frames(annotations, sequence)
        for sequence in requested
    }
    records = {}
    staging = ROOT / "outputs/tmp/monocular_capture_gallery/3dpw_images"
    shutil.rmtree(staging, ignore_errors=True)
    video_paths = {
        sequence: viewer / "assets/video" / f"{sequence}.mp4"
        for sequence in requested
    }
    cache_root = args.cached_video_root.expanduser().resolve()
    source_sizes = {}
    for sequence, output in video_paths.items():
        cached = cache_root / sequence / sequence / f"{sequence}.mp4"
        if not cached.is_file():
            continue
        _, _, source_width, source_height = video_properties(cached)
        source_sizes[sequence] = (source_width, source_height)
        if args.overwrite or not output.exists() or output.stat().st_size == 0:
            transcode_cached_video(cached, output)
            print(f"{sequence}: transcoded cached input video", flush=True)
    pending = {
        sequence: output
        for sequence, output in video_paths.items()
        if not output.exists() or output.stat().st_size == 0
    }
    if pending:
        mux_selected_from_tar(
            archive,
            pending,
            {sequence: expected[sequence] for sequence in pending},
            args.fps,
        )
    for sequence in requested:
        output = video_paths[sequence]
        frames, fps, video_width, video_height = video_properties(output)
        if frames != expected[sequence] or not np.isclose(fps, args.fps):
            raise RuntimeError(
                f"{sequence}: MP4 reports {frames} frames at {fps} FPS."
            )
        source_width, source_height = source_sizes.get(
            sequence,
            (video_width, video_height),
        )
        relative = output.relative_to(viewer)
        records[sequence] = {
            "video": relative.as_posix(),
            "video_frames": frames,
            "video_fps": fps,
            "video_width": video_width,
            "video_height": video_height,
            "bbox_coordinate_width": source_width,
            "bbox_coordinate_height": source_height,
            "video_sha256": sha256(output),
            "video_distribution": "local_only_licensed_3dpw",
        }

    for case in manifest["cases"]:
        case.update(records[case["case_id"]])
    manifest["contains_licensed_input_video"] = True
    manifest["distribution"] = "local_only_licensed_3dpw"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "viewer": str(viewer),
                "videos": len(records),
                "bytes": sum((viewer / item["video"]).stat().st_size for item in records.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
