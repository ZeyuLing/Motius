#!/usr/bin/env python3
"""Losslessly mux staged 3DPW Test JPEG frames into per-sequence videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _video_properties(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    annotations = sorted((root / "sequenceFiles/test").glob("*.pkl"))
    if len(annotations) != 24:
        raise RuntimeError(
            f"Expected 24 official 3DPW Test annotations, found {len(annotations)}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records = []

    for annotation in annotations:
        with annotation.open("rb") as stream:
            payload = pickle.load(stream, encoding="latin1")
        sequence = str(payload["sequence"])
        frame_ids = np.asarray(payload["img_frame_ids"])
        expected_frames = len(frame_ids)
        images = sorted((root / "imageFiles" / sequence).glob("image_*.jpg"))
        if len(images) != expected_frames:
            raise RuntimeError(
                f"{sequence}: expected {expected_frames} frames, found {len(images)}."
            )
        expected_names = [
            f"image_{index:05d}.jpg" for index in range(expected_frames)
        ]
        if [path.name for path in images] != expected_names:
            raise RuntimeError(f"{sequence}: image filenames are not contiguous.")

        # Preserve the official JPEG bitstreams while using an extension
        # accepted by all upstream demos (GEM-SMPL rejects Matroska inputs).
        output = output_dir / f"{sequence}.avi"
        if args.overwrite or not output.exists():
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    f"{args.fps:g}",
                    "-start_number",
                    "0",
                    "-i",
                    str(images[0].parent / "image_%05d.jpg"),
                    "-c:v",
                    "copy",
                    "-an",
                    "-y",
                    str(output),
                ],
                check=True,
            )
        actual_frames, actual_fps = _video_properties(output)
        if actual_frames != expected_frames or not np.isclose(
            actual_fps,
            args.fps,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{sequence}: muxed video reports {actual_frames} frames at "
                f"{actual_fps} FPS."
            )
        records.append(
            {
                "sequence_id": sequence,
                "video": output.name,
                "frames": actual_frames,
                "fps": actual_fps,
                "video_sha256": _sha256(output),
                "annotation_sha256": _sha256(annotation),
                "codec": "MJPEG stream copy from official JPEG frames",
            }
        )

    manifest = {
        "schema_revision": "motius_3dpw_test_videos_v1",
        "source": "official 3DPW Test JPEG frames",
        "lossless_mux": True,
        "population": len(records),
        "total_frames": sum(record["frames"] for record in records),
        "videos": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Materialized {manifest['population']} videos / "
        f"{manifest['total_frames']} frames under {output_dir}."
    )


if __name__ == "__main__":
    main()
