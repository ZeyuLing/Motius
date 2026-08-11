#!/usr/bin/env python3
"""Build a unified Three.js viewer for a native animated body mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

try:
    from tools.build_representation_demo import (
        _quantize_vertex_normals,
        _quantize_vertices,
    )
except ModuleNotFoundError:
    from build_representation_demo import (
        _quantize_vertex_normals,
        _quantize_vertices,
    )


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools/templates/native_mesh_viewer.html"


def build(args: argparse.Namespace) -> Path:
    sequences = [np.load(path) for path in args.input]
    if args.faces_input:
        if args.faces_input.suffix == ".npz":
            with np.load(args.faces_input) as faces_archive:
                faces = np.asarray(
                    faces_archive[args.faces_key],
                    dtype=np.uint32,
                )
        else:
            faces = np.fromfile(args.faces_input, dtype=np.uint32).reshape(
                -1,
                3,
            )
    else:
        faces = np.asarray(sequences[0][args.faces_key], dtype=np.uint32)
    vertices = []
    trajectory = []
    segments = []
    frame = 0
    for index, sequence in enumerate(sequences):
        if args.faces_key in sequence:
            current_faces = np.asarray(
                sequence[args.faces_key],
                dtype=np.uint32,
            )
            if not np.array_equal(faces, current_faces):
                raise ValueError("All native mesh inputs must share topology")
        current = np.asarray(
            sequence[args.vertices_key],
            dtype=np.float32,
        )
        if current.ndim != 3 or current.shape[-1] != 3:
            raise ValueError("vertices must have shape (frames, vertices, 3)")
        vertices.append(current)
        end = frame + len(current)
        caption = (
            args.segment_caption[index]
            if index < len(args.segment_caption)
            else args.caption
        )
        segments.append(
            {
                "caption": caption,
                "start_frame": frame,
                "end_frame": end,
            }
        )
        if args.trajectory_key:
            joints = np.asarray(sequence[args.trajectory_key])
            trajectory.extend(joints[:, args.trajectory_joint].tolist())
        frame = end

    if (
        len(sequences) == 1
        and args.segment_frames_key
        and args.segment_frames_key in sequences[0]
    ):
        intervals = np.asarray(sequences[0][args.segment_frames_key])
        segments = [
            {
                "caption": (
                    args.segment_caption[index]
                    if index < len(args.segment_caption)
                    else args.caption
                ),
                "start_frame": int(interval[0]),
                "end_frame": int(interval[1]),
            }
            for index, interval in enumerate(intervals)
        ]
    if (
        len(sequences) == 1
        and args.condition_root_key
        and args.condition_root_key in sequences[0]
    ):
        root_2d = np.asarray(
            sequences[0][args.condition_root_key],
            dtype=np.float32,
        )
        trajectory = np.stack(
            [
                root_2d[:, 0],
                np.zeros(len(root_2d), dtype=np.float32),
                root_2d[:, 1],
            ],
            axis=-1,
        ).tolist()

    vertices_array = np.concatenate(vertices, axis=0)
    floor_height = float(vertices_array[..., 1].min())
    vertices_array[..., 1] -= floor_height
    if trajectory:
        trajectory = np.asarray(trajectory, dtype=np.float32)
        trajectory[..., 1] -= floor_height
        trajectory_payload = trajectory.tolist()
    else:
        trajectory_payload = []

    quantized, minimum, scale = _quantize_vertices(vertices_array)
    normals = _quantize_vertex_normals(vertices_array, faces)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    quantized.tofile(output / "vertices.u16")
    normals.tofile(output / "normals.i8")
    faces.reshape(-1).tofile(output / "indices.u32")
    shutil.copyfile(TEMPLATE, output / "index.html")

    bounds_min = vertices_array.min(axis=(0, 1))
    bounds_max = vertices_array.max(axis=(0, 1))
    center = (bounds_min + bounds_max) / 2
    span = float(max(
        bounds_max[0] - bounds_min[0],
        bounds_max[2] - bounds_min[2],
    ))
    item = {
        "case_id": args.case,
        "references": [args.caption],
        "segments": segments if len(segments) > 1 else [],
        "condition_intervals": (
            np.asarray(
                sequences[0]["condition_intervals"],
                dtype=np.int32,
            ).tolist()
            if len(sequences) == 1
            and "condition_intervals" in sequences[0]
            else []
        ),
        "frames": int(len(vertices_array)),
        "fps": int(args.fps),
        "trajectory": trajectory_payload,
        "trajectory_on_floor": bool(args.condition_root_key),
        "mesh": {
            "vertices_file": "vertices.u16",
            "normals_file": "normals.i8",
            "indices_file": "indices.u32",
            "vertex_count": int(vertices_array.shape[1]),
            "quantization_min": minimum.tolist(),
            "quantization_scale": scale.tolist(),
            "center": center.tolist(),
            "horizontal_span": span,
            "height": float(bounds_max[1] - bounds_min[1]),
        },
    }
    manifest = {
        "schema_version": 1,
        "method": args.label,
        "representation": args.representation,
        "condition_legend": {
            "conditioned": {"color": "#d95f02"},
            "generated": {"color": "#29a6a1"},
        },
        "cases": [item],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output / "index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vertices-key", default="core_vertices")
    parser.add_argument("--faces-key", default="core_faces")
    parser.add_argument("--faces-input", type=Path)
    parser.add_argument("--trajectory-key")
    parser.add_argument("--trajectory-joint", type=int, default=0)
    parser.add_argument("--condition-root-key")
    parser.add_argument("--segment-frames-key")
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--segment-caption", action="append", default=[])
    parser.add_argument("--representation", required=True)
    parser.add_argument("--fps", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
