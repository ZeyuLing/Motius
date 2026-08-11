#!/usr/bin/env python3
"""Build a unified Three.js viewer for an articulated native robot mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools/templates/native_articulated_viewer.html"


def _quat_rotate(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    vector = quaternion[:3]
    scalar = quaternion[3]
    return (
        points
        + 2 * scalar * np.cross(vector, points)
        + 2 * np.cross(vector, np.cross(vector, points))
    )


def _bounds(
    vertices: np.ndarray,
    transforms: np.ndarray,
    parts: list,
) -> tuple:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for part in parts:
        local = vertices[
            part["vertex_offset"]:
            part["vertex_offset"] + part["vertex_count"]
        ]
        low = local.min(axis=0)
        high = local.max(axis=0)
        corners = np.asarray(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ],
            dtype=np.float32,
        )
        for transform in transforms[:, part["transform_index"]]:
            world = _quat_rotate(corners, transform[3:7]) + transform[:3]
            minimum = np.minimum(minimum, world.min(axis=0))
            maximum = np.maximum(maximum, world.max(axis=0))
    return minimum, maximum


def _simplify_parts(
    vertices: np.ndarray,
    indices: np.ndarray,
    parts: list,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, list]:
    import trimesh

    output_vertices = []
    output_indices = []
    output_parts = []
    vertex_offset = 0
    index_offset = 0
    for source_part in parts:
        part = dict(source_part)
        local_vertices = vertices[
            part["vertex_offset"]:
            part["vertex_offset"] + part["vertex_count"]
        ]
        local_faces = indices[
            part["index_offset"]:
            part["index_offset"] + part["index_count"]
        ].reshape(-1, 3)
        if len(local_faces) > max_faces:
            mesh = trimesh.Trimesh(
                vertices=local_vertices,
                faces=local_faces,
                process=False,
            ).simplify_quadric_decimation(face_count=max_faces)
            local_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            local_faces = np.asarray(mesh.faces, dtype=np.uint32)
        else:
            local_vertices = np.asarray(local_vertices, dtype=np.float32)
            local_faces = np.asarray(local_faces, dtype=np.uint32)
        part.update(
            {
                "vertex_offset": vertex_offset,
                "vertex_count": int(len(local_vertices)),
                "index_offset": index_offset,
                "index_count": int(local_faces.size),
            }
        )
        output_vertices.append(local_vertices)
        output_indices.append(local_faces.reshape(-1))
        output_parts.append(part)
        vertex_offset += len(local_vertices)
        index_offset += local_faces.size
    return (
        np.concatenate(output_vertices, axis=0),
        np.concatenate(output_indices, axis=0),
        output_parts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Keep only the first N frames in the public preview asset.",
    )
    parser.add_argument(
        "--max-part-faces",
        type=int,
        help="Decimate each rigid mesh part for a lightweight web preview.",
    )
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    spec = source["motionbricks"]
    vertices_path = args.source_root / Path(spec["vertices"]).name
    indices_path = args.source_root / Path(spec["indices"]).name
    transforms_path = args.source_root / Path(spec["transforms"]).name
    vertices = np.fromfile(vertices_path, dtype=np.float32).reshape(-1, 3)
    indices = np.fromfile(indices_path, dtype=np.uint32)
    parts = spec["metadata"]
    if args.max_part_faces is not None:
        vertices, indices, parts = _simplify_parts(
            vertices,
            indices,
            parts,
            args.max_part_faces,
        )
    transforms = np.fromfile(
        transforms_path,
        dtype=np.float32,
    ).reshape(spec["frames"], spec["transform_count"], 7)
    if args.max_frames is not None:
        transforms = transforms[: args.max_frames]
    minimum, maximum = _bounds(vertices, transforms, parts)
    floor_offset = float(-minimum[1] + 0.002)
    minimum[1] += floor_offset
    maximum[1] += floor_offset
    center = (minimum + maximum) / 2
    horizontal_span = float(
        max(maximum[0] - minimum[0], maximum[2] - minimum[2])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    vertices.astype(np.float32).tofile(assets / "vertices.f32")
    indices.astype(np.uint32).tofile(assets / "indices.u32")
    transforms.astype(np.float32).tofile(assets / "transforms.f32")

    manifest = {
        "schema_version": 1,
        "method": args.label,
        "representation": args.representation,
        "cases": [
            {
                "case_id": args.case,
                "references": [args.caption],
                "frames": int(transforms.shape[0]),
                "fps": int(spec["fps"]),
                "mesh": {
                    "vertices_file": "assets/vertices.f32",
                    "indices_file": "assets/indices.u32",
                    "transforms_file": "assets/transforms.f32",
                    "transform_count": int(spec["transform_count"]),
                    "parts": parts,
                    "floor_offset": floor_offset,
                    "center": [
                        float(center[0]),
                        0.0,
                        float(center[2]),
                    ],
                    "height": float(maximum[1] - minimum[1]),
                    "horizontal_span": horizontal_span,
                },
            }
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(TEMPLATE, args.output_dir / "index.html")
    print((args.output_dir / "index.html").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
