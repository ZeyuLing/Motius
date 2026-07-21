#!/usr/bin/env python3
"""Export the neutral SMPL body template for Three.js GPU skinning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/body_models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pose-corrective-rank",
        type=int,
        default=64,
        help="Low-rank SMPL pose-corrective basis size; use 0 to omit it.",
    )
    return parser.parse_args()


def low_rank_pose_correctives(
    posedirs: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Factor SMPL pose blend shapes for compact browser-side reconstruction."""

    matrix = np.asarray(posedirs, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"posedirs must be a matrix, got {matrix.shape}")
    if not 1 <= rank <= min(matrix.shape):
        raise ValueError(f"rank must be in [1,{min(matrix.shape)}], got {rank}")
    basis, singular_values, projection = np.linalg.svd(matrix, full_matrices=False)
    retained_basis = basis[:, :rank] * singular_values[:rank]
    retained_projection = projection[:rank]
    energy = float(
        np.square(singular_values[:rank]).sum()
        / np.maximum(np.square(singular_values).sum(), 1e-12)
    )
    return (
        retained_basis.astype(np.float32),
        retained_projection.astype(np.float32),
        energy,
    )


def main() -> None:
    args = parse_args()
    import smplx

    model = smplx.create(
        str(args.model_dir.expanduser().resolve()),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
        use_pca=False,
    )
    model.eval()
    with torch.no_grad():
        result = model(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
        )

    vertices = model.v_template.detach().cpu().numpy().astype("<f4")
    joints = result.joints[0, :24].detach().cpu().numpy().astype("<f4")
    faces = np.asarray(model.faces, dtype="<u2")
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    parents = model.parents[:24].detach().cpu().numpy().astype(np.int16)

    # Three.js supports four skinning influences. SMPL weights are strongly
    # local, so retain and renormalize the four largest values per vertex.
    top = np.argpartition(weights, -4, axis=1)[:, -4:]
    top_weights = np.take_along_axis(weights, top, axis=1)
    order = np.argsort(top_weights, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1).astype(np.uint8)
    top_weights = np.take_along_axis(top_weights, order, axis=1)
    retained = top_weights.sum(axis=1, keepdims=True)
    top_weights = top_weights / np.maximum(retained, 1e-8)
    quantized_weights = np.rint(top_weights * 65535.0).clip(0, 65535).astype("<u2")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    vertices.tofile(output / "vertices.f32")
    joints.tofile(output / "joints.f32")
    faces.tofile(output / "faces.u16")
    top.tofile(output / "skin_indices.u8")
    quantized_weights.tofile(output / "skin_weights.u16")
    corrective_metadata = None
    if args.pose_corrective_rank:
        posedirs = model.posedirs.detach().cpu().numpy().T.astype(np.float32)
        basis, projection, energy = low_rank_pose_correctives(
            posedirs, args.pose_corrective_rank
        )
        rank = int(args.pose_corrective_rank)
        basis = basis.reshape(len(vertices), 3, rank).transpose(2, 0, 1)
        packed_basis = np.zeros((rank, len(vertices), 4), dtype="<f2")
        packed_basis[..., :3] = basis.astype("<f2")
        packed_basis.tofile(output / "pose_corrective_basis.rgba16f")
        projection.astype("<f4").tofile(output / "pose_corrective_projection.f32")
        corrective_metadata = {
            "rank": rank,
            "feature_dimensions": int(projection.shape[1]),
            "basis_layout": "rank,vertex,rgba",
            "retained_energy": energy,
            "files": {
                "basis": "pose_corrective_basis.rgba16f",
                "projection": "pose_corrective_projection.f32",
            },
        }
    metadata = {
        "schema_version": 2 if corrective_metadata else 1,
        "model": "SMPL neutral beta=0",
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "joints": 24,
        "parents": parents.tolist(),
        "files": {
            "vertices": "vertices.f32",
            "joints": "joints.f32",
            "faces": "faces.u16",
            "skin_indices": "skin_indices.u8",
            "skin_weights": "skin_weights.u16",
        },
        "top4_retained_weight": {
            "mean": float(retained.mean()),
            "minimum": float(retained.min()),
            "p01": float(np.percentile(retained, 1)),
        },
    }
    if corrective_metadata:
        metadata["pose_correctives"] = corrective_metadata
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
