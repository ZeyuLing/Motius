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
    return parser.parse_args()


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
    metadata = {
        "schema_version": 1,
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
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
