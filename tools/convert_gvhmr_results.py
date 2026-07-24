#!/usr/bin/env python3
"""Materialize official GVHMR demo parameters with its own body-model code."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch


PARAMETER_KEYS = ("global_orient", "body_pose", "betas", "transl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    model,
    smplx_to_smpl: torch.Tensor,
    joint_regressor: torch.Tensor,
    parameters: dict,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    on_device = {
        name: parameters[name].to(device)
        for name in PARAMETER_KEYS
    }
    with torch.inference_mode():
        output = model(**on_device)
        vertices = torch.stack(
            [
                torch.matmul(smplx_to_smpl, frame_vertices)
                for frame_vertices in output.vertices
            ]
        )
        joints = torch.einsum("jv,fvc->fjc", joint_regressor, vertices)
    return (
        vertices.detach().cpu().numpy().astype(np.float32, copy=False),
        joints.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--runtime-root", default=Path.cwd(), type=Path)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    runtime_root = args.runtime_root.expanduser().resolve()
    actual_revision = subprocess.run(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != args.runtime_revision:
        raise RuntimeError(
            f"Runtime revision mismatch: {actual_revision} != "
            f"{args.runtime_revision}"
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    prediction = torch.load(args.input, map_location="cpu", weights_only=True)
    world = prediction["smpl_params_global"]
    camera = prediction["smpl_params_incam"]
    for group_name, group in (
        ("smpl_params_global", world),
        ("smpl_params_incam", camera),
    ):
        missing = [name for name in PARAMETER_KEYS if name not in group]
        if missing:
            raise KeyError(f"{group_name} is missing {missing}.")

    from hmr4d.utils.smplx_utils import make_smplx

    model = make_smplx("supermotion").to(device).eval()
    smplx_to_smpl = torch.load(
        runtime_root / "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        map_location=device,
        weights_only=True,
    )
    joint_regressor = torch.load(
        runtime_root / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
        map_location=device,
        weights_only=True,
    )
    vertices_camera, joints_camera = materialize(
        model,
        smplx_to_smpl,
        joint_regressor,
        camera,
        device=device,
    )
    vertices_world, joints_world = materialize(
        model,
        smplx_to_smpl,
        joint_regressor,
        world,
        device=device,
    )

    frames = len(vertices_world)
    exported = {
        "K_fullimg": prediction["K_fullimg"].cpu().numpy(),
        "vertices_camera": vertices_camera,
        "vertices_world": vertices_world,
        "joints_camera": joints_camera,
        "joints_world": joints_world,
        "valid": np.ones(frames, dtype=bool),
        "frame_ids": np.arange(frames, dtype=np.int64),
        "runtime_revision": np.asarray(args.runtime_revision),
        "checkpoint_sha256": np.asarray(sha256_file(args.checkpoint)),
        "materialization": np.asarray(
            "official_make_smplx_supermotion_then_smplx2smpl_sparse"
        ),
    }
    for prefix, group in (
        ("smpl_params_global", world),
        ("smpl_params_incam", camera),
    ):
        for name in PARAMETER_KEYS:
            exported[f"{prefix}_{name}"] = (
                group[name].detach().cpu().numpy().astype(np.float32, copy=False)
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **exported)


if __name__ == "__main__":
    main()
