"""Run inside the pinned GEM-SMPL environment to export numeric native results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _cpu_array(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def _apply_sparse_vertex_map(
    vertex_map: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:
    """Apply a 2D sparse vertex map to a ``[T, V, 3]`` motion."""

    frames, source_vertices, coordinates = vertices.shape
    if coordinates != 3 or vertex_map.shape[1] != source_vertices:
        raise ValueError(
            f"Incompatible vertex map {tuple(vertex_map.shape)} and motion "
            f"{tuple(vertices.shape)}."
        )
    flattened = vertices.permute(1, 0, 2).reshape(source_vertices, frames * 3)
    mapped = torch.sparse.mm(vertex_map, flattened)
    return mapped.reshape(vertex_map.shape[0], frames, 3).permute(1, 0, 2)


def export(input_path: Path, output_path: Path, *, device: str) -> None:
    """Materialize official SMPL-X geometry on the neutral SMPL topology."""

    payload = _load(input_path)
    from gem.utils.body_model import __file__ as body_asset_module
    from gem.utils.smplx_utils import make_smplx

    body_asset_root = Path(body_asset_module).parent
    smplx_to_smpl = _load(body_asset_root / "smplx2smpl_sparse.pt").to(device)
    smpl_joint_regressor = _load(
        body_asset_root / "smpl_neutral_J_regressor.pt"
    ).to(device)
    body_model = make_smplx("supermotion").to(device).eval()
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for group_name, joints_name, vertices_name in (
            ("body_params_incam", "joints_camera", "vertices_camera"),
            ("body_params_global", "joints_world", "vertices_world"),
        ):
            if group_name not in payload:
                continue
            parameters = payload[group_name]
            for name, value in parameters.items():
                arrays[f"{group_name}.{name}"] = _cpu_array(value)
            arguments = {
                name: parameters[name].to(device)
                for name in ("body_pose", "betas", "global_orient", "transl")
            }
            output = body_model(**arguments)
            smpl_vertices = _apply_sparse_vertex_map(
                smplx_to_smpl,
                output.vertices,
            )
            smpl_joints = torch.matmul(smpl_joint_regressor, smpl_vertices)
            arrays[joints_name] = _cpu_array(smpl_joints)
            arrays[vertices_name] = _cpu_array(smpl_vertices)
    if "K_fullimg" in payload:
        arrays["K_fullimg"] = _cpu_array(payload["K_fullimg"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    export(args.input, args.output, device=args.device)


if __name__ == "__main__":
    main()
