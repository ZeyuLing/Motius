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


def export(input_path: Path, output_path: Path, *, device: str) -> None:
    """Use the official SMPL24 regressor; do not synthesize vertices."""

    payload = _load(input_path)
    from gem.utils.smplx_utils import make_smplx

    body_model = make_smplx("supermotion_smpl24").to(device).eval()
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for group_name, joints_name in (
            ("body_params_incam", "joints_camera"),
            ("body_params_global", "joints_world"),
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
            arrays[joints_name] = _cpu_array(body_model(**arguments))
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
