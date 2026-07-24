"""Run inside the pinned GEM-X environment to export SOMA-native numeric results."""

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


def export(
    input_path: Path,
    output_path: Path,
    *,
    device: str,
    soma_assets: Path,
) -> None:
    """Forward native SOMA parameters to 77 named joints, never to SMPL."""

    payload = _load(input_path)
    from gem.utils.soma_utils.soma_layer import SomaLayer

    soma = SomaLayer(
        data_root=str(soma_assets),
        low_lod=True,
        device=device,
        identity_model_type="mhr",
        mode="warp",
    ).to(device).eval()
    arrays: dict[str, np.ndarray] = {}
    parameter_names = (
        "body_pose",
        "identity_coeffs",
        "scale_params",
        "global_orient",
        "transl",
    )
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
                for name in parameter_names
            }
            arrays[joints_name] = _cpu_array(soma(**arguments)["joints"])
    if "K_fullimg" in payload:
        arrays["K_fullimg"] = _cpu_array(payload["K_fullimg"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--soma-assets",
        type=Path,
        default=Path("inputs/soma_assets"),
    )
    args = parser.parse_args()
    export(
        args.input,
        args.output,
        device=args.device,
        soma_assets=args.soma_assets,
    )


if __name__ == "__main__":
    main()
