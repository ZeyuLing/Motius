#!/usr/bin/env python3
"""Export the stage trace of an existing official or Motius GVHMR run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from pytorch3d.transforms import quaternion_to_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.gvhmr.parity import capture_gvhmr_trace
from motius.models.gvhmr.vendor.hmr4d.utils.geo.hmr_cam import (
    create_camera_sensor,
    estimate_K,
)
from motius.models.gvhmr.vendor.hmr4d.utils.geo_transform import (
    compute_cam_angvel,
)
from motius.models.gvhmr.vendor.hmr4d.utils.video_io_utils import get_video_lwh


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_run_data(
    run_dir: Path,
    *,
    static_camera: bool,
    use_dpvo: bool,
    focal_length_mm: Optional[int],
) -> tuple[dict, dict, dict]:
    """Reconstruct the exact official model input from persisted caches."""

    video = run_dir / "0_input_video.mp4"
    preprocess = run_dir / "preprocess"
    length, width, height = get_video_lwh(video)
    bbox = _load(preprocess / "bbx.pt")
    if static_camera:
        rotation_world_to_camera = torch.eye(3).repeat(length, 1, 1)
    else:
        trajectory = _load(preprocess / "slam_results.pt")
        if use_dpvo:
            trajectory = torch.as_tensor(trajectory)
            quaternion = trajectory[:, [6, 3, 4, 5]]
            rotation_world_to_camera = quaternion_to_matrix(quaternion).mT
        else:
            rotation_world_to_camera = torch.from_numpy(
                trajectory[:, :3, :3]
            )

    if focal_length_mm is None:
        intrinsics = estimate_K(width, height).repeat(length, 1, 1)
    else:
        intrinsics = create_camera_sensor(
            width,
            height,
            focal_length_mm,
        )[2].repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbox["bbx_xys"],
        "kp2d": _load(preprocess / "vitpose.pt"),
        "K_fullimg": intrinsics,
        "cam_angvel": compute_cam_angvel(rotation_world_to_camera),
        "f_imgseq": _load(preprocess / "vit_features.pt"),
    }
    prediction = _load(run_dir / "hmr4d_results.pt")
    return bbox, data, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--static-camera", action="store_true")
    parser.add_argument("--use-dpvo", action="store_true")
    parser.add_argument("--focal-length-mm", type=int)
    args = parser.parse_args()

    bbox, data, prediction = load_run_data(
        args.run_dir.expanduser().resolve(),
        static_camera=args.static_camera,
        use_dpvo=args.use_dpvo,
        focal_length_mm=args.focal_length_mm,
    )
    output = capture_gvhmr_trace(
        args.output,
        name=args.name,
        bbox=bbox,
        data=data,
        prediction=prediction,
        metadata={
            "static_camera": args.static_camera,
            "use_dpvo": args.use_dpvo,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
