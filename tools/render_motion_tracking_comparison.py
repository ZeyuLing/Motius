#!/usr/bin/env python3
"""Render persisted MuJoCo tracking rollouts as a synchronized comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout",
        action="append",
        required=True,
        metavar="LABEL=NPZ",
        help="Persisted rollout; repeat to add a controller",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def _load(specification: str) -> tuple[str, np.ndarray, np.ndarray, float]:
    if "=" not in specification:
        raise ValueError(f"Expected LABEL=NPZ, got {specification!r}.")
    label, raw_path = specification.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        reference = np.asarray(archive["reference_qpos"], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    return label, qpos, reference, fps


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    draw.rectangle((14, 14, 34 + width, 48), fill=(20, 27, 25, 220))
    draw.text((24, 22), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def main() -> None:
    args = _arguments()
    if args.fps <= 0 or args.size < 256:
        raise ValueError("FPS must be positive and panel size must be at least 256.")
    loaded = [_load(specification) for specification in args.rollout]
    source_fps = loaded[0][3]
    if any(not np.isclose(item[3], source_fps) for item in loaded):
        raise ValueError("All rollouts must share one physical clock.")

    labels = ["GT reference", *[item[0] for item in loaded]]
    trajectories = [loaded[0][2], *[item[1] for item in loaded]]
    frame_count = min(len(trajectory) for trajectory in trajectories)
    output_count = max(2, int(round((frame_count - 1) / source_fps * args.fps)) + 1)
    frame_indices = np.rint(
        np.linspace(0, frame_count - 1, output_count)
    ).astype(np.int64)

    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("Install `motius[motion-tracking-mujoco]` to render.") from exc

    scene = (
        REPO_ROOT
        / "motius/simulators/mujoco/assets/unitree_g1/scene_mjx_flat_terrain.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.2
    camera.azimuth = 140.0
    camera.elevation = -18.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    try:
        for frame_index in frame_indices:
            panels = []
            for label, trajectory in zip(labels, trajectories):
                camera.lookat[:] = trajectory[frame_index, :3]
                data.qpos[:] = trajectory[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                panels.append(_label_frame(renderer.render().copy(), label))
            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        writer.close()
        renderer.close()
    print(args.output)


if __name__ == "__main__":
    main()
