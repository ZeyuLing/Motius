#!/usr/bin/env python3
"""Render a Motius monocular-capture result for a public model card."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render_motion135_smpl_demo import SMPLRenderer  # noqa: E402


def _outputs_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to((ROOT / "outputs").resolve())
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _faces(method: str, renderer: SMPLRenderer, soma_assets: Path | None) -> np.ndarray:
    if method == "gem-smpl":
        return renderer.faces
    if soma_assets is None:
        raise ValueError("--soma-assets is required for GEM-X.")
    with np.load(soma_assets, allow_pickle=False) as archive:
        return np.asarray(archive["triangles_low"], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("gem-smpl", "gem-x"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--soma-assets", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    output = _outputs_path(args.output_dir)
    with np.load(args.input.expanduser().resolve(), allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices_world"], dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices_world must have shape (T, V, 3), got {vertices.shape}.")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices_world contains non-finite values.")

    vertices = vertices.copy()
    vertices[..., 1] -= float(vertices[..., 1].min())
    renderer = SMPLRenderer(
        args.model_dir.expanduser().resolve(),
        args.device,
        args.width,
        args.height,
    )
    renderer.faces = _faces(args.method, renderer, args.soma_assets)
    frames = renderer.render(vertices, args.fps, len(vertices))

    mp4 = output / f"{args.name}.mp4"
    webp = output / f"{args.name}.webp"
    metadata = output / f"{args.name}.json"
    imageio.mimwrite(
        mp4,
        frames,
        fps=args.fps,
        quality=8,
        macro_block_size=1,
    )
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        webp,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / args.fps),
        loop=0,
        quality=86,
        method=6,
    )
    payload = {
        "method": args.method,
        "source": "Motius infer_monocular_motion_capture vertices_world",
        "representation": "SMPL-6890" if args.method == "gem-smpl" else "SOMA-4505",
        "frames": len(frames),
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "mp4": mp4.name,
        "webp": webp.name,
    }
    metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
