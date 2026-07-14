#!/usr/bin/env python3
"""Generate ARDY Core-27 demos and a Core->SMPL-22 joint preview.

This tool runs entirely through Motius. It never imports an upstream ARDY
checkout. In environments without the gated LLM2Vec/Llama text encoder, pass
``--synthetic-text-feat`` to verify the released Core checkpoint, decoder, and
retarget visualization paths. Such outputs are not semantic T2M results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion import ardy_core27_to_smpl22_joints  # noqa: E402
from motius.models.ardy.network.skeleton import CoreSkeleton27  # noqa: E402
from motius.pipelines.ardy import ARDYPipeline  # noqa: E402
from motius.motion.skeleton import SMPL22_PARENTS  # noqa: E402


DEFAULT_CASES = [
    {
        "sample_id": "001840",
        "caption": "someone executes a roundhouse kick with their left foot.",
    },
    {
        "sample_id": "004545",
        "caption": "a person jumping while raising both hands and moving apart legs.",
    },
    {
        "sample_id": "006944",
        "caption": "a person moves their right hand left, right, up, and down.",
    },
]


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _synthetic_text_inputs(caption: str, device: torch.device, tokens: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(_seed_from_text(caption))
    feat = torch.randn(1, tokens, 4096, generator=generator, device=device)
    mask = torch.ones(1, tokens, dtype=torch.bool, device=device)
    return feat, mask


def _normalize_for_view(joints: np.ndarray, *, x_offset: float) -> np.ndarray:
    value = np.asarray(joints, dtype=np.float32).copy()
    origin = value[0, 0].copy()
    origin[1] = 0.0
    value -= origin
    value[..., 1] -= float(value[..., 1].min())
    value[..., 0] += x_offset
    return value


def _draw_skeleton(ax, points: np.ndarray, parents, *, color: str):
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=9, c=color, depthshade=False)
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        segment = points[[parent, joint]]
        ax.plot(segment[:, 0], segment[:, 2], segment[:, 1], color=color, linewidth=2.0)


def _render_gif(core: np.ndarray, smpl: np.ndarray, path: Path, fps: int):
    core_skeleton = CoreSkeleton27()
    core_parents = [int(x) for x in core_skeleton.joint_parents.tolist()]
    core_view = _normalize_for_view(core, x_offset=-1.0)
    smpl_view = _normalize_for_view(smpl, x_offset=1.0)
    all_points = np.concatenate([core_view, smpl_view], axis=1)
    center = all_points.mean(axis=(0, 1))
    radius = max(1.8, float(np.ptp(all_points.reshape(-1, 3), axis=0).max()) * 0.62)

    frames = []
    for frame in range(len(core_view)):
        fig = plt.figure(figsize=(7.5, 4.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        _draw_skeleton(ax, core_view[frame], core_parents, color="#087991")
        _draw_skeleton(ax, smpl_view[frame], SMPL22_PARENTS, color="#9a4e31")
        ax.text(-1.0, -1.2, 1.95, "ARDY Core-27", color="#087991", ha="center")
        ax.text(1.0, -1.2, 1.95, "SMPL-22 joints", color="#9a4e31", ha="center")
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(0.0, max(2.1, center[1] + radius))
        ax.view_init(elev=15, azim=-78)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        frames.append(image)
        plt.close(fig)
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


def _load_cases(path: Path | None):
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("cases", payload.get("preview_cases", []))
    cases = []
    for item in payload:
        cases.append(
            {
                "sample_id": str(item.get("sample_id", item.get("id", len(cases)))),
                "caption": str(item["caption"]),
            }
        )
    if not cases:
        raise ValueError(f"no cases found in {path}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ardy/core_humanml3d_demo"))
    parser.add_argument("--checkpoint", default="core8")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--synthetic-text-feat", action="store_true")
    parser.add_argument("--hf-home", type=Path, default=Path("checkpoints/ardy"))
    args = parser.parse_args()

    if args.synthetic_text_feat:
        args.hf_home.mkdir(parents=True, exist_ok=True)
        import os

        os.environ.setdefault("HF_HOME", str(args.hf_home.resolve()))

    pipe = ARDYPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "text_encoder": False if args.synthetic_text_feat else None,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": pipe.bundle.model_name,
        "frames": args.frames,
        "fps": args.fps,
        "synthetic_text_feat": bool(args.synthetic_text_feat),
        "note": (
            "Synthetic text features validate checkpoint/decoder/retarget plumbing only; "
            "they are not semantic T2M results."
            if args.synthetic_text_feat
            else "Official text encoder path."
        ),
        "cases": [],
    }
    for index, case in enumerate(_load_cases(args.cases)):
        caption = case["caption"]
        kwargs = {}
        if args.synthetic_text_feat:
            text_feat, text_mask = _synthetic_text_inputs(caption, pipe.device, args.tokens)
            kwargs.update(text_feat=text_feat, text_pad_mask=text_mask)
        result = pipe.text_to_motion(
            caption,
            args.frames,
            num_denoising_steps=args.steps,
            seed=args.seed + index,
            return_numpy=True,
            **kwargs,
        )
        core = np.asarray(result["posed_joints"][0, : args.frames], dtype=np.float32)
        smpl = ardy_core27_to_smpl22_joints(core)
        stem = f"ardy_core_{case['sample_id']}"
        npz_path = args.output_dir / f"{stem}.npz"
        gif_path = args.output_dir / f"{stem}_core_smpl22.gif"
        np.savez_compressed(
            npz_path,
            core_joints=core,
            smpl22_joints=smpl,
            caption=caption,
            sample_id=case["sample_id"],
            fps=args.fps,
        )
        _render_gif(core, smpl, gif_path, args.fps)
        manifest["cases"].append(
            {
                "sample_id": case["sample_id"],
                "caption": caption,
                "npz": str(npz_path),
                "gif": str(gif_path),
            }
        )
        print(gif_path)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
