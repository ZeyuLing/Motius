#!/usr/bin/env python3
"""Generate and render compact two-person SMPL Model Zoo previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from motius.motion.representation.interhuman262 import interhuman262_to_joints
from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip
from motius.pipelines.intergen import InterGenPipeline
from motius.pipelines.intermask import InterMaskPipeline
from render_model_zoo_gallery import gif_frame_durations_ms
from render_motion135_smpl_demo import SMPLRenderer


CASES = {
    "intergen": [
        ("handshake", "two people shake hands and then step apart"),
        ("help_stand", "one person helps another person stand up"),
    ],
    "intermask": [
        ("hug", "two people hug each other and then step back"),
        ("gentle_push", "one person gently pushes the other person backward"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intergen-artifact", required=True)
    parser.add_argument("--intermask-artifact", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("assets/model_zoo"))
    parser.add_argument("--work-root", type=Path, default=Path("outputs/release/interaction_previews"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--motion-len", type=int, default=90)
    parser.add_argument("--refine-iters", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def pair_to_vertices(
    motion: np.ndarray,
    renderer: SMPLRenderer,
    smpl_rest,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    people = []
    errors = []
    for person in range(2):
        joints = interhuman262_to_joints(motion[:, person])
        fit = retarget_hml263_clip(
            joints,
            smpl_rest=smpl_rest,
            device=args.device,
            source_fps=args.fps,
            target_fps=args.fps,
            refine_iters=args.refine_iters,
            floor_align=False,
            rotation_init="position_ik",
        )
        people.append(renderer.vertices(fit["global_orient"], fit["body_pose"], fit["transl"]))
        errors.append(float(np.asarray(fit["fit_mpjpe_mm"]).mean()))
    verts = np.stack(people, axis=1)
    verts[..., 1] -= float(verts[..., 1].min())
    return verts, float(np.mean(errors))


def main() -> None:
    args = parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    renderer = SMPLRenderer(Path(args.model_dir), args.device, args.width, args.height)
    smpl_rest = load_smpl_rest(args.model_dir, args.device)
    pipelines = {
        "intergen": InterGenPipeline.from_pretrained(
            args.intergen_artifact,
            bundle_kwargs={"device": args.device},
        ),
        "intermask": InterMaskPipeline.from_pretrained(
            args.intermask_artifact,
            bundle_kwargs={"device": args.device, "dataset_name": "interhuman"},
        ),
    }
    for method, cases in CASES.items():
        out_dir = args.out_root / method
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, (case_id, prompt) in enumerate(cases):
            seed = 2027 + index
            motion = pipelines[method](prompt, motion_len=args.motion_len, seed=seed)[0]
            verts, fit_mpjpe_mm = pair_to_vertices(motion, renderer, smpl_rest, args)
            frames = renderer.render_pair(verts, args.fps, args.motion_len)
            stem = f"{method}_interhuman_{case_id}_smpl_pair"
            gif = out_dir / f"{stem}_{args.width}_{args.fps}fps.gif"
            mp4 = args.work_root / f"{stem}.mp4"
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=gif_frame_durations_ms(len(frames), args.fps),
                loop=0,
                disposal=1,
                optimize=False,
            )
            imageio.mimwrite(mp4, frames, fps=args.fps, quality=8, macro_block_size=1)
            metadata = {
                "method": method,
                "case_id": case_id,
                "prompt": prompt,
                "seed": seed,
                "motion_len": args.motion_len,
                "representation": "InterHuman-262 per person",
                "mesh": "neutral SMPL position-IK",
                "fit_mpjpe_mm": fit_mpjpe_mm,
                "gif": str(gif),
                "mp4": str(mp4),
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(metadata), flush=True)
        del pipelines[method]
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
