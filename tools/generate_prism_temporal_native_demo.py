#!/usr/bin/env python3
"""Generate PRISM-KT temporal completion through the public Pipeline API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius import Pipeline


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_smpl(path: Path, result: dict, condition_frames: int = 0) -> None:
    smpl = result["smpl"]
    global_orient = _numpy(smpl["global_orient"]).astype(np.float32)
    body_pose = _numpy(smpl["body_pose"]).astype(np.float32)
    translation = _numpy(smpl["transl"]).astype(np.float32)
    while global_orient.ndim > 2 and global_orient.shape[0] == 1:
        global_orient = global_orient[0]
    while body_pose.ndim > 2 and body_pose.shape[0] == 1:
        body_pose = body_pose[0]
    while translation.ndim > 2 and translation.shape[0] == 1:
        translation = translation[0]
    betas = _numpy(
        smpl.get("betas", np.zeros(10, dtype=np.float32))
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_orient": global_orient.reshape(len(global_orient), 3),
        "body_pose": body_pose.reshape(len(body_pose), -1),
        "transl": translation.reshape(len(translation), 3),
        "betas": betas.reshape(-1)[-10:],
        "mocap_framerate": np.asarray(
            result.get("fps", 30),
            dtype=np.float32,
        ),
    }
    if condition_frames:
        payload["condition_intervals"] = np.asarray(
            [[0, condition_frames]],
            dtype=np.int32,
        )
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--condition-frames", type=int, default=5)
    args = parser.parse_args()

    pipe = Pipeline.from_pretrained(args.artifact, device=args.device)
    caption = "a person walks forward and then turns right"
    reference = pipe.infer_text_to_motion(
        caption,
        args.frames,
        seed=17,
        num_inference_steps=25,
    )
    _save_smpl(args.prefix, reference)
    generated = pipe.infer_temporal_motion_completion(
        caption,
        str(args.prefix),
        num_frames=args.frames,
        condition_num_frames=args.condition_frames,
        seed=23,
        num_inference_steps=25,
    )
    _save_smpl(args.output, generated, args.condition_frames)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
