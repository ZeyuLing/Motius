#!/usr/bin/env python3
"""Generate the native MaskControl temporal-completion Model Card demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius import Pipeline
from motius.motion import convert_motion
from motius.motion.skeleton import SMPL22_PARENTS


def _resample(array: np.ndarray, source_fps: int, target_fps: int) -> np.ndarray:
    if source_fps == target_fps or len(array) < 2:
        return array
    frames = round((len(array) - 1) * target_fps / source_fps) + 1
    source = np.linspace(0.0, 1.0, len(array))
    target = np.linspace(0.0, 1.0, frames)
    flattened = array.reshape(len(array), -1)
    result = np.stack(
        [
            np.interp(target, source, flattened[:, index])
            for index in range(flattened.shape[1])
        ],
        axis=-1,
    )
    return result.reshape((frames,) + array.shape[1:]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--prefix-ratio", type=float, default=0.2)
    args = parser.parse_args()

    pipe = Pipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"device": args.device},
        device=args.device,
    )
    caption = "a person walks forward and then turns right"
    reference = pipe.infer_text_to_motion(
        [caption],
        [args.frames],
        seed=17,
    )[0]
    generated = pipe.infer_temporal_motion_completion(
        [caption],
        [reference],
        mode="prefix",
        prefix_ratio=args.prefix_ratio,
        seed=23,
    )[0]
    joints = np.asarray(
        convert_motion(generated, "hml263", "joints"),
        dtype=np.float32,
    )
    joints = _resample(joints, 20, 30)
    condition_frames = round(args.frames * args.prefix_ratio * 30 / 20)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        motion_hml263=np.asarray(generated, dtype=np.float32),
        joints=joints[:, None],
        parents=np.asarray(SMPL22_PARENTS, dtype=np.int32),
        caption=np.asarray(caption),
        case_id=np.asarray("maskcontrol_temporal_motion_completion"),
        fps=np.asarray(30, dtype=np.int32),
        condition_intervals=np.asarray(
            [[0, condition_frames]],
            dtype=np.int32,
        ),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
