#!/usr/bin/env python3
"""Generate the native MotionStreamer temporal-completion Model Card demo."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--condition-frames", type=int, default=30)
    args = parser.parse_args()

    pipe = Pipeline.from_pretrained(args.artifact, device=args.device)
    caption = "a person walks forward and then turns right"
    reference = pipe.infer_text_to_motion(
        [caption],
        [args.frames],
    )[0]
    generated = pipe.infer_temporal_motion_completion(
        [caption],
        [args.frames],
        [reference],
        condition_num_frames=args.condition_frames,
    )[0]
    joints = np.asarray(
        convert_motion(generated, "ms272", "joints"),
        dtype=np.float32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        motion_ms272=np.asarray(generated, dtype=np.float32),
        joints=joints[:, None],
        parents=np.asarray(SMPL22_PARENTS, dtype=np.int32),
        caption=np.asarray(caption),
        case_id=np.asarray("motionstreamer_temporal_motion_completion"),
        fps=np.asarray(30, dtype=np.int32),
        condition_intervals=np.asarray(
            [[0, args.condition_frames]],
            dtype=np.int32,
        ),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
