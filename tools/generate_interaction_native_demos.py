#!/usr/bin/env python3
"""Generate native InterHuman skeleton demos through Motius pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motius.motion.representation.interhuman262 import interhuman262_to_joints
from motius.pipelines.intergen import InterGenPipeline
from motius.pipelines.intermask import InterMaskPipeline


CASES = {
    "intergen": (
        ("handshake", "two people shake hands and then step apart"),
        ("help_stand", "one person helps another person stand up"),
    ),
    "intermask": (
        ("hug", "two people hug each other and then step back"),
        ("gentle_push", "one person gently pushes the other person backward"),
    ),
}
PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
     16, 17, 18, 19],
    dtype=np.int32,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intergen-artifact", required=True)
    parser.add_argument("--intermask-artifact", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = (
        (
            "intergen",
            InterGenPipeline,
            args.intergen_artifact,
            {"device": args.device},
        ),
        (
            "intermask",
            InterMaskPipeline,
            args.intermask_artifact,
            {"device": args.device, "dataset_name": "interhuman"},
        ),
    )
    for method, pipeline_type, artifact, bundle_kwargs in specs:
        pipeline = pipeline_type.from_pretrained(
            artifact,
            bundle_kwargs=bundle_kwargs,
        )
        for index, (case_id, caption) in enumerate(CASES[method]):
            motion = pipeline(
                caption,
                motion_len=args.frames,
                seed=2027 + index,
            )[0]
            joints = np.stack(
                [
                    interhuman262_to_joints(motion[:, person])
                    for person in range(2)
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            joints[..., 1] -= joints[..., 1].min()
            output = args.output / f"{method}_{case_id}.npz"
            np.savez_compressed(
                output,
                joints=joints,
                parents=PARENTS,
                caption=np.asarray(caption),
                method=np.asarray(method),
                case_id=np.asarray(case_id),
                fps=np.asarray(30, dtype=np.int32),
                representation=np.asarray("interhuman-262-native-joints"),
            )
            print(output, flush=True)
        del pipeline
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
