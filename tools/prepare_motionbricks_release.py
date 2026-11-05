#!/usr/bin/env python3
"""Convert official SafeTensors into a native, self-describing Motius artifact."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from motius.models.motionbricks import MotionBricksReleaseBundle


def prepare(model, output, source=None):
    if source is None:
        root = snapshot_download(
            "nvidia/MotionBricks", revision="3cb362d", allow_patterns=[f"{model}/**"]
        )
        source = Path(root) / model
    bundle = MotionBricksReleaseBundle(source, model=model, device="cpu")
    return bundle.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["soma", "g1"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.model, args.output, args.source))
