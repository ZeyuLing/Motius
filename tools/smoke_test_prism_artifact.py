#!/usr/bin/env python3
"""Load a self-contained PRISM artifact and run a minimal T2M inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motius.pipelines.prism import PRISMPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    args = parser.parse_args()

    pipe = PRISMPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={
            "device": args.device,
            "transformer_dtype": "bf16",
            "text_dtype": "bf16",
        },
    )
    result = pipe.text_to_motion(
        "a person walks forward and waves with the right hand",
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=1.0,
        seed=20260716,
    )

    motion138 = np.asarray(result["motion_138"])
    motion135 = np.asarray(result["motion_135"])
    smpl = result["smpl"]
    assert motion138.shape == (args.num_frames, 138), motion138.shape
    assert motion135.shape == (args.num_frames, 135), motion135.shape
    assert np.asarray(smpl["transl"]).shape == (args.num_frames, 3)
    assert np.isfinite(motion138).all()
    assert np.isfinite(motion135).all()

    report = {
        "artifact": str(Path(args.artifact).resolve()),
        "variant": result["variant"],
        "device": str(pipe.bundle.device),
        "transformer_dtype": str(next(pipe.bundle.transformer.parameters()).dtype),
        "vae_dtype": str(next(pipe.bundle.vae.parameters()).dtype),
        "text_dtype": str(next(pipe.bundle.text_encoder.parameters()).dtype),
        "motion_138_shape": list(motion138.shape),
        "motion_135_shape": list(motion135.shape),
        "motion_138_abs_max": float(np.abs(motion138).max()),
        "finite": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
