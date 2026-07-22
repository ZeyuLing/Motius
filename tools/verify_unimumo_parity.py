#!/usr/bin/env python3
"""Compare the Motius UniMuMo artifact with the official implementation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=0.4)
    return parser.parse_args()


def prepare_motion(path: Path) -> np.ndarray:
    motion = np.asarray(np.load(path)[:20], dtype=np.float32)
    motion_60fps = F.interpolate(
        torch.from_numpy(motion).T.unsqueeze(0),
        scale_factor=3,
        mode="linear",
    ).transpose(1, 2)
    padded = np.zeros((1, 600, 263), dtype=np.float32)
    padded[:, : motion_60fps.shape[1]] = motion_60fps.numpy()
    return padded


def main() -> None:
    args = parse_args()
    motion = prepare_motion(args.motion)
    description = [
        "energetic electronic dance music <separation> "
        "a person dances energetically"
    ]

    from unimumo.models import UniMuMo as OfficialUniMuMo

    torch.manual_seed(args.seed)
    official = OfficialUniMuMo.from_checkpoint(
        str(args.official_checkpoint), device=args.device
    )
    official_motion_codes = official.encode_motion(motion).cpu()
    official_caption = official.generate_text(motion_feature=motion)
    official_music, official_motion = official.music_motion_lm.generate_sample(
        batch={"text": description, "music_code": None, "motion_code": None},
        duration=args.duration,
        conditional_guidance_scale=4.0,
        temperature=0.0,
        return_result_only=True,
    )
    official_music = official_music.cpu()
    official_motion = official_motion.cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    from motius.pipelines.unimumo import UniMuMoPipeline

    motius = UniMuMoPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"local_files_only": True},
        device=args.device,
    )
    motius_motion_codes = motius.bundle.encode_motion(motion).cpu()
    motius_caption = list(
        motius.infer_motion_to_text(motion, input_fps=60).captions
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    motius_music, motius_motion = motius.bundle.generate_codes(
        description,
        timesteps=round(args.duration * 50),
        mode="music_motion",
        guidance_scale=4.0,
        temperature=0.0,
        top_k=250,
        generator=generator,
    )
    motius_music = motius_music.cpu()
    motius_motion = motius_motion.cpu()

    report = {
        "motion_code_equal": bool(
            torch.equal(official_motion_codes, motius_motion_codes)
        ),
        "motion_code_diff_count": int(
            (official_motion_codes != motius_motion_codes).sum()
        ),
        "generation_music_equal": bool(
            torch.equal(official_music, motius_music)
        ),
        "generation_music_diff_count": int(
            (official_music != motius_music).sum()
        ),
        "generation_motion_equal": bool(
            torch.equal(official_motion, motius_motion)
        ),
        "generation_motion_diff_count": int(
            (official_motion != motius_motion).sum()
        ),
        "official_caption": official_caption,
        "motius_caption": motius_caption,
        "shapes": {
            "encoded": list(motius_motion_codes.shape),
            "generated": list(motius_music.shape),
        },
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")

    exact_fields = (
        "motion_code_equal",
        "generation_music_equal",
        "generation_motion_equal",
    )
    if not all(report[field] for field in exact_fields):
        raise SystemExit("UniMuMo parity check failed")


if __name__ == "__main__":
    main()
