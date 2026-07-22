#!/usr/bin/env python3
"""Compare official and Motius UniMuMo motion-to-music inference."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--segment-index", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def error_summary(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    error = actual.float() - reference.float()
    return {
        "allclose": bool(torch.allclose(reference, actual, rtol=1e-5, atol=1e-5)),
        "rmse": float(error.square().mean().sqrt()),
        "max_abs_error": float(error.abs().max()),
    }


def main() -> None:
    args = parse_args()
    motion = np.load(args.motion)
    first = (args.segment_index - 1) * 120
    motion = np.asarray(motion[first : first + 120], dtype=np.float32)[None]
    if motion.shape != (1, 120, 263):
        raise ValueError(f"Expected a two-second (1,120,263) clip, got {motion.shape}")

    from unimumo.models import UniMuMo as OfficialUniMuMo

    official = OfficialUniMuMo.from_checkpoint(
        str(args.official_checkpoint), device=args.device
    )
    official_motion = official.encode_motion(motion)
    torch.manual_seed(args.seed)
    official_music = official.music_motion_lm.generate_single_modality(
        music_code=None,
        motion_code=official_motion,
        text_description=["<separation>"],
        conditional_guidance_scale=args.guidance_scale,
        temperature=args.temperature,
    )
    official_waveform = official.music_vqvae.decode(official_music).detach().cpu()
    official_motion = official_motion.detach().cpu()
    official_music = official_music.detach().cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from motius.pipelines.unimumo import UniMuMoPipeline

    motius = UniMuMoPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"local_files_only": True},
        device=args.device,
    )
    motius_motion = motius.bundle.encode_motion(motion)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    motius_music, conditioned_motion = motius.bundle.generate_codes(
        ["<separation>"],
        timesteps=motius_motion.shape[-1],
        mode="motion2music",
        motion_codes=motius_motion,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=250,
        generator=generator,
    )
    motius_waveform = motius.bundle.decode_audio(
        official_music.to(motius.device)
    ).detach().cpu()
    report = {
        "motion_code_equal": bool(
            torch.equal(official_motion, motius_motion.detach().cpu())
        ),
        "motion_code_diff_count": int(
            (official_motion != motius_motion.detach().cpu()).sum()
        ),
        "music_code_equal": bool(
            torch.equal(official_music, motius_music.detach().cpu())
        ),
        "music_code_diff_count": int(
            (official_music != motius_music.detach().cpu()).sum()
        ),
        "conditioned_motion_unchanged": bool(
            torch.equal(motius_motion, conditioned_motion)
        ),
        "audio_decoder": error_summary(official_waveform, motius_waveform),
        "shapes": {
            "motion_codes": list(official_motion.shape),
            "music_codes": list(official_music.shape),
            "waveform": list(official_waveform.shape),
        },
        "sampling": {
            "seed": args.seed,
            "guidance_scale": args.guidance_scale,
            "temperature": args.temperature,
            "top_k": 250,
            "prompt": "<separation>",
        },
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    passed = (
        report["motion_code_equal"]
        and report["music_code_equal"]
        and report["conditioned_motion_unchanged"]
        and report["audio_decoder"]["allclose"]
    )
    if not passed:
        raise SystemExit("UniMuMo motion-to-music parity check failed")


if __name__ == "__main__":
    main()
