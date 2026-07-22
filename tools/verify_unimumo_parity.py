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


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    difference = candidate - reference
    return {
        "allclose": bool(torch.allclose(reference, candidate, rtol=1e-5, atol=1e-5)),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "max_abs_error": float(difference.abs().max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=250)
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
    official_normalized = torch.as_tensor(
        official.normalize_motion(motion),
        dtype=torch.float32,
        device=args.device,
    )
    official_waveform = torch.zeros(
        (
            len(official_normalized),
            1,
            official_normalized.shape[1] * 32_000 // int(official.motion_fps),
        ),
        dtype=torch.float32,
        device=args.device,
    )
    with torch.no_grad():
        official_music_embeddings = official.motion_vqvae.music_encoder(
            official_waveform
        ).cpu()
        official_raw_motion_embeddings = official.motion_vqvae.motion_encoder(
            official_normalized.transpose(1, 2)
        ).cpu()
        _, official_motion_embeddings = official.motion_vqvae.encode(
            official_waveform, official_normalized
        )
        official_motion_embeddings = official_motion_embeddings.cpu()
    official_motion_codes = official.encode_motion(motion).cpu()
    official_caption = official.generate_text(motion_feature=motion)
    torch.manual_seed(args.seed)
    official_music, official_motion = official.music_motion_lm.generate_sample(
        batch={"text": description, "music_code": None, "motion_code": None},
        duration=args.duration,
        conditional_guidance_scale=4.0,
        temperature=args.temperature,
        return_result_only=True,
    )
    official_decoded_motion = official.motion_vqvae.decode_from_code(
        official_music, official_motion
    )
    official_decoded_motion = official.denormalize_motion(
        official_decoded_motion.detach().float().cpu().numpy()
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
    motius_normalized = motius.bundle.normalize_motion(
        torch.as_tensor(motion, dtype=torch.float32, device=motius.device)
    )
    motius_music_embeddings = motius.bundle._zero_audio_embeddings(
        len(motius_normalized), motius_normalized.shape[1]
    )
    motius_raw_motion_embeddings = motius.bundle.motion_codec.encoder(
        motius_normalized.transpose(1, 2)
    )
    motius_motion_embeddings = motius.bundle.motion_codec.encode_embeddings(
        motius_normalized, motius_music_embeddings
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
        temperature=args.temperature,
        top_k=args.top_k,
        generator=generator,
    )
    motius_music = motius_music.cpu()
    motius_motion = motius_motion.cpu()
    motius_decoded_motion = (
        motius.bundle.decode_motion(
            official_music.to(motius.device),
            official_motion.to(motius.device),
        )
        .float()
        .cpu()
        .numpy()
    )
    decoder_error = motius_decoded_motion - official_decoded_motion

    report = {
        "motion_code_equal": bool(
            torch.equal(official_motion_codes, motius_motion_codes)
        ),
        "motion_code_diff_count": int(
            (official_motion_codes != motius_motion_codes).sum()
        ),
        "encoder_stages": {
            "zero_audio": tensor_error(
                official_music_embeddings, motius_music_embeddings
            ),
            "raw_motion": tensor_error(
                official_raw_motion_embeddings,
                motius_raw_motion_embeddings,
            ),
            "pre_quant_motion": tensor_error(
                official_motion_embeddings,
                motius_motion_embeddings,
            ),
        },
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
        "decoder_allclose": bool(
            np.allclose(
                official_decoded_motion,
                motius_decoded_motion,
                rtol=1e-5,
                atol=1e-5,
            )
        ),
        "decoder_rmse": float(np.sqrt(np.mean(decoder_error**2))),
        "decoder_max_abs_error": float(np.abs(decoder_error).max()),
        "official_caption": official_caption,
        "motius_caption": motius_caption,
        "sampling": {
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
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
        "decoder_allclose",
    )
    if not all(report[field] for field in exact_fields):
        raise SystemExit("UniMuMo parity check failed")


if __name__ == "__main__":
    main()
