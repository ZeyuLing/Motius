#!/usr/bin/env python3
"""Materialize PromptHMR geometry with a private licensed SMPL-X model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.pipelines.prompthmr import (
    parse_prompthmr_results,
    replay_prompthmr_with_licensed_model,
)
from motius.motion.representation.monocular_capture import (
    save_monocular_capture_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay official PromptHMR parameters through a user-supplied "
            "licensed SMPL-X model. No model files are downloaded or copied."
        )
    )
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output must end in .motius.npz and must not mimic an official file.",
    )
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument(
        "--model-gender",
        choices=("neutral", "female", "male"),
        default="neutral",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="User-declared licensed SMPL-X release, checked against file metadata.",
    )
    parser.add_argument("--model-sha256")
    parser.add_argument("--video-checkpoint-sha256", required=True)
    parser.add_argument("--image-checkpoint-sha256")
    parser.add_argument("--original-fps", type=float, required=True)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument(
        "--space",
        choices=("camera", "world", "both"),
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-vertices",
        action="store_true",
        help="Materialize named joints only.",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=("box", "keypoint", "mask"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    if not output.name.endswith(".motius.npz"):
        raise ValueError(
            "Licensed replay output must end in .motius.npz; it is a "
            "pickle-free Motius artifact, not native PromptHMR results.pkl."
        )
    if "outputs" not in output.parts:
        raise ValueError("Licensed replay output must live under outputs/.")
    checkpoint_hashes = {"video_head": args.video_checkpoint_sha256}
    if args.image_checkpoint_sha256:
        checkpoint_hashes["image_model"] = args.image_checkpoint_sha256
    parsed = parse_prompthmr_results(
        args.official_results,
        checkpoint_sha256=args.video_checkpoint_sha256,
        checkpoint_sha256s=checkpoint_hashes,
        original_fps=args.original_fps,
        output_fps=args.output_fps,
        prompt_types=args.prompt_types,
    )
    spaces = (
        ("camera", "world")
        if args.space == "both"
        else (args.space,)
    )
    materialized = replay_prompthmr_with_licensed_model(
        parsed,
        args.smplx_model,
        gender=args.model_gender,
        model_version=args.model_version,
        expected_sha256=args.model_sha256,
        spaces=spaces,
        include_vertices=not args.no_vertices,
        batch_size=args.batch_size,
        device=args.device,
    )

    save_monocular_capture_result(materialized, output)
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_text(
        json.dumps(materialized.public_manifest(), indent=2) + "\n"
    )
    print(f"Wrote licensed replay result: {output}")
    print(f"Wrote public provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
