#!/usr/bin/env python3
"""Export SMPL animation to FBX or bake it onto a rigged character FBX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.fbx import (  # noqa: E402
    SMPLAnimation,
    export_smpl_fbx,
    retarget_smpl_to_fbx,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="motion135 .npy/.npz or SMPL .npz")
    parser.add_argument("output", type=Path, help="output .fbx under outputs/")
    parser.add_argument(
        "--input-format", choices=("motion135", "smpl-npz"), default="motion135"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=("smpl", "smplh", "smplx"), default="smpl")
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--betas", type=Path, help="optional .npy shape coefficients")
    parser.add_argument("--blender", type=Path, help="Blender 3.6+ executable")
    parser.add_argument(
        "--character-fbx",
        type=Path,
        help="bake onto this already rigged and skinned character instead of exporting SMPL",
    )
    parser.add_argument(
        "--bone-map", type=Path, help="JSON map from SMPL-22 to target bone names"
    )
    parser.add_argument("--target-armature", help="armature object name when the FBX contains several")
    parser.add_argument(
        "--allow-partial-map",
        action="store_true",
        help="allow target rigs that do not map every SMPL-22 body bone",
    )
    parser.add_argument(
        "--root-motion-scale",
        default="auto",
        help="'auto' or a positive numeric scale for target root translation",
    )
    return parser


def _array_from_file(path: Path, *keys: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.ndarray):
        return value
    try:
        for key in keys:
            if key in value:
                return np.asarray(value[key])
        raise KeyError(f"{path} contains none of the required arrays: {keys}.")
    finally:
        value.close()


def _animation(args: argparse.Namespace) -> SMPLAnimation:
    betas = _array_from_file(args.betas, "betas") if args.betas else None
    if args.input_format == "motion135":
        motion = _array_from_file(args.input, "motion135", "motion")
        return SMPLAnimation.from_motion135(motion, betas=betas, fps=args.fps)

    with np.load(args.input, allow_pickle=False) as data:
        required = {"global_orient", "body_pose"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{args.input} is missing SMPL arrays: {missing}.")
        translation_key = "transl" if "transl" in data else "trans" if "trans" in data else None
        if translation_key is None:
            raise KeyError(f"{args.input} must contain 'transl' or 'trans'.")
        shape = betas if betas is not None else data["betas"] if "betas" in data else None
        return SMPLAnimation.from_smpl(
            data["global_orient"],
            data["body_pose"],
            data[translation_key],
            betas=shape,
            fps=args.fps,
        )


def _root_scale(value: str):
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError as error:
        raise ValueError("--root-motion-scale must be 'auto' or a number.") from error


def main() -> None:
    args = _parser().parse_args()
    animation = _animation(args)
    common = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "gender": args.gender,
        "blender_executable": args.blender,
    }
    if args.character_fbx:
        bone_map = json.loads(args.bone_map.read_text()) if args.bone_map else None
        if bone_map is not None and not isinstance(bone_map, dict):
            raise TypeError("--bone-map JSON must contain an object.")
        result = retarget_smpl_to_fbx(
            animation,
            args.character_fbx,
            args.output,
            bone_map=bone_map,
            target_armature=args.target_armature,
            strict_bone_map=not args.allow_partial_map,
            root_motion_scale=_root_scale(args.root_motion_scale),
            **common,
        )
    else:
        result = export_smpl_fbx(animation, args.output, **common)
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "manifest": str(result.manifest_path),
                "mode": result.mode,
                "frames": result.frames,
                "fps": result.fps,
                "armature": result.armature_name,
                "bone_map": result.bone_map,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
