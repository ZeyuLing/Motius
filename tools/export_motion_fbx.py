#!/usr/bin/env python3
"""Export a Motius motion representation onto a Mixamo-compatible FBX rig."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.fbx import export_motion_to_fbx  # noqa: E402


_CHECKPOINT_NATIVE = {
    "ardy_330",
    "ardy_g1_414",
    "motionbricks_g1_413",
    "motionbricks_g1_414",
    "motionbricks_g1_418",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input .npy or .npz motion")
    parser.add_argument("output", type=Path, help="output .fbx under outputs/")
    parser.add_argument("--source", required=True, help="public representation name")
    parser.add_argument(
        "--character",
        required=True,
        help="packaged character slug (atlas/nova/gear) or a rigged .fbx path",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=("smpl", "smplh", "smplx"), default="smpl")
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--key", help="array key when input is an .npz archive")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--person-index", type=int, choices=(0, 1))
    parser.add_argument("--betas", type=Path, help="optional .npy beta coefficients")
    parser.add_argument("--mean", type=Path, help="optional source normalization mean")
    parser.add_argument("--std", type=Path, help="optional source normalization std")
    parser.add_argument("--device", help="IK device, for example cuda:0")
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--floor-align", action="store_true")
    parser.add_argument("--robot-xml", type=Path, help="MuJoCo model for G1 qpos")
    parser.add_argument("--bone-map", type=Path, help="SMPL-22 to target-bone JSON map")
    parser.add_argument("--target-armature")
    parser.add_argument("--allow-partial-map", action="store_true")
    parser.add_argument("--root-motion-scale", default="auto")
    parser.add_argument("--blender", type=Path)
    return parser


def _load_array(path: Path, key: str | None = None):
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.ndarray):
        return value
    try:
        if key:
            if key not in value:
                raise KeyError(f"{path} does not contain {key!r}; keys: {value.files}.")
            return np.asarray(value[key])
        if len(value.files) != 1:
            raise ValueError(
                f"{path} contains multiple arrays {value.files}; select one with --key."
            )
        return np.asarray(value[value.files[0]])
    finally:
        value.close()


def _load_input(args: argparse.Namespace):
    source = args.source.casefold().replace("-", "_")
    if source == "smpl":
        with np.load(args.input, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    return _load_array(args.input, args.key)


def _root_scale(value: str):
    if value == "auto":
        return value
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError("--root-motion-scale must be 'auto' or a positive number.") from error
    if number <= 0:
        raise ValueError("--root-motion-scale must be positive.")
    return number


def main() -> None:
    args = _parser().parse_args()
    source = args.source.casefold().replace("-", "_")
    if source in _CHECKPOINT_NATIVE:
        raise ValueError(
            f"{source} requires the pipeline's motion_rep object. Use the Python API "
            "example in docs/motion/fbx.md so the decoder and its stats stay paired."
        )
    if (args.mean is None) != (args.std is None):
        raise ValueError("--mean and --std must be provided together.")
    bridge_kwargs = {
        "refine_iters": args.refine_iters,
        "floor_align": args.floor_align,
    }
    if args.device:
        bridge_kwargs["device"] = args.device
    if args.mean:
        bridge_kwargs["mean"] = _load_array(args.mean)
        bridge_kwargs["std"] = _load_array(args.std)
    bone_map = json.loads(args.bone_map.read_text()) if args.bone_map else None
    if bone_map is not None and not isinstance(bone_map, dict):
        raise TypeError("--bone-map JSON must contain an object.")
    betas = _load_array(args.betas) if args.betas else None
    result = export_motion_to_fbx(
        _load_input(args),
        args.source,
        args.character,
        args.output,
        model_path=args.model_path,
        model_type=args.model_type,
        gender=args.gender,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        betas=betas,
        person_index=args.person_index,
        robot_xml=args.robot_xml,
        bridge_kwargs=bridge_kwargs,
        bone_map=bone_map,
        target_armature=args.target_armature,
        strict_bone_map=not args.allow_partial_map,
        root_motion_scale=_root_scale(args.root_motion_scale),
        blender_executable=args.blender,
    )
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
