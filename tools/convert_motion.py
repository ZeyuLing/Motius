#!/usr/bin/env python3
"""Convert motion arrays between public Motius representations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motius.motion.representation.convert import convert_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HML263, MS272, HY-Motion-201, DART276, motion135, or joints"
    )
    parser.add_argument("input", type=Path, help="Input .npy or .npz file")
    parser.add_argument("output", type=Path, help="Output .npy or .npz file")
    parser.add_argument("--src", required=True, help="Source representation")
    parser.add_argument("--dst", required=True, help="Target representation")
    parser.add_argument("--input-key", help="Array key when input is .npz")
    parser.add_argument("--bone-offsets", type=Path, help="(22,3) .npy offsets for FK routes")
    parser.add_argument("--smpl-model-dir", type=Path, help="SMPL assets for HML263 -> motion135")
    parser.add_argument("--gender", choices=("neutral", "male", "female"))
    parser.add_argument("--model-type", choices=("smpl", "smplh", "smplx"))
    parser.add_argument("--src-fps", type=float)
    parser.add_argument("--dst-fps", type=float)
    parser.add_argument(
        "--coordinate-system", choices=("humanml", "amass"),
        help="Coordinate system used by SMPL/joint inputs when targeting HML263",
    )
    parser.add_argument(
        "--resample", choices=("auto", "stride", "linear", "none"),
        help="Temporal resampling mode for HML263 targets",
    )
    parser.add_argument("--feet-threshold", type=float)
    parser.add_argument("--device", default=None, help="Torch device for IK routes")
    parser.add_argument("--refine-iters", type=int, default=None, help="Optional HML-to-SMPL IK steps")
    parser.add_argument(
        "--rotation-space", choices=("local", "global"), default="local",
        help="Rotation space of a motion135 source",
    )
    return parser.parse_args()


def load_array(path: Path, key: str | None, *, preserve_mapping: bool = False):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    with loaded:
        if key:
            if key not in loaded:
                raise KeyError(f"{key!r} not found; available keys: {loaded.files}")
            return np.asarray(loaded[key])
        if preserve_mapping:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
        if len(loaded.files) != 1:
            raise ValueError(f"input has multiple arrays {loaded.files}; pass --input-key")
        return np.asarray(loaded[loaded.files[0]])


def save_array(path: Path, array: np.ndarray, source: str, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, array)
    elif path.suffix == ".npz":
        np.savez_compressed(path, motion=array, source=source, representation=target)
    else:
        raise ValueError("output must end in .npy or .npz")


def main() -> None:
    args = parse_args()
    source_key = args.src.lower().replace("-", "").replace("_", "")
    motion = load_array(
        args.input,
        args.input_key,
        preserve_mapping=source_key in {"smpl", "smplparams", "smplhparams"},
    )
    kwargs = {"rotation_space": args.rotation_space}
    if args.bone_offsets:
        kwargs["bone_offsets"] = np.load(args.bone_offsets, allow_pickle=False)
    if args.smpl_model_dir:
        kwargs["model_dir"] = args.smpl_model_dir
        kwargs["model_path"] = args.smpl_model_dir
    for name in (
        "gender",
        "model_type",
        "src_fps",
        "dst_fps",
        "coordinate_system",
        "resample",
        "feet_threshold",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.device:
        kwargs["device"] = args.device
    if args.refine_iters is not None:
        kwargs["refine_iters"] = args.refine_iters

    converted = convert_motion(motion, args.src, args.dst, **kwargs)
    if not isinstance(converted, np.ndarray):
        converted = converted.detach().cpu().numpy()
    save_array(args.output, converted, args.src, args.dst)
    input_shape = (
        {key: value.shape for key, value in motion.items()}
        if isinstance(motion, Mapping)
        else motion.shape
    )
    print(f"{args.src} {input_shape} -> {args.dst} {converted.shape}: {args.output}")


if __name__ == "__main__":
    main()
