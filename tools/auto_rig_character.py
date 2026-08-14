#!/usr/bin/env python3
"""Fit an SMPL22-compatible skeleton and skin a static humanoid character."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion import auto_rig_character


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically bind an upright T/A-pose humanoid GLB, GLTF, FBX, "
            "OBJ, PLY, or STL to Motius's SMPL22 skeleton."
        )
    )
    parser.add_argument("character", type=Path, help="Unrigged source character.")
    parser.add_argument("output", type=Path, help="Rigged .fbx, .glb, or .gltf output.")
    parser.add_argument(
        "--blender",
        type=Path,
        default=None,
        help="Blender 3.6+ executable (otherwise MOTIUS_BLENDER or PATH).",
    )
    parser.add_argument(
        "--up-axis",
        default="auto",
        choices=("auto", "X", "Y", "Z", "-X", "-Y", "-Z"),
        help=(
            "Source up axis. auto trusts GLB/GLTF/FBX metadata and assumes Z-up "
            "for OBJ/PLY/STL."
        ),
    )
    parser.add_argument("--top-k", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--weight-falloff", type=float, default=1.75)
    parser.add_argument("--side-penalty", type=float, default=0.025)
    parser.add_argument(
        "--weight-method",
        choices=("automatic", "capsules"),
        default="capsules",
        help="Deterministic capsules (default) or Blender bone heat.",
    )
    parser.add_argument(
        "--replace-existing-rig",
        action="store_true",
        help="Remove an existing armature and weights before re-rigging.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = auto_rig_character(
        args.character,
        args.output,
        blender_executable=args.blender,
        up_axis=args.up_axis,
        top_k=args.top_k,
        weight_falloff=args.weight_falloff,
        side_penalty=args.side_penalty,
        weight_method=args.weight_method,
        replace_existing_rig=args.replace_existing_rig,
    )
    print(f"Rigged character: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Armature: {result.armature_name}; meshes: {', '.join(result.mesh_names)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
