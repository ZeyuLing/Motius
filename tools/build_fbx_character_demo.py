#!/usr/bin/env python3
"""Build a synchronized skeleton, SMPL FBX, and character-binding demo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion import (  # noqa: E402
    SMPLAnimation,
    export_motion_to_fbx,
    export_smpl_fbx,
)


def _load_motion(path: Path, frames: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.ndarray):
        motion = value
    else:
        try:
            key = "motion_135" if "motion_135" in value else value.files[0]
            motion = np.asarray(value[key])
        finally:
            value.close()
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 135:
        raise ValueError(f"Expected motion135 with shape (T,135), got {motion.shape}.")
    if len(motion) < frames:
        raise ValueError(f"Requested {frames} frames, but {path} contains {len(motion)}.")
    return motion[:frames]


def _label(identifier: str) -> str:
    return identifier.rsplit("/", 1)[-1].replace("_", " ").title()


def build(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion = _load_motion(args.motion135, args.frames)
    outputs = {"smpl": args.output_dir / "smpl.fbx"}
    outputs.update(
        {identifier: args.output_dir / f"{identifier.rsplit('/', 1)[-1]}.fbx" for identifier in args.characters}
    )
    if not args.skip_export:
        export_smpl_fbx(
            SMPLAnimation.from_motion135(motion, fps=args.fps),
            outputs["smpl"],
            model_path=args.model_path,
            backend=args.backend,
            source_metadata={"case_id": args.case_id, "source_representation": "motion135"},
        )
        for identifier in args.characters:
            export_motion_to_fbx(
                motion,
                "motion135",
                identifier,
                outputs[identifier],
                model_path=args.model_path,
                source_fps=args.fps,
                output_fps=args.fps,
                backend=args.backend,
            )
    missing = [str(path) for path in outputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing exported FBX files: {missing}.")

    representation = json.loads(args.representation_data.read_text())
    skeleton = representation["representations"]["humanml3d"]
    if len(skeleton["positions"]) < args.frames:
        raise ValueError("Representation demo does not contain enough skeleton frames.")
    panels = [
        {
            "kind": "skeleton",
            "label": "SMPL-22 Skeleton",
            "parents": skeleton["parents"],
            "positions": skeleton["positions"][: args.frames],
        },
        {"kind": "fbx", "label": "SMPL Mesh", "file": outputs["smpl"].name},
    ]
    panels.extend(
        {
            "kind": "fbx",
            "label": _label(identifier),
            "identifier": identifier,
            "file": outputs[identifier].name,
        }
        for identifier in args.characters
    )
    payload = {
        "schema_version": 1,
        "case_id": args.case_id,
        "fps": args.fps,
        "frames": args.frames,
        "panels": panels,
    }
    (args.output_dir / "data.js").write_text(
        "window.MOTIUS_FBX_CHARACTER_DEMO="
        + json.dumps(payload, separators=(",", ":"))
        + ";\n"
    )
    shutil.copy2(ROOT / "tools" / "fbx_character_demo_viewer.html", args.output_dir / "index.html")
    manifest = {
        **payload,
        "motion135": str(args.motion135),
        "representation_data": str(args.representation_data),
        "model_path": str(args.model_path),
        "viewer": str(args.output_dir / "index.html"),
        "fbx_outputs": {key: str(value) for key, value in outputs.items()},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion135", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--representation-data",
        type=Path,
        default=Path("assets/motion/representation_demo/data.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fbx_character_demo/004822"))
    parser.add_argument("--case-id", default="004822")
    parser.add_argument("--characters", nargs="+", default=("mixamo/amy", "mixamo/maria", "mixamo/michelle", "mixamo/remy"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", choices=("auto", "fbxsdk", "blender"), default="fbxsdk")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
