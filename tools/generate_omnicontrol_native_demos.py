#!/usr/bin/env python3
"""Generate native HumanML3D skeleton demos through the public OmniControl API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius import Pipeline
from motius.motion import convert_motion
from motius.motion.skeleton import SMPL22_PARENTS


def _resample(array: np.ndarray, source_fps: int, target_fps: int) -> np.ndarray:
    if source_fps == target_fps or len(array) < 2:
        return array
    frames = round((len(array) - 1) * target_fps / source_fps) + 1
    source = np.linspace(0.0, 1.0, len(array))
    target = np.linspace(0.0, 1.0, frames)
    flattened = array.reshape(len(array), -1)
    result = np.stack(
        [np.interp(target, source, flattened[:, index])
         for index in range(flattened.shape[1])],
        axis=-1,
    )
    return result.reshape((frames,) + array.shape[1:]).astype(np.float32)


def _save(
    output: Path,
    motion: np.ndarray,
    caption: str,
    case_id: str,
    *,
    condition: np.ndarray | None = None,
    condition_intervals: np.ndarray | None = None,
) -> None:
    joints = np.asarray(
        convert_motion(motion, "hml263", "joints"),
        dtype=np.float32,
    )
    joints = _resample(joints, 20, 30)
    payload = {
        "motion_hml263": np.asarray(motion, dtype=np.float32),
        "joints": joints[:, None],
        "parents": np.asarray(SMPL22_PARENTS, dtype=np.int32),
        "caption": np.asarray(caption),
        "case_id": np.asarray(case_id),
        "fps": np.asarray(30, dtype=np.int32),
    }
    if condition is not None:
        condition_joints = np.asarray(
            convert_motion(condition, "hml263", "joints"),
            dtype=np.float32,
        )
        condition_joints = _resample(condition_joints, 20, 30)
        payload["condition_joints"] = condition_joints[:, None]
    if condition_intervals is not None:
        payload["condition_intervals"] = np.asarray(
            condition_intervals,
            dtype=np.int32,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    pipe = Pipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"device": args.device},
    )
    caption = "a person walks forward and waves with the right hand"
    reference = pipe.infer_text_to_motion(
        [caption],
        [args.frames],
        seed=17,
    )[0]
    _save(
        args.output_dir / "text_to_motion.npz",
        reference,
        caption,
        "omnicontrol_text_to_motion",
    )

    temporal = pipe.infer_temporal_motion_completion(
        [caption],
        [reference],
        lengths=[len(reference)],
        control_mode="prefix",
        prefix_ratio=0.2,
        seed=19,
    )[0]
    _save(
        args.output_dir / "temporal_motion_completion.npz",
        temporal,
        caption,
        "omnicontrol_temporal_motion_completion",
        condition=reference,
        condition_intervals=np.asarray(
            [[0, round(len(reference) * 0.2 * 30 / 20)]],
            dtype=np.int32,
        ),
    )

    controlled = pipe.infer_kinematic_motion_control(
        [caption],
        [reference],
        lengths=[len(reference)],
        control_mode="trajectory",
        joint_indices=[21],
        axes="xyz",
        seed=23,
    )[0]
    _save(
        args.output_dir / "kinematic_motion_control.npz",
        controlled,
        "text plus a constrained right-wrist trajectory",
        "omnicontrol_kinematic_motion_control",
        condition=reference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
