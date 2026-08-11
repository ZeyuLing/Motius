#!/usr/bin/env python3
"""Generate native SOMA-30 KIMODO demos through the public Pipeline APIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius import Pipeline
from motius.motion.retarget.smpl_soma import SOMA30_IN_SOMA77, SOMA30_PARENTS


def _joints(result: dict) -> np.ndarray:
    values = np.asarray(result["posed_joints"], dtype=np.float32)
    while values.ndim > 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"unexpected KIMODO posed_joints shape: {values.shape}")
    if values.shape[1] == 77:
        values = values[:, SOMA30_IN_SOMA77]
    if values.shape[1] != len(SOMA30_PARENTS):
        raise ValueError(
            "KIMODO native preview expects SOMA-30 or expanded SOMA-77 joints, "
            f"got {values.shape[1]}"
        )
    return values


def _save(
    output: Path,
    result: dict,
    caption: str,
    case_id: str,
    **extra,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joints=_joints(result)[:, None],
        parents=np.asarray(SOMA30_PARENTS, dtype=np.int32),
        caption=np.asarray(caption),
        case_id=np.asarray(case_id),
        fps=np.asarray(30, dtype=np.int32),
        **extra,
    )


def _validate_runtime_devices(pipe: Pipeline, requested_device: str) -> None:
    if not str(requested_device).startswith("cuda"):
        return
    model = pipe.bundle.model
    denoiser_device = next(model.denoiser.parameters()).device
    text_device = model.text_encoder.get_device()
    if denoiser_device.type != "cuda" or text_device.type != "cuda":
        raise RuntimeError(
            "KIMODO was requested on CUDA but did not place every inference "
            f"component there (denoiser={denoiser_device}, text={text_device})"
        )
    print(
        "KIMODO runtime devices: "
        f"denoiser={denoiser_device}, text_encoder={text_device}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pipe = Pipeline.from_pretrained(args.artifact, device=args.device)
    _validate_runtime_devices(pipe, args.device)
    prompts = [
        "a person walks forward",
        "the person turns right and waves",
        "the person sits down",
    ]
    frames = [60, 60, 90]
    sequential = pipe.infer_sequential_text_to_motion(prompts, frames)
    boundaries = np.cumsum([0, *frames])
    _save(
        args.output_dir / "sequential_text_to_motion.npz",
        sequential,
        "three ordered text segments",
        "kimodo_sequential_text_to_motion",
        segment_frames=np.stack(
            [boundaries[:-1], boundaries[1:]],
            axis=-1,
        ).astype(np.int32),
        segment_captions=np.asarray(prompts),
    )

    temporal_caption = "a person walks forward and then turns right"
    temporal_reference = pipe.infer_text_to_motion(
        temporal_caption,
        120,
    )
    condition_frames = 30
    temporal = pipe.infer_temporal_motion_completion(
        [temporal_caption],
        [temporal_reference],
        condition_frames=condition_frames,
    )[0]
    _save(
        args.output_dir / "temporal_motion_completion.npz",
        temporal,
        temporal_caption,
        "kimodo_temporal_motion_completion",
        condition_intervals=np.asarray(
            [[0, condition_frames]],
            dtype=np.int32,
        ),
    )

    keyframes = np.asarray([0, 30, 60, 90, 119], dtype=np.int32)
    waypoints = np.asarray(
        [[0.0, 0.0], [0.45, 0.15], [0.9, 0.25], [1.35, 0.15], [1.8, 0.0]],
        dtype=np.float32,
    )
    root_path = pipe.root2d_constraint(
        frame_indices=keyframes.tolist(),
        smooth_root_2d=waypoints.tolist(),
    )
    controlled = pipe.infer_kinematic_motion_control(
        "a person follows a curved walking path",
        num_frames=120,
        constraints=[root_path],
    )
    dense_frame = np.arange(120)
    trajectory = np.stack(
        [
            np.interp(dense_frame, keyframes, waypoints[:, 0]),
            np.zeros(120, dtype=np.float32),
            np.interp(dense_frame, keyframes, waypoints[:, 1]),
        ],
        axis=-1,
    ).astype(np.float32)
    _save(
        args.output_dir / "kinematic_motion_control.npz",
        controlled,
        "text plus a constrained root trajectory",
        "kimodo_kinematic_motion_control",
        trajectory=trajectory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
