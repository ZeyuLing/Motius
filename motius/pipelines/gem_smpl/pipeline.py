"""Repository-native monocular capture pipeline for NVIDIA GEM-SMPL."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from motius.models.gem_smpl import GemSmplBundle
from motius.motion.representation.monocular_capture import MonocularCaptureResult
from motius.pipelines.base_pipeline import BasePipeline
from motius.pipelines.gem_smpl.parser import (
    load_gem_smpl_payload,
    parse_gem_smpl_file,
)
from motius.registry import PIPELINES
from motius.utils.monocular_parity import MonocularParityTrace


def _video_fps(path: str | Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            "Could not determine input video FPS; pass original_fps explicitly."
        )
    return fps


def _capture_official_trace(
    result_path: Path,
    *,
    trace_path: str | Path,
    static_camera: bool,
    seed: Optional[int],
    deterministic: bool,
) -> MonocularParityTrace:
    """Capture all persisted boundaries of the fixed official four-stage demo."""

    import torch

    output_dir = result_path.parent
    preprocess = output_dir / "preprocess"
    trace = MonocularParityTrace(
        "motius-gem-smpl",
        metadata={
            "static_camera": bool(static_camera),
            "seed": seed,
            "deterministic": bool(deterministic),
            "contract": "exact persisted-stage parity",
        },
    )
    trace.capture(
        "01_tracking",
        {"bbx_xys": torch.load(preprocess / "bbx.pt", map_location="cpu")},
    )
    trace.capture(
        "02_keypoints",
        {"kp2d": torch.load(preprocess / "vitpose.pt", map_location="cpu")},
    )
    features = torch.load(preprocess / "vit_features.pt", map_location="cpu")
    trace.capture(
        "03_visual_features",
        {
            "f_imgseq": features,
            "has_img_mask": torch.ones(len(features), dtype=torch.bool),
        },
    )
    trace.capture(
        "04_model_input",
        torch.load(
            preprocess / "model_input.pt",
            map_location="cpu",
            weights_only=False,
        ),
    )
    trace.capture("05_model_output", load_gem_smpl_payload(result_path))
    trace.save(trace_path)
    return trace


def _append_result_trace(
    trace_path: str | Path,
    result: MonocularCaptureResult,
) -> None:
    trace = MonocularParityTrace.load(trace_path)
    track = result.tracks[0]
    trace.capture(
        "06_geometry",
        {
            "joints_camera": track.joints_camera,
            "joints_world": track.joints_world,
            "vertices_camera": track.vertices_camera,
            "vertices_world": track.vertices_world,
        },
    )
    trace.capture(
        "07_public_result",
        {
            "frame_ids": track.frame_ids,
            "valid": track.valid,
            "poses_axis_angle": track.poses_axis_angle,
            "shape_parameters": track.shape_parameters,
            "joints_camera": track.joints_camera,
            "joints_world": track.joints_world,
            "root_translation_camera": track.root_translation_camera,
            "root_translation_world": track.root_translation_world,
            "camera_intrinsics": result.camera_intrinsics,
            "frame_timestamps": result.frame_timestamps,
        },
    )
    trace.save(trace_path)


@PIPELINES.register_module()
class GemSmplPipeline(BasePipeline):
    """Run source-pinned GEM-SMPL through the standard Motius V2M API."""

    BUNDLE_CLS = "motius.models.gem_smpl.GemSmplBundle"

    def __init__(self, bundle: GemSmplBundle):
        super().__init__(bundle)

    def infer_monocular_motion_capture(
        self,
        video: str | Path,
        output_root: str | Path,
        *,
        original_fps: Optional[float] = None,
        static_camera: bool = False,
        render: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        precomputed_stage_dir: Optional[str | Path] = None,
        materialize_geometry: bool = True,
        parity_trace: Optional[str | Path] = None,
        extra_args: Sequence[str] = (),
        timeout: Optional[float] = None,
    ) -> MonocularCaptureResult:
        """Recover camera/world SMPL motion from a monocular RGB video."""

        fps = _video_fps(video) if original_fps is None else float(original_fps)
        official = self.bundle.run_official_demo(
            video,
            output_root,
            static_camera=static_camera,
            render=render,
            seed=seed,
            deterministic=deterministic,
            precomputed_stage_dir=precomputed_stage_dir,
            extra_args=extra_args,
            timeout=timeout,
        )
        if parity_trace is not None:
            _capture_official_trace(
                official,
                trace_path=parity_trace,
                static_camera=static_camera,
                seed=seed,
                deterministic=deterministic,
            )
        numeric = (
            self.bundle.convert_official_result(official, timeout=timeout)
            if materialize_geometry
            else official
        )
        result = parse_gem_smpl_file(numeric, original_fps=fps)
        if parity_trace is not None:
            _append_result_trace(parity_trace, result)
        return result

    def run(self, video: str | Path, output_root: str | Path, **kwargs):
        """Compatibility alias for the task-specific inference method."""

        return self.infer_monocular_motion_capture(video, output_root, **kwargs)

    def parse_output(
        self,
        source: str | Path,
        *,
        original_fps: float,
    ) -> MonocularCaptureResult:
        return parse_gem_smpl_file(source, original_fps=original_fps)

    def __call__(self, video: str | Path, output_root: str | Path, **kwargs):
        return self.infer_monocular_motion_capture(video, output_root, **kwargs)


GemSmplMonocularPipeline = GemSmplPipeline


__all__ = ["GemSmplMonocularPipeline", "GemSmplPipeline"]
