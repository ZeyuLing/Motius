"""Repository-native monocular capture pipeline for NVIDIA GEM-X."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from motius.models.gem_x import GemXBundle
from motius.motion.representation.monocular_capture import MonocularCaptureResult
from motius.pipelines.base_pipeline import BasePipeline
from motius.pipelines.gem_x.composition import GemXComposition
from motius.pipelines.gem_x.parser import (
    load_gem_x_payload,
    parse_gem_x_file,
)
from motius.registry import PIPELINES
from motius.utils.deterministic_geometry import reproducible_geometry_when_requested
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
        "motius-gem-x",
        metadata={
            "static_camera": bool(static_camera),
            "seed": seed,
            "deterministic": bool(deterministic),
            "contract": "exact persisted-stage parity",
        },
    )
    trace.capture(
        "01_tracking",
        torch.load(preprocess / "bbx.pt", map_location="cpu"),
    )
    trace.capture(
        "02_keypoints",
        {"kp2d": torch.load(preprocess / "vitpose.pt", map_location="cpu")},
    )
    features = torch.load(preprocess / "vit_features.pt", map_location="cpu")
    trace.capture(
        "03_visual_features",
        {
            "f_imgseq": features["pose_tokens"],
            "noisy_pred_cam": features.get("pred_cam"),
            "has_img_mask": torch.ones(len(features["pose_tokens"]), dtype=torch.bool),
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
    trace.capture("05_model_output", load_gem_x_payload(result_path))
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
class GemXPipeline(BasePipeline, GemXComposition):
    """Run source-pinned GEM-X through the standard Motius V2M API."""

    BUNDLE_CLS = "motius.models.gem_x.GemXBundle"

    def __init__(self, bundle: GemXBundle):
        BasePipeline.__init__(self, bundle)
        GemXComposition.__init__(self, bundle)

    @reproducible_geometry_when_requested
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
        """Recover camera/world SOMA motion from a monocular RGB video."""

        fps = _video_fps(video) if original_fps is None else float(original_fps)
        import shutil
        import random
        import torch
        from motius.configuration.gem_x_provenance import expected_output_path
        from motius.pipelines.gem_x.frontend import prepare_video
        if extra_args:
            raise ValueError('The native Pipeline accepts explicit keyword arguments, not source CLI arguments')
        if timeout is not None:
            raise ValueError('timeout applies to external processes; this Pipeline executes in the caller process')
        official = expected_output_path(video, output_root)
        official.parent.mkdir(parents=True, exist_ok=True)
        preprocess = official.parent / 'preprocess'
        preprocess.mkdir(exist_ok=True)
        if precomputed_stage_dir is not None:
            source=Path(precomputed_stage_dir)
            for name in ('bbx.pt','vitpose.pt','vit_features.pt','slam_results.pt'):
                path=source/name
                if not path.is_file():raise FileNotFoundError(path)
                if path.resolve() != (preprocess/name).resolve():shutil.copy2(path,preprocess/name)
        # Match the source load order: initialize networks before seeding sampling.
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
        data=prepare_video(video,preprocess,self.bundle.artifact_root,device=self.bundle.device,static_camera=static_camera)
        torch.save(data,preprocess/'model_input.pt')
        if seed is not None:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        with torch.no_grad():prediction=self.predict(data,static_cam=static_camera)
        payload={group:{name:value.detach().cpu() for name,value in prediction[group].items()}
                 for group in ('body_params_incam','body_params_global')}
        payload['K_fullimg']=prediction['K_fullimg'].detach().cpu()
        torch.save(payload,official)
        if parity_trace is not None:
            _capture_official_trace(
                official,
                trace_path=parity_trace,
                static_camera=static_camera,
                seed=seed,
                deterministic=deterministic,
            )
        numeric=official
        if materialize_geometry or render:
            from motius.pipelines.gem_x.export_native import materialize_prediction
            arrays=materialize_prediction(payload,self.bundle.artifact_root/'inputs/soma_assets',self.bundle.device)
            numeric=official.with_suffix('.npz')
            np.savez_compressed(numeric,**arrays)
            if render:
                from motius.pipelines.gvhmr.rendering import render_prediction
                geometry={key:torch.from_numpy(value) for key,value in arrays.items() if '.' not in key}
                render_prediction(geometry,video,official.parent,None,self.bundle.device,faces=self.body_model.faces)
        result = parse_gem_x_file(numeric, original_fps=fps)
        if parity_trace is not None:
            _append_result_trace(parity_trace, result)
        return result

    def save_pretrained(self,save_directory,**kwargs):
        result=self.bundle.save_pretrained(save_directory,**kwargs)
        if self.tokenizer is not None:self.tokenizer.save_pretrained(Path(save_directory)/'text_encoder')
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
        return parse_gem_x_file(source, original_fps=original_fps)

    def __call__(self, video: str | Path, output_root: str | Path, **kwargs):
        return self.infer_monocular_motion_capture(video, output_root, **kwargs)


GemXMonocularPipeline = GemXPipeline


__all__ = ["GemXMonocularPipeline", "GemXPipeline"]
