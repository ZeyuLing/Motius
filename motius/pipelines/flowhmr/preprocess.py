"""Asset configuration for the pinned official FlowHMR video front end."""
import os
import os.path as osp
import shutil
import importlib.util
from pathlib import Path
from typing import Optional
import torch

class FlowHMRDependencyError(RuntimeError):
    pass

class FlowHMRVideoPreprocessor:
    def __init__(
        self,
        device: Optional[str] = None,
        *,
        sam3d_ckpt: Optional[str] = None,
        sam3d_mhr: Optional[str] = None,
        sam3d_config: Optional[str] = None,
        yolox_ckpt: Optional[str] = None,
        ffmpeg: Optional[str] = None,
        sam_token_only: bool = False,
        sam_fast: bool = False,
        sam_optimize_conditioning: bool = True,
        sam_frame_batch_size: int = 1,
        sam_precision: str = "source",
    ):
        if isinstance(sam_frame_batch_size, bool) or not isinstance(sam_frame_batch_size, int) or sam_frame_batch_size < 1:
            raise ValueError('sam_frame_batch_size must be a positive integer')
        if sam_frame_batch_size > 1 and not (sam_fast or sam_token_only):
            raise ValueError('SAM frame batching requires token-only conditioning')
        if sam_precision not in ("source", "float16", "float32"):
            raise ValueError("sam_precision must be source, float16 or float32")
        self.sam_precision = sam_precision
        self.sam_checkpoint_dtypes = []
        self.sam_frame_batch_size = sam_frame_batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sam3d_ckpt = (
            sam3d_ckpt
            or os.environ.get("FLOWHMR_SAM3D_CKPT")
        )
        self.sam3d_mhr = (
            sam3d_mhr
            or os.environ.get("FLOWHMR_SAM3D_MHR")
        )
        self.sam3d_config = (
            sam3d_config
            or os.environ.get("FLOWHMR_SAM3D_CONFIG")
        )
        self.yolox_ckpt = yolox_ckpt or os.environ.get("FLOWHMR_YOLOX_CKPT")
        self.ffmpeg = ffmpeg or os.environ.get("FLOWHMR_FFMPEG") or "ffmpeg"

        self._detector = None
        self._sam = None
        self.sam_token_only = bool(sam_token_only)
        self.sam_fast = bool(sam_fast)
        self.sam_optimize_conditioning = bool(sam_optimize_conditioning)

    @staticmethod
    def _release_cuda_cache():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_detector(self):
        self._detector = None
        self._release_cuda_cache()

    def release_sam(self):
        if self._sam is not None:
            self._sam.close()
            self._sam = None
        self._release_cuda_cache()

    def release(self):
        self.release_detector()
        self.release_sam()

    @staticmethod
    def _require_module(name: str, hint: str) -> None:
        if importlib.util.find_spec(name) is None:
            raise FlowHMRDependencyError(
                f"Python package '{name}' is required for V2M video preprocessing "
                f"but is not installed. {hint}"
            )

    def _require_ffmpeg(self) -> None:
        if shutil.which(self.ffmpeg) is None and not osp.isfile(self.ffmpeg):
            raise FlowHMRDependencyError(
                f"ffmpeg binary not found ('{self.ffmpeg}'). Install ffmpeg or set "
                "FLOWHMR_FFMPEG to a valid binary."
            )

    def _require_file(self, path: Optional[str], what: str, env: str, extra: str = "") -> None:
        if not path:
            raise FlowHMRDependencyError(
                f"{what} path is not configured. Set {env} (or pass the matching "
                f"constructor argument). {extra}"
            )
        if not osp.isfile(path):
            raise FlowHMRDependencyError(f"{what} weight not found: {path}. {extra}")

    def build_detector(self):
        from motius.pipelines.flowhmr.tracking import YOLOXHumanDetector
        if self._detector is None:
            self._require_file(self.yolox_ckpt, "YOLOX", "FLOWHMR_YOLOX_CKPT")
            self._detector = YOLOXHumanDetector(self.yolox_ckpt, self.device)
        return self._detector

    def detect_and_track(self, video_path, **kwargs):
        from motius.pipelines.flowhmr.tracking import VideoProcessor
        callback = getattr(self, "progress_callback", None)
        return VideoProcessor(self.build_detector()).compute_bboxes(video_path,
            progress=(lambda current, total, frame: callback("Track selected person", current+1, total)) if callback else None, **kwargs), None

    def build_sam_extractor(self):
        from motius.pipelines.flowhmr.sam.sam_3d_body_estimator import SAM3DBodyEstimator
        from motius.pipelines.flowhmr.sam.build_models import load_sam_3d_body
        from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
        from motius.pipelines.flowhmr.features import _maybe_convert_backbone_to_fp16
        if self._sam is None:
            self._require_file(self.sam3d_ckpt, "SAM", "FLOWHMR_SAM3D_CKPT")
            self._require_file(self.sam3d_mhr, "MHR", "FLOWHMR_SAM3D_MHR")
            precision_args = {} if self.sam_precision == "source" else {"precision": self.sam_precision}
            model, cfg = load_sam_3d_body(checkpoint_path=self.sam3d_ckpt,
                mhr_path=self.sam3d_mhr, config_path=self.sam3d_config or "", device=self.device, **precision_args)
            self.sam_checkpoint_dtypes = list(getattr(model, "checkpoint_backbone_dtypes", []))
            if self.device.type == "cuda":
                _maybe_convert_backbone_to_fp16(model, self.device)
            self._sam = Sam3DTokenExtractor(SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg),
                                          token_only=self.sam_token_only, fast=self.sam_fast,
                                          optimize_conditioning=self.sam_optimize_conditioning)
        return self._sam

    def extract_features(self, video_path, bbox_xyxy, camera_K, **kwargs):
        return self.build_sam_extractor().extract_video_tokens(video_path,
            bbox_xyxy=torch.as_tensor(bbox_xyxy), K_all=torch.as_tensor(camera_K),
            frame_batch_size=self.sam_frame_batch_size,
            progress_cb=(lambda stage, current, total, message: self.progress_callback("Extract body and hand features", current, total))
                if getattr(self, "progress_callback", None) else None, **kwargs)

    def transcode_to_30fps(self, video_path, out_dir, max_frames=None, fps=30):
        from motius.pipelines.flowhmr.tracking import VideoProcessor
        # Unique invocation directory avoids upstream's filename-only cache reuse.
        import tempfile
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output = tempfile.mkdtemp(prefix="transcode-", dir=out_dir)
        return VideoProcessor.transcode(video_path, output,
            duration=None if max_frames is None else max_frames / fps, force_fps=fps,
            ffmpeg_executable=self.ffmpeg)

    def prepare_bboxes(self, video_path, bbox_xyxy, **kwargs):
        import cv2
        from motius.pipelines.flowhmr.tracking import VideoProcessor
        boxes = torch.as_tensor(bbox_xyxy, dtype=torch.float32).clone()
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or not torch.isfinite(boxes).all():
            raise ValueError("Expected finite (T,4) xyxy boxes")
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return torch.from_numpy(VideoProcessor.smooth_bboxes(boxes.numpy(), width, height)), None
