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
    ):
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
        from .vendor.flowhmr.utils.runtime_video_processing import YOLOXHumanDetector
        if self._detector is None:
            self._require_file(self.yolox_ckpt, "YOLOX", "FLOWHMR_YOLOX_CKPT")
            self._detector = YOLOXHumanDetector(self.yolox_ckpt, self.device)
        return self._detector

    def detect_and_track(self, video_path, **kwargs):
        from .vendor.flowhmr.utils.runtime_video_processing import VideoProcessor
        return VideoProcessor(self.build_detector()).compute_bboxes(video_path, **kwargs), None

    def build_sam_extractor(self):
        from .vendor.sam3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from .vendor.flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor, _maybe_convert_backbone_to_fp16
        if self._sam is None:
            self._require_file(self.sam3d_ckpt, "SAM", "FLOWHMR_SAM3D_CKPT")
            self._require_file(self.sam3d_mhr, "MHR", "FLOWHMR_SAM3D_MHR")
            model, cfg = load_sam_3d_body(checkpoint_path=self.sam3d_ckpt,
                mhr_path=self.sam3d_mhr, config_path=self.sam3d_config or "", device=self.device)
            if self.device.type == "cuda":
                _maybe_convert_backbone_to_fp16(model, self.device)
            self._sam = Sam3DTokenExtractor(SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg))
        return self._sam

    def extract_features(self, video_path, bbox_xyxy, camera_K, **kwargs):
        return self.build_sam_extractor().extract_video_tokens(video_path,
            bbox_xyxy=torch.as_tensor(bbox_xyxy), K_all=torch.as_tensor(camera_K), **kwargs)

    def transcode_to_30fps(self, video_path, out_dir, max_frames=None, fps=30):
        from .vendor.flowhmr.utils.runtime_video_processing import VideoProcessor
        # Unique invocation directory avoids upstream's filename-only cache reuse.
        import tempfile
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output = tempfile.mkdtemp(prefix="transcode-", dir=out_dir)
        return VideoProcessor.transcode(video_path, output,
            duration=None if max_frames is None else max_frames / fps, force_fps=fps,
            ffmpeg_executable=self.ffmpeg)

    def prepare_bboxes(self, video_path, bbox_xyxy, **kwargs):
        import cv2
        from .vendor.flowhmr.utils.runtime_video_processing import VideoProcessor
        boxes = torch.as_tensor(bbox_xyxy, dtype=torch.float32).clone()
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or not torch.isfinite(boxes).all():
            raise ValueError("Expected finite (T,4) xyxy boxes")
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return torch.from_numpy(VideoProcessor.smooth_bboxes(boxes.numpy(), width, height)), None
