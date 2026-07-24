"""HyMotion-V2M stage-2 video preprocessor: raw video -> SAM-3D feature stream.

Faithful, self-contained port of the original ``V2MRuntime`` preprocessing
chain (``hymotion/utils/v2m_runtime.py`` + ``sam3d_wrapper.py`` + ``bbox.py``):

    mp4 -> ffmpeg transcode @30fps
        -> YOLOX human detection + ByteTrack -> best single-person bbox track
        -> per-frame SAM-3D-Body token (body / body+lhand+rhand)
        -> pinhole camera (identity extrinsics)
        -> {feature (T, D), camera_RT (T,4,4), camera_K (T,3,3), camera_is_static}

The heavy stage-2 dependencies (``ffmpeg`` binary, ``yolox``, ``supervision``,
the ``sam_3d_body`` package and its **gated** ``facebook/sam-3d-body-dinov3``
weights) are imported / resolved **lazily**, so importing this module — and the
stage-1 feature->motion pipeline — never requires them.  When something is
missing, :class:`V2MDependencyError` is raised with an actionable message.

This module is fully self-contained: it does **not** import or point at any
external project's source tree.  ``sam_3d_body`` is treated as a third-party
dependency exactly like ``yolox`` / ``supervision`` — installed via pip, or a
checkout location the *caller* supplies explicitly.  Nothing here defaults to
another user's repository.

Resource resolution order (each path): explicit constructor arg > environment
variable > (for ckpts) derived from an explicitly-given ``sam3d_repo``.  If a
required resource is not resolvable, :class:`V2MDependencyError` says exactly
which arg / env var to set.  Configure via env:

    HYMOTION_V2M_SAM3D_REPO   # dir holding the ``sam_3d_body`` package + checkpoints
                              # (only needed if sam_3d_body is not pip-installed)
    HYMOTION_V2M_SAM3D_CKPT   # sam-3d-body-dinov3/model.ckpt
    HYMOTION_V2M_SAM3D_MHR    # sam-3d-body-dinov3/assets/mhr_model.pt
    HYMOTION_V2M_YOLOX_CKPT   # yolox_l.pth
    HYMOTION_V2M_FFMPEG       # ffmpeg binary (default: ``ffmpeg`` on PATH)
"""

from __future__ import annotations

import importlib.util
import math
import os
import os.path as osp
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

class V2MDependencyError(RuntimeError):
    """Raised when a stage-2 dependency (package / binary / weight) is missing."""


# ---------------------------------------------------------------------------
# bbox geometry helpers (vendored from hymotion/utils/bbox.py)
# ---------------------------------------------------------------------------
def _bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0):
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center, scale = center[0], scale[0]
    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _get_warp_matrix(center, scale, rot, output_size, shift=(0.0, 0.0)) -> np.ndarray:
    import cv2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w, dst_h = output_size[0], output_size[1]
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))


def _warp_intrinsics(K_full: np.ndarray, warp_mat_2x3: np.ndarray) -> np.ndarray:
    A = np.eye(3, dtype=np.float32)
    A[:2, :3] = warp_mat_2x3.astype(np.float32)
    return A @ K_full.astype(np.float32)


# ---------------------------------------------------------------------------
# SAM-3D-Body token extractor (hooks on decoder / decoder_hand)
# ---------------------------------------------------------------------------
class _Sam3DTokenExtractor:
    """Wraps an unmodified ``SAM3DBodyEstimator`` and captures body/hand tokens.

    Replicates the original ``Sam3DTokenExtractor``: during a ``full`` pass the
    internal ``decoder`` emits the body token and ``decoder_hand`` emits the
    left- then right-hand tokens; we concat ``token[0,0,:]`` of each → 3072-D.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.faces = estimator.faces
        self._body_tokens: List[Tensor] = []
        self._hand_tokens: List[Tensor] = []
        model = estimator.model
        model.decoder.register_forward_hook(
            lambda _m, _i, o: self._body_tokens.append(o[0].detach())
        )
        model.decoder_hand.register_forward_hook(
            lambda _m, _i, o: self._hand_tokens.append(o[0].detach())
        )

    @torch.no_grad()
    def extract_frame(self, frame_rgb, bboxes, cam_int) -> Tensor:
        self._body_tokens.clear()
        self._hand_tokens.clear()
        all_out = self.estimator.process_one_image(
            frame_rgb,
            bboxes=bboxes,
            masks=None,
            cam_int=cam_int,
            det_cat_id=0,
            bbox_thr=0.5,
            nms_thr=0.3,
            use_mask=False,
            inference_type="full",
        )
        if not all_out:
            # no person detected in crop -> zero token (3072)
            return torch.zeros(3072)
        if len(self._body_tokens) < 1 or len(self._hand_tokens) < 2:
            raise RuntimeError(
                "SAM token capture failed: "
                f"body={len(self._body_tokens)}, hand={len(self._hand_tokens)}; "
                "expected >=1 body and >=2 hand tokens in full mode."
            )
        body = self._body_tokens[0][0, 0, :]
        lhand = self._hand_tokens[0][0, 0, :]
        rhand = self._hand_tokens[1][0, 0, :]
        return torch.cat([body, lhand, rhand]).cpu()


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class V2MVideoPreprocessor:
    """Video -> SAM-3D feature stream + pinhole camera (stage-2 front end)."""

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        sam3d_repo: Optional[str] = None,
        sam3d_ckpt: Optional[str] = None,
        sam3d_mhr: Optional[str] = None,
        yolox_ckpt: Optional[str] = None,
        ffmpeg: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # NOTE: no path defaults to any external project. ``sam3d_repo`` is only
        # consulted when ``sam_3d_body`` is not pip-importable, and must be
        # supplied by the caller (arg or env var).
        self.sam3d_repo = sam3d_repo or os.environ.get("HYMOTION_V2M_SAM3D_REPO")
        self.sam3d_ckpt = (
            sam3d_ckpt
            or os.environ.get("HYMOTION_V2M_SAM3D_CKPT")
            or self._derive_from_repo("checkpoints/sam-3d-body-dinov3/model.ckpt")
        )
        self.sam3d_mhr = (
            sam3d_mhr
            or os.environ.get("HYMOTION_V2M_SAM3D_MHR")
            or self._derive_from_repo(
                "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
            )
        )
        self.yolox_ckpt = yolox_ckpt or os.environ.get("HYMOTION_V2M_YOLOX_CKPT")
        self.ffmpeg = ffmpeg or os.environ.get("HYMOTION_V2M_FFMPEG") or "ffmpeg"

        self._detector = None
        self._sam = None

    def _derive_from_repo(self, rel: str) -> Optional[str]:
        """Derive a ckpt path from an explicitly-given ``sam3d_repo`` only."""
        return osp.join(self.sam3d_repo, rel) if self.sam3d_repo else None

    # -- dependency checks -------------------------------------------------
    @staticmethod
    def _require_module(name: str, hint: str) -> None:
        if importlib.util.find_spec(name) is None:
            raise V2MDependencyError(
                f"Python package '{name}' is required for V2M video preprocessing "
                f"but is not installed. {hint}"
            )

    def _require_ffmpeg(self) -> None:
        if shutil.which(self.ffmpeg) is None and not osp.isfile(self.ffmpeg):
            raise V2MDependencyError(
                f"ffmpeg binary not found ('{self.ffmpeg}'). Install ffmpeg or set "
                "HYMOTION_V2M_FFMPEG to a valid binary."
            )

    def _require_file(self, path: Optional[str], what: str, env: str, extra: str = "") -> None:
        if not path:
            raise V2MDependencyError(
                f"{what} path is not configured. Set {env} (or pass the matching "
                f"constructor arg / sam3d_repo). {extra}"
            )
        if not osp.isfile(path):
            raise V2MDependencyError(f"{what} weight not found: {path}. {extra}")

    # -- lazy builders -----------------------------------------------------
    def _load_sam_package(self):
        """Dynamically load the ``sam_3d_body`` package from ``sam3d_repo``."""
        if importlib.util.find_spec("sam_3d_body") is not None:
            import sam_3d_body  # type: ignore

            return sam_3d_body
        if not self.sam3d_repo:
            raise V2MDependencyError(
                "sam_3d_body is not installed. Either `pip install` the SAM-3D-Body "
                "package (facebook/sam-3d-body), or set HYMOTION_V2M_SAM3D_REPO to a "
                "checkout containing the 'sam_3d_body' package."
            )
        pkg_dir = osp.join(self.sam3d_repo, "sam_3d_body")
        init_py = osp.join(pkg_dir, "__init__.py")
        if not osp.isfile(init_py):
            raise V2MDependencyError(
                "sam_3d_body package not importable and not found under "
                f"'{pkg_dir}'. Point HYMOTION_V2M_SAM3D_REPO at the sam-3d-body "
                "checkout (containing the 'sam_3d_body' package)."
            )
        spec = importlib.util.spec_from_file_location(
            "sam_3d_body", init_py, submodule_search_locations=[pkg_dir]
        )
        if spec is None or spec.loader is None:
            raise V2MDependencyError(f"Failed to load sam_3d_body from {init_py}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sam_3d_body"] = mod
        spec.loader.exec_module(mod)
        return mod

    def build_sam_extractor(self) -> _Sam3DTokenExtractor:
        if self._sam is not None:
            return self._sam
        self._require_file(
            self.sam3d_ckpt,
            "SAM-3D-Body",
            "HYMOTION_V2M_SAM3D_CKPT",
            "Request access to the gated repo facebook/sam-3d-body-dinov3 on "
            "HuggingFace, download model.ckpt + assets/mhr_model.pt, and point "
            "HYMOTION_V2M_SAM3D_CKPT / _MHR at them.",
        )
        self._require_file(self.sam3d_mhr, "SAM-3D-Body MHR", "HYMOTION_V2M_SAM3D_MHR")
        sam_mod = self._load_sam_package()
        load_sam_3d_body = getattr(sam_mod, "load_sam_3d_body")
        SAM3DBodyEstimator = getattr(sam_mod, "SAM3DBodyEstimator")
        sam_model, sam_cfg = load_sam_3d_body(
            checkpoint_path=self.sam3d_ckpt,
            mhr_path=self.sam3d_mhr,
            device=self.device,
        )
        estimator = SAM3DBodyEstimator(sam_3d_body_model=sam_model, model_cfg=sam_cfg)
        self._sam = _Sam3DTokenExtractor(estimator)
        return self._sam

    def build_detector(self):
        if self._detector is not None:
            return self._detector
        self._require_module(
            "yolox", "Install YOLOX (pip install yolox) for human detection."
        )
        self._require_file(
            self.yolox_ckpt,
            "YOLOX",
            "HYMOTION_V2M_YOLOX_CKPT",
            "Download yolox_l.pth (Megvii YOLOX release) and set HYMOTION_V2M_YOLOX_CKPT.",
        )
        from yolox.data.data_augment import ValTransform  # type: ignore
        from yolox.exp import get_exp  # type: ignore
        from yolox.utils import postprocess  # type: ignore

        exp = get_exp(None, "yolox-l")
        model = exp.get_model()
        ckpt = torch.load(self.yolox_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        self._detector = {
            "model": model,
            "num_classes": exp.num_classes,
            "preproc": ValTransform(),
            "postprocess": postprocess,
            "test_size": (640, 640),
        }
        return self._detector

    # -- detection ---------------------------------------------------------
    def _detect_one(self, frame_bgr, bbox_thr=0.4, nms_thr=0.3) -> np.ndarray:
        det = self.build_detector()
        h, w = frame_bgr.shape[:2]
        ratio = min(det["test_size"][0] / h, det["test_size"][1] / w)
        img_pre, _ = det["preproc"](frame_bgr, None, det["test_size"])
        img_tensor = torch.from_numpy(img_pre).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = det["model"](img_tensor)
            outputs = det["postprocess"](
                outputs, det["num_classes"], bbox_thr, nms_thr
            )
        if outputs[0] is None:
            return np.empty((0, 5), dtype=np.float32)
        d = outputs[0].cpu().numpy()  # (N,7) x1y1x2y2,obj,cls,clsid
        d = d[d[:, 6].astype(int) == 0]  # person only
        if len(d) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes = d[:, :4] / ratio
        scores = d[:, 4] * d[:, 5]
        return np.hstack([boxes, scores[:, None]]).astype(np.float32)

    def detect_and_track(
        self,
        video_path: str,
        *,
        bbox_thr: float = 0.4,
        nms_thr: float = 0.3,
        bytetrack_thresh: float = 0.25,
        bytetrack_match: float = 0.8,
        smooth_window: int = 5,
        rescale: float = 1.05,
    ) -> Tuple[Tensor, np.ndarray]:
        """YOLOX + ByteTrack -> best single-person bbox per frame + warp mats."""
        import cv2
        self._require_module(
            "supervision", "Install supervision (pip install supervision) for ByteTrack."
        )
        import supervision as sv

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_box = np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32)

        tracker = sv.ByteTrack(
            track_activation_threshold=bytetrack_thresh,
            minimum_matching_threshold=bytetrack_match,
        )
        tracks = defaultdict(lambda: defaultdict(list))
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            res = self._detect_one(frame_bgr, bbox_thr, nms_thr)
            if len(res) > 0:
                xyxy = res[:, :4].astype(np.float32)
                conf = res[:, 4].astype(np.float32)
                cls = np.zeros(len(res), dtype=int)
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                conf = np.empty((0,), dtype=np.float32)
                cls = np.empty((0,), dtype=int)
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            dets = tracker.update_with_detections(dets)
            tids = dets.tracker_id if dets.tracker_id is not None else np.empty((0,), dtype=np.int64)
            for tid, box in zip(tids, dets.xyxy):
                tracks[int(tid)]["bboxes"].append(box.astype(np.float32))
                tracks[int(tid)]["frames"].append(frame_idx)
            frame_idx += 1
        cap.release()
        if frame_idx <= 0:
            raise ValueError("no frames read from video")

        # best track = (length, avg_area)
        if len(tracks) == 0:
            bbox_arr = np.tile(full_box[None, :], (frame_idx, 1)).astype(np.float32)
        else:
            stats = []
            for tid, v in tracks.items():
                bb = np.stack(v["bboxes"], axis=0).astype(np.float32)
                areas = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
                stats.append((int(tid), bb.shape[0], float(np.mean(areas))))
            stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best = int(stats[0][0])
            bf = np.asarray(tracks[best]["frames"], dtype=np.int64)
            bb = np.stack(tracks[best]["bboxes"], axis=0).astype(np.float32)
            order = np.argsort(bf)
            bf, bb = bf[order], bb[order]
            all_frames = np.arange(frame_idx, dtype=np.float32)
            bbox_arr = np.empty((frame_idx, 4), dtype=np.float32)
            if bf.size == 1:
                bbox_arr[:] = bb[0]
            else:
                for j in range(4):
                    bbox_arr[:, j] = np.interp(
                        all_frames, bf.astype(np.float32), bb[:, j],
                        left=float(bb[0, j]), right=float(bb[-1, j]),
                    ).astype(np.float32)

        # smooth (two passes of moving average)
        if smooth_window > 1 and bbox_arr.shape[0] > 1:
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            for _ in range(2):
                for j in range(4):
                    pad = smooth_window // 2
                    padded = np.pad(bbox_arr[:, j], (pad, pad), mode="edge")
                    bbox_arr[:, j] = np.convolve(padded, kernel, mode="valid")

        # squarify + rescale
        centers, scales = _bbox_xyxy2cs(bbox_arr, padding=rescale)
        scales_fixed = _fix_aspect_ratio(scales, aspect_ratio=1.0)
        half = scales_fixed * 0.5
        bbox_sq = np.concatenate([centers - half, centers + half], axis=1).astype(np.float32)
        bbox_sq[:, 0] = np.clip(bbox_sq[:, 0], 0, width - 1)
        bbox_sq[:, 1] = np.clip(bbox_sq[:, 1], 0, height - 1)
        bbox_sq[:, 2] = np.clip(bbox_sq[:, 2], 1, width)
        bbox_sq[:, 3] = np.clip(bbox_sq[:, 3], 1, height)

        T = bbox_sq.shape[0]
        warp_mats = np.empty((T, 2, 3), dtype=np.float32)
        for t in range(T):
            warp_mats[t] = _get_warp_matrix(
                centers[t], scales_fixed[t], rot=0.0, output_size=(512, 512)
            )
        return torch.from_numpy(bbox_sq).float(), warp_mats

    # -- camera ------------------------------------------------------------
    @staticmethod
    def _default_K(height: int, width: int) -> np.ndarray:
        f = float((height ** 2 + width ** 2) ** 0.5)
        return np.array(
            [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def estimate_camera(self, video_path: str, num_frames: int) -> Tuple[Tensor, Tensor]:
        """Pinhole intrinsics from H,W; identity extrinsics (static camera).

        TODO(vipe): integrate NVIDIA ViPE (https://github.com/nv-tlabs/vipe) to
        replace the static-camera assumption with real per-frame camera motion
        for in-the-wild video.  Blocked for now: no ViPE-capable environment
        (ViPE needs its own heavy env -- torch 2.7.0+cu128 / transformers
        4.48.3 -- incompatible with this repo's env; the dev box is a T4 without
        ViPE installed).  Design notes from research, so this is a quick lift
        once an env is available:

          * Run mode (recommended): a self-contained *adapter* that consumes
            ViPE artifacts (run ``vipe infer <video> -o <dir>`` in a separate
            ViPE env; this pipeline never imports ``vipe``).  Resolve the
            artifacts dir / ViPE command via ``HYMOTION_V2M_VIPE_*`` env vars,
            mirroring the SAM/YOLOX lazy-dependency pattern above.
          * Reading is dependency-free: ViPE's pose artifact is a plain npz with
            ``data`` = (N, 4, 4) **camera-to-world** matrices and ``inds`` =
            frame indices; intrinsics artifact holds per-frame (fx, fy, cx, cy)
            for a PINHOLE model.  No ``vipe`` import needed to read them.
          * Convert: ``camera_RT = inv(c2w)`` (world->camera, our convention),
            then resample ``inds``->dense 30fps frame grid to match the feature
            stream length.  Set ``camera_is_static=False`` and infer
            ``movement_type`` from the extrinsic frame-to-frame motion.
          * Gravity/world-frame caveat (correctness-critical): ViPE anchors its
            world frame to the first camera and is NOT guaranteed y-up, while
            ``HyMotionV2MPipeline._compute_wv_transform`` assumes world +y is
            gravity-up (BEDLAM training convention).  ``R_to_first`` (camera_R)
            is frame-relative and convention-invariant, but the velocity term
            (camera_T) is decomposed along the up axis, so a wrong up axis
            corrupts the "human motion vs camera motion" disentanglement.
            Start with a configurable ``world_up`` (first-frame approximation),
            optionally upgrade to gravity from the ViPE depth/SLAM ground plane.
            Validate against a clip where GT camera is available before trusting.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        K0 = self._default_K(height, width)
        K_all = np.repeat(K0[None], num_frames, axis=0)
        RT_all = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))
        return torch.from_numpy(K_all).float(), torch.from_numpy(RT_all).float()

    # -- transcode ---------------------------------------------------------
    def transcode_to_30fps(
        self, video_path: str, out_dir: str, max_frames: Optional[int] = None, fps: int = 30
    ) -> str:
        self._require_ffmpeg()
        os.makedirs(out_dir, exist_ok=True)
        base = osp.splitext(osp.basename(video_path))[0]
        out_path = osp.join(out_dir, f"{base}_{fps}fps.mp4")
        cmd = [self.ffmpeg, "-y", "-i", video_path]
        if max_frames is not None and max_frames > 0:
            cmd += ["-t", f"{(max_frames + 0.5) / fps:.6f}"]
        cmd += [
            "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-an", "-avoid_negative_ts", "make_zero", out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg transcode failed: {(e.stderr or '')[:500]}")
        return out_path

    # -- features ----------------------------------------------------------
    def extract_features(
        self,
        video_path: str,
        bbox_xyxy: Tensor,
        K_all: Tensor,
        token_dim: int,
        max_frames: Optional[int] = None,
    ) -> Tensor:
        import cv2
        sam = self.build_sam_extractor()
        bbox_xyxy = bbox_xyxy.detach().cpu().float()
        K_all = K_all.detach().cpu().float()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        feats: List[Tensor] = []
        i = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                idx = min(i, bbox_xyxy.shape[0] - 1)
                b = bbox_xyxy[idx].reshape(1, 4).numpy()
                cam = K_all[idx].reshape(1, 3, 3)
                token = sam.extract_frame(frame_rgb, bboxes=b, cam_int=cam)
                feats.append(token[:token_dim].float())
                i += 1
                if max_frames is not None and max_frames > 0 and i >= max_frames:
                    break
        finally:
            cap.release()
        if not feats:
            raise ValueError("no frames decoded for feature extraction")
        return torch.stack(feats, dim=0)

    # -- orchestrator ------------------------------------------------------
    def run(
        self,
        video_path: str,
        *,
        token_dim: int,
        work_dir: str,
        max_frames: Optional[int] = None,
        transcode: bool = True,
        bbox_xyxy: Optional[Tensor] = None,
    ) -> dict:
        """Full preprocessing: video -> feature + camera conditioning dict.

        Returns ``{feature (T,token_dim), camera_RT (T,4,4), camera_K (T,3,3),
        camera_is_static}``.  ``camera_K`` is the **crop** intrinsics (matching
        the original v2m model input); extrinsics are identity (static camera).
        """
        used = (
            self.transcode_to_30fps(video_path, osp.join(work_dir, "raw_videos"), max_frames)
            if transcode
            else video_path
        )
        if bbox_xyxy is None:
            bbox_xyxy, warp_mats = self.detect_and_track(used)
        else:
            bbox_xyxy, warp_mats = self.prepare_bboxes(
                used,
                bbox_xyxy,
                smooth_window=1,
            )
        T = int(bbox_xyxy.shape[0])
        K_all, RT_all = self.estimate_camera(used, T)
        # crop intrinsics for the motion model (SAM uses full-image K_all)
        K_np = K_all.numpy()
        K_crop = np.stack(
            [_warp_intrinsics(K_np[t], warp_mats[t]) for t in range(T)], axis=0
        )
        feature = self.extract_features(
            used, bbox_xyxy, K_all, token_dim=token_dim, max_frames=max_frames
        )
        L = int(feature.shape[0])
        return {
            "feature": feature,
            "camera_RT": RT_all[:L],
            "camera_K": torch.from_numpy(K_crop[:L]).float(),
            "camera_K_full": K_all[:L],
            "camera_is_static": True,
            "bbox_xyxy": bbox_xyxy[:L],
            "video_path": used,
        }

    @staticmethod
    def prepare_bboxes(
        video_path: str,
        bbox_xyxy: Tensor,
        *,
        smooth_window: int = 5,
        rescale: float = 1.05,
    ) -> Tuple[Tensor, np.ndarray]:
        """Validate, smooth, and square an externally supplied dense track."""

        import cv2

        boxes = np.asarray(torch.as_tensor(bbox_xyxy).cpu(), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
            raise ValueError(
                f"bbox_xyxy must have shape (frames, 4), got {boxes.shape}."
            )
        if not np.isfinite(boxes).all():
            raise ValueError("bbox_xyxy contains non-finite values.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if smooth_window > 1 and len(boxes) > 1:
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            pad = smooth_window // 2
            for _ in range(2):
                for coordinate in range(4):
                    padded = np.pad(
                        boxes[:, coordinate],
                        (pad, pad),
                        mode="edge",
                    )
                    boxes[:, coordinate] = np.convolve(
                        padded,
                        kernel,
                        mode="valid",
                    )

        centers, scales = _bbox_xyxy2cs(boxes, padding=rescale)
        scales = _fix_aspect_ratio(scales, aspect_ratio=1.0)
        half = scales * 0.5
        boxes = np.concatenate(
            [centers - half, centers + half],
            axis=1,
        ).astype(np.float32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)

        warp_mats = np.stack(
            [
                _get_warp_matrix(
                    centers[index],
                    scales[index],
                    rot=0.0,
                    output_size=(512, 512),
                )
                for index in range(len(boxes))
            ],
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(boxes), warp_mats
