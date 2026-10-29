import datetime
import os
import os.path as osp
import sys
import time
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor


from motius.pipelines.flowhmr.sam.sam_3d_body_estimator import SAM3DBodyEstimator
from motius.pipelines.flowhmr.sam.build_models import load_sam_3d_body
_repo_root = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))

# Checkpoint paths
_sam_ckpt_dir = osp.join(_repo_root, "ckpts", "sam-3d-body-dinov3")
SAM3D_CKPT_PATH = osp.join(_sam_ckpt_dir, "model.ckpt")
SAM3D_MHR_PATH = osp.join(_sam_ckpt_dir, "assets", "mhr_model.pt")

# Visualize function (module-level cache)
_visualize_fn_cache = None


def get_visualize_fn():
    global _visualize_fn_cache
    if _visualize_fn_cache is not None:
        return _visualize_fn_cache

    from motius.motion.representation.sam3d_metadata.mhr70 import pose_info as mhr70_pose_info
    from motius.visualization.flowhmr_sam import build_visualize_fn

    _visualize_fn_cache = build_visualize_fn(mhr70_pose_info)
    return _visualize_fn_cache


class _TokensCaptured(Exception):
    """Private early return once all three conditioning tokens are available."""


class Sam3DTokenExtractor:
    """Capture original SAM network tokens through official or Fast GPU frontends.

    Fast uses the same network/weights and splits paired [left,right] decoder batches;
    it never substitutes the prompted body token or average hand features.
    """

    def __init__(self, estimator, *, token_only=False, fast=False, optimize_conditioning=True):
        self.estimator = estimator
        self.token_only = bool(token_only)
        self.fast = bool(fast)
        if self.fast and not self.token_only:
            raise ValueError('Fast SAM requires full-hand token-only conditioning')
        self._capture_active = False
        self._conditioning_cache = {} if optimize_conditioning else False
        self.faces = estimator.faces
        self._body_tokens: List[torch.Tensor] = []
        self._hand_tokens: List[torch.Tensor] = []
        self._hand_left_chunks: List[torch.Tensor] = []
        self._hand_right_chunks: List[torch.Tensor] = []
        self._hooks = self._install_hooks()

    def _install_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        model = self.estimator.model
        body_tokens = self._body_tokens
        hand_tokens = self._hand_tokens

        def _body_hook(_module, _input, output):
            # output = (pose_token, pose_output); pose_token: (B*P, num_tok, D)
            body_tokens.append(output[0].detach())

        def _hand_hook(_module, _input, output):
            value = output[0].detach()
            if self.fast and self._capture_active:
                count = body_tokens[0].shape[0]
                if value.shape[0] % 2:
                    raise RuntimeError('Fast SAM must return paired [left, right] hand batches')
                left, right = value.chunk(2, dim=0)
                self._hand_left_chunks.append(left)
                self._hand_right_chunks.append(right)
                captured = sum(chunk.shape[0] for chunk in self._hand_left_chunks)
                if captured > count:
                    raise RuntimeError('Fast SAM captured more hand tokens than body frames')
                if captured == count:
                    hand_tokens.extend((torch.cat(self._hand_left_chunks, 0), torch.cat(self._hand_right_chunks, 0)))
            else:
                hand_tokens.append(value)
            if self.token_only and self._capture_active and len(hand_tokens) == 2:
                # FlowHMR consumes body[0], left[0], right[0]. The subsequent
                # MHR reconstruction/keypoint-prompt pass cannot affect them.
                raise _TokensCaptured

        h1 = model.decoder.register_forward_hook(_body_hook)
        h2 = model.decoder_hand.register_forward_hook(_hand_hook)
        return [h1, h2]

    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if isinstance(self._conditioning_cache, dict):
            self._conditioning_cache.clear()
        self._body_tokens.clear()
        self._hand_tokens.clear()
        self._hand_left_chunks.clear()
        self._hand_right_chunks.clear()

    # Core: single-frame extraction
    @torch.no_grad()
    def extract_frame(
        self,
        img: Union[str, np.ndarray],
        bboxes=None,
        masks=None,
        cam_int=None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
        is_vis: bool = False,
    ) -> Tuple[Tensor, list]:
        self._body_tokens.clear()
        self._hand_tokens.clear()
        self._hand_left_chunks.clear()
        self._hand_right_chunks.clear()

        if self.token_only and (inference_type != 'full' or is_vis):
            raise ValueError('token_only requires full body/hand feature extraction without visualization')
        captured = False
        self._capture_active = True
        try:
            if self.fast:
                if masks is not None or use_mask or bboxes is None or cam_int is None:
                    raise ValueError('Fast SAM requires explicit selected boxes/intrinsics and no mask prompt')
                if isinstance(img, str):
                    from .sam_data.utils.io import load_image
                    img = load_image(img, backend='cv2', image_format='rgb')
                from .sam.inference import run_fast_conditioning_inference
                all_out = run_fast_conditioning_inference(self.estimator, img, bboxes, cam_int,
                                                          cache=self._conditioning_cache)
            else:
                all_out = self.estimator.process_one_image(
                    img, bboxes=bboxes, masks=masks, cam_int=cam_int,
                    det_cat_id=det_cat_id, bbox_thr=bbox_thr, nms_thr=nms_thr,
                    use_mask=use_mask, inference_type=inference_type,
                    **({"token_only": True, "conditioning_cache": self._conditioning_cache if masks is None and not use_mask else False} if self.token_only else {}),
                )
        except _TokensCaptured:
            captured, all_out = True, []
        finally:
            self._capture_active = False

        if not captured and not all_out:
            return torch.zeros(0), all_out

        if inference_type != "full":
            return torch.zeros(0), all_out

        if len(self._body_tokens) < 1 or len(self._hand_tokens) < 2:
            raise RuntimeError(
                f"SAM token capture failed: "
                f"body_tokens={len(self._body_tokens)}, hand_tokens={len(self._hand_tokens)}"
            )
        # body_tokens: [body_first_pass, keypoint_prompt_pass]
        # hand_tokens: [lhand_pass, rhand_pass]
        body_token = self._body_tokens[0]
        lhand_token = self._hand_tokens[0]
        rhand_token = self._hand_tokens[1]

        all_token = torch.cat(
            [
                body_token[0, 0, :],
                lhand_token[0, 0, :],
                rhand_token[0, 0, :],
            ]
        ).cpu()

        return all_token, all_out

    @torch.no_grad()
    def extract_frames(self, images, boxes, cameras):
        """Batched independent frames with body/left/right token order preserved."""
        if not self.token_only:
            raise ValueError('Frame batches require full-hand conditioning')
        if not images or len(images) != len(boxes) or len(images) != len(cameras):
            raise ValueError('Expected equally sized nonempty images/boxes/cameras')
        from .sam.inference import run_fast_frame_batch, run_official_frame_batch
        self._body_tokens.clear()
        self._hand_tokens.clear()
        self._hand_left_chunks.clear()
        self._hand_right_chunks.clear()
        self._capture_active = True
        try:
            if self.fast:
                run_fast_frame_batch(self.estimator, images, boxes, cameras, cache=self._conditioning_cache)
            else:
                run_official_frame_batch(self.estimator, images, boxes, cameras, cache=self._conditioning_cache)
        except _TokensCaptured:
            pass
        finally:
            self._capture_active = False
        if len(self._body_tokens) != 1 or len(self._hand_tokens) != 2:
            raise RuntimeError('Batched SAM did not capture body and both hands')
        return torch.cat([tokens[:, 0] for tokens in (self._body_tokens[0], *self._hand_tokens)], -1).cpu()

    # Helpers
    @staticmethod
    def _get_total_frames(video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                total += 1
        cap.release()
        return total

    @staticmethod
    def _pad_crop(tensor: Tensor, target_len: int) -> Tensor:
        if target_len < 0:
            raise ValueError(f"target_len must be >= 0, got {target_len}")
        if tensor.ndim == 0:
            raise ValueError("tensor must have at least 1 dimension")

        src_len = int(tensor.shape[0])
        if target_len == 0:
            return tensor[:0]

        if src_len == 0:
            out_shape = (target_len, *tensor.shape[1:])
            return torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)

        if src_len < target_len:
            pad_shape = [target_len - src_len] + [1] * (tensor.ndim - 1)
            pad = tensor[-1:].repeat(*pad_shape)
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor[:target_len]

    # Video-level: extract tokens for all frames
    def extract_video_tokens(
        self,
        video_path: str,
        *,
        bbox_xyxy: Tensor,
        K_all: Tensor,
        token_dim: int = 3072,
        inference_type: str = "full",
        use_mask: bool = False,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        max_frames: Optional[int] = None,
        frame_batch_size: int = 1,
        progress_cb: Optional[Callable[[str, int, Optional[int], str], None]] = None,
    ) -> Tensor:
        if inference_type != "full":
            raise ValueError("extract_video_tokens only supports inference_type='full'")

        if isinstance(frame_batch_size, bool) or not isinstance(frame_batch_size, int) or frame_batch_size < 1:
            raise ValueError('frame_batch_size must be a positive integer')
        if frame_batch_size > 1 and not self.token_only:
            raise ValueError('Frame batching requires token-only full-hand conditioning')
        total_frames = self._get_total_frames(video_path)

        bbox_xyxy = bbox_xyxy.detach().cpu().to(torch.float32)
        K_all = K_all.detach().cpu().to(torch.float32)
        bbox_xyxy = self._pad_crop(bbox_xyxy, total_frames)
        K_all = self._pad_crop(K_all, total_frames)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")

        effective_max = max_frames if (max_frames is not None and max_frames > 0) else None
        pbar_total = min(total_frames, effective_max) if effective_max is not None else total_frames

        _call_progress(progress_cb, "sam", 0, pbar_total, "sam: start")

        feats: List[Tensor] = []
        t_start = time.time()
        try:
            i = 0
            while i < pbar_total:
                images = []
                for _ in range(min(frame_batch_size, pbar_total-i)):
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    images.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if not images:
                    break
                previous = i
                if frame_batch_size == 1:
                    token, _outputs = self.extract_frame(images[0],
                        bboxes=bbox_xyxy[i].reshape(1, 4).numpy(),
                        cam_int=K_all[i].reshape(1, 3, 3), bbox_thr=bbox_thr,
                        nms_thr=nms_thr, use_mask=use_mask, inference_type=inference_type)
                    feats.append(token[:token_dim].to(torch.float32))
                else:
                    if use_mask:
                        raise ValueError('Batched SAM does not support mask prompts')
                    tokens = self.extract_frames(images, bbox_xyxy[i:i+len(images)].numpy(), K_all[i:i+len(images)])
                    feats.extend(tokens[:, :token_dim].to(torch.float32))
                i += len(images)
                _call_progress(progress_cb, "sam", i, pbar_total, "sam: running")
                if i//10 != previous//10 or i == pbar_total:
                    elapsed = time.time() - t_start
                    fps = i / elapsed if elapsed > 0 else 0
                    eta = (pbar_total - i) / fps if fps > 0 else 0
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"    [{now_str}] SAM extract: {i}/{pbar_total} "
                        f"({100 * i / pbar_total:.1f}%) "
                        f"| {elapsed:.1f}s elapsed, {fps:.2f} fps, ETA {eta:.1f}s"
                    )
                if effective_max is not None and i >= effective_max:
                    break
        finally:
            cap.release()
            _call_progress(progress_cb, "sam", len(feats), pbar_total, "sam: done")

        if len(feats) == 0:
            raise ValueError(f"no frames extracted from video: {video_path}")
        return torch.stack(feats, dim=0)

    # Debug: visualize SAM outputs to video
    def debug_sam_video(
        self,
        video_path: str,
        *,
        output_dir: str,
        bbox_xyxy: Optional[Tensor] = None,
        K_all: Optional[Tensor] = None,
        inference_type: str = "full",
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        motion_fps: int = 30,
        vis_stride: int = 1,
        max_vis_frames: Optional[int] = 300,
        save_video_name: str = "debug_sam_vis.mp4",
    ) -> str:
        """Render SAM 3D body estimation results on each frame and save as video."""
        if vis_stride < 1:
            raise ValueError(f"vis_stride must be >= 1, got {vis_stride}")
        os.makedirs(output_dir, exist_ok=True)

        visualize_sample_together = get_visualize_fn()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")

        out_path = osp.join(output_dir, save_video_name)
        writer = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pad / crop bbox & K to match total_frames
        if bbox_xyxy is not None and K_all is not None:
            tf = total_frames if total_frames > 0 else 1
            bbox_xyxy = self._pad_crop(bbox_xyxy.detach().cpu().to(torch.float32), tf)
            K_all = self._pad_crop(K_all.detach().cpu().to(torch.float32), tf)

        i = 0
        written = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if i % vis_stride != 0:
                    i += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if bbox_xyxy is not None and K_all is not None:
                    b = bbox_xyxy[i].reshape(1, 4)
                    cam_int = K_all[i].reshape(1, 3, 3)
                    _token, outputs = self.extract_frame(
                        frame_rgb,
                        bboxes=b,
                        bbox_thr=bbox_thr,
                        nms_thr=nms_thr,
                        use_mask=use_mask,
                        inference_type=inference_type,
                        is_vis=True,
                        cam_int=cam_int,
                    )
                else:
                    _token, outputs = self.extract_frame(
                        frame_rgb,
                        bbox_thr=bbox_thr,
                        nms_thr=nms_thr,
                        use_mask=use_mask,
                        inference_type=inference_type,
                        is_vis=True,
                    )

                if outputs:
                    rend = visualize_sample_together(frame_bgr, outputs, self.faces)
                    rend = np.clip(rend, 0, 255).astype(np.uint8)
                    if writer is None:
                        h, w = rend.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                        writer = cv2.VideoWriter(out_path, fourcc, float(motion_fps), (w, h))
                    writer.write(rend)
                    written += 1
                    if max_vis_frames is not None and written >= max_vis_frames:
                        break

                i += 1
                if i % 100 == 0:
                    print(f">>> sam debug vis: {i}/{total_frames} frames, written={written}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        print(f">>> SAM debug saved: {out_path} ({written} frames)")
        return out_path


# Helper: progress callback (standalone)
def _call_progress(
    progress_cb: Optional[Callable],
    stage: str,
    current: int,
    total: Optional[int],
    message: str,
) -> None:
    """Invoke progress callback if provided."""
    if progress_cb is not None:
        progress_cb(stage, current, total, message)


def _maybe_convert_backbone_to_fp16(sam_model, device: torch.device) -> None:
    if sam_model.backbone_dtype != torch.bfloat16:
        return
    gpu_supports_bf16_tc = torch.cuda.get_device_capability(device)[0] >= 8
    if gpu_supports_bf16_tc:
        return

    print(f" [sam3d] GPU {torch.cuda.get_device_name(device)} has no BF16 Tensor Cores, "
          f"converting backbone to FP16")
    for p in sam_model.backbone.parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float16)
    for buf in sam_model.backbone.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
    sam_model.backbone_dtype = torch.float16


def build_sam3d_extractor(device: torch.device) -> Sam3DTokenExtractor:
    if not osp.exists(SAM3D_CKPT_PATH):
        raise FileNotFoundError(f"SAM3D checkpoint not found: {SAM3D_CKPT_PATH}")
    if not osp.exists(SAM3D_MHR_PATH):
        raise FileNotFoundError(f"SAM3D MHR checkpoint not found: {SAM3D_MHR_PATH}")

    sam_model, sam_cfg = load_sam_3d_body(
        checkpoint_path=SAM3D_CKPT_PATH,
        mhr_path=SAM3D_MHR_PATH,
        device=device,
    )
    _maybe_convert_backbone_to_fp16(sam_model, device)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=sam_model, model_cfg=sam_cfg)
    return Sam3DTokenExtractor(estimator)
