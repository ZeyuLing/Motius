from __future__ import annotations
import os,sys
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from pathlib import Path

def detect_and_track(video_path: str, preprocess_dir: str, checkpoint, device='cuda') -> torch.Tensor:
    """Run YOLOv8 person detection, select top-1 track, smooth bboxes.

    Returns:
        bbx_xys: (L, 3) tensor of [center_x, center_y, bbox_size]
    """
    cache_path = os.path.join(preprocess_dir, 'bbx.pt')
    if os.path.exists(cache_path):
        print(f'[Stage 1a] Loading cached bounding boxes from {cache_path}')
        return torch.load(cache_path, map_location='cpu')
    print('[Stage 1a] Running YOLOv8 person detection ...')
    from ultralytics import YOLO
    from motius.pipelines.gem_video import get_video_lwh, read_video_np
    L, W, H = get_video_lwh(video_path)
    frames = read_video_np(video_path)
    from motius.pipelines.gvhmr.tracking import Tracker
    model = Tracker(checkpoint,device=device).yolo
    all_boxes: list[list[np.ndarray]] = [[] for _ in range(len(frames))]
    for i, frame in enumerate(tqdm(frames, desc='YOLO detect', leave=False)):
        frame_bgr = frame[..., ::-1].copy()
        results = model(frame_bgr, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                all_boxes[i].append(np.append(xyxy, conf))
    bbx_xyxy_list = []
    for i in range(len(frames)):
        if len(all_boxes[i]) == 0:
            if len(bbx_xyxy_list) > 0:
                bbx_xyxy_list.append(bbx_xyxy_list[-1].copy())
            else:
                bbx_xyxy_list.append(np.array([W / 4, H / 4, 3 * W / 4, 3 * H / 4]))
        else:
            boxes_np = np.stack(all_boxes[i])
            areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
            best = areas.argmax()
            bbx_xyxy_list.append(boxes_np[best, :4])
    bbx_xyxy = torch.tensor(np.stack(bbx_xyxy_list), dtype=torch.float32)
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
    from motius.motion.representation.gem_smpl.geometry import get_bbx_xys_from_xyxy
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)
    from motius.motion.representation.gem_smpl.smoothing import moving_average_smooth
    bbx_xys = moving_average_smooth(bbx_xys, window_size=5, dim=0)
    os.makedirs(preprocess_dir, exist_ok=True)
    torch.save(bbx_xys, cache_path)
    print(f'[Stage 1a] Bounding boxes saved to {cache_path} ({len(bbx_xys)} frames)')
    return bbx_xys

class CocoPoseExtractor:
    """Lightweight COCO-17 ViTPose-H extractor.

    Uses custom ViT-H backbone + deconv head matching the ViTPose checkpoint.
    If the checkpoint is unavailable, falls back to zero-confidence keypoints.
    """
    CKPT_PATH = 'inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth'
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    def __init__(self, checkpoint, device: str='cuda'):
        self.device = device
        self.model = None
        self._fallback = False
        ckpt_path = checkpoint
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'ViTPose-H COCO checkpoint not found at {ckpt_path}')
        self.model = self._build_model(ckpt_path)
        self.model.to(device).eval()

    @staticmethod
    def _build_model(ckpt_path):
        from motius.models.vision.gem_smpl import GemSmplPoseBundle
        return GemSmplPoseBundle(ckpt_path,device='cpu')

    def _preprocess(self, frames: np.ndarray, bbx_xys: torch.Tensor) -> torch.Tensor:
        """Crop person regions to 256x256 square, then slice to 256x192."""
        T = len(frames)
        out = np.zeros((T, 3, 256, 192), dtype=np.float32)
        for i in range(T):
            cx, cy, s = (float(bbx_xys[i, 0]), float(bbx_xys[i, 1]), float(bbx_xys[i, 2]))
            hs = s / 2
            src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
            dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            crop256 = cv2.warpAffine(frames[i], M, (256, 256), flags=cv2.INTER_LINEAR)
            crop = crop256[:, 32:224]
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - self.MEAN) / self.STD
            out[i] = crop.transpose(2, 0, 1)
        return torch.from_numpy(out)

    def _heatmap_to_keypoints(self, heatmaps: np.ndarray, bbx_xys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Argmax + subpixel refinement -> image-space keypoints.

        Args:
            heatmaps: (N, 17, H, W)
            bbx_xys: (N, 3)
        Returns:
            preds: (N, 17, 2)
            maxvals: (N, 17, 1)
        """
        N, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(N, K, -1)
        idx = flat.argmax(-1)
        px = (idx % W).astype(np.float32)
        py = (idx // W).astype(np.float32)
        for n in range(N):
            for k in range(K):
                hm = heatmaps[n, k]
                x, y = (int(px[n, k]), int(py[n, k]))
                if 1 < x < W - 1:
                    px[n, k] += np.sign(hm[y, x + 1] - hm[y, x - 1]) * 0.25
                if 1 < y < H - 1:
                    py[n, k] += np.sign(hm[y + 1, x] - hm[y - 1, x]) * 0.25
        preds = np.stack([px, py], axis=-1)
        cx = bbx_xys[:, 0:1]
        cy = bbx_xys[:, 1:2]
        sz = bbx_xys[:, 2:3]
        preds[..., 0] = preds[..., 0] / W * sz * (192.0 / 256.0) + (cx - sz * (192.0 / 256.0) / 2)
        preds[..., 1] = preds[..., 1] / H * sz + (cy - sz / 2)
        maxvals_raw = flat.max(-1)
        maxvals = (1 / (1 + np.exp(-maxvals_raw)))[..., np.newaxis]
        return (preds, maxvals)

    @staticmethod
    def _flip_heatmap_coco17(hm: torch.Tensor) -> torch.Tensor:
        """Flip heatmaps horizontally and swap left-right joint channels."""
        hm = hm.flip(-1)
        pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        hm_new = hm.clone()
        for left, right in pairs:
            hm_new[:, left] = hm[:, right]
            hm_new[:, right] = hm[:, left]
        return hm_new

    @torch.no_grad()
    def extract(self, frames: np.ndarray, bbx_xys: torch.Tensor, batch_size: int=16) -> torch.Tensor:
        """Extract COCO-17 keypoints.

        Args:
            frames: (L, H, W, 3) uint8 RGB
            bbx_xys: (L, 3)
        Returns:
            kp2d: (L, 17, 3) tensor [x, y, confidence]
        """
        L = len(frames)
        if self._fallback:
            print('[Stage 1b] Returning zero keypoints (no ViTPose checkpoint)')
            return torch.zeros(L, 17, 3)
        imgs = self._preprocess(frames, bbx_xys)
        results = []
        for j in tqdm(range(0, L, batch_size), desc='ViTPose COCO17', leave=False):
            batch = imgs[j:j + batch_size].to(self.device)
            heatmap_normal = self.model(batch)
            heatmap_flipped = self.model(batch.flip(-1))
            heatmap_flipped = self._flip_heatmap_coco17(heatmap_flipped)
            heatmap = (heatmap_normal + heatmap_flipped) * 0.5
            hm_np = heatmap.cpu().numpy()
            bxy_np = bbx_xys[j:j + batch_size].numpy()
            preds, maxvals = self._heatmap_to_keypoints(hm_np, bxy_np)
            kp = np.concatenate([preds, maxvals], axis=-1)
            results.append(torch.from_numpy(kp).float())
        return torch.cat(results, dim=0)

def get_image_features(video_path: str, bbx_xys: torch.Tensor, hmr2_ckpt: str, preprocess_dir: str, device='cuda') -> tuple[torch.Tensor, torch.Tensor]:
    """Extract HMR2 ViT features, with caching.

    Args:
        video_path: path to input video
        bbx_xys: (L, 3) bounding boxes
        hmr2_ckpt: path to HMR2 checkpoint
        preprocess_dir: directory for caching preprocessed data

    Returns:
        f_imgseq: (L, 1024) image features
        has_img_mask: (L,) bool mask (all True)
    """
    cache_path = os.path.join(preprocess_dir, 'vit_features.pt')
    if os.path.exists(cache_path):
        print(f'[Stage 1c] Loading cached HMR2 features from {cache_path}')
        f_imgseq = torch.load(cache_path, map_location='cpu')
        has_img_mask = torch.ones(f_imgseq.shape[0], dtype=torch.bool)
        return (f_imgseq, has_img_mask)
    if not os.path.exists(hmr2_ckpt):
        print(f"\n[ERROR] HMR2 checkpoint not found at {hmr2_ckpt}.\nDownload from GVHMR's Google Drive: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD\nPlace hmr2/epoch=10-step=25000.ckpt under inputs/checkpoints/\n")
        raise FileNotFoundError(hmr2_ckpt)
    print(f'[Stage 1c] Extracting HMR2 features (ckpt: {hmr2_ckpt}) ...')
    from motius.pipelines.gem_smpl.features import HMR2FeatureExtractor
    extractor = HMR2FeatureExtractor(hmr2_ckpt, device=device)
    f_imgseq = extractor.extract_video_features(video_path, bbx_xys, batch_size=16)
    os.makedirs(preprocess_dir, exist_ok=True)
    torch.save(f_imgseq, cache_path)
    print(f'[Stage 1c] HMR2 features saved to {cache_path} ({f_imgseq.shape})')
    has_img_mask = torch.ones(f_imgseq.shape[0], dtype=torch.bool)
    return (f_imgseq, has_img_mask)

def get_camera_static(L: int, width: int, height: int):
    """Static camera: identity rotations, zero velocity.

    Returns:
        R_w2c: (L, 3, 3)
        cam_angvel: (L, 6)
        cam_tvel: (L, 3)
        K_fullimg: (L, 3, 3)
    """
    from motius.motion.representation.gem_smpl.camera import estimate_K
    from motius.motion.representation.gem_smpl.geometry import compute_cam_angvel
    R_w2c = torch.eye(3).unsqueeze(0).expand(L, -1, -1).clone()
    cam_angvel = compute_cam_angvel(R_w2c, padding_last=True)
    cam_tvel = torch.zeros(L, 3)
    K = estimate_K(width, height)
    K_fullimg = K.unsqueeze(0).expand(L, -1, -1).clone()
    return (R_w2c, cam_angvel, cam_tvel, K_fullimg)


def prepare_video(video,preprocess_dir,artifact_root,device='cuda',static_camera=False):
    from motius.pipelines.gem_video import get_video_lwh,read_video_np
    video=str(video);preprocess_dir=str(preprocess_dir);root=Path(artifact_root)
    Path(preprocess_dir).mkdir(parents=True,exist_ok=True)
    length,width,height=get_video_lwh(video)
    boxes=detect_and_track(video,preprocess_dir,root/'yolov8x.pt',device)
    cache=Path(preprocess_dir)/'vitpose.pt'
    if cache.exists():keypoints=torch.load(cache,map_location='cpu',weights_only=True)
    else:
        frames=read_video_np(video)
        pose=CocoPoseExtractor(root/'inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth',device=device)
        keypoints=pose.extract(frames,boxes,batch_size=32)
        torch.save(keypoints,cache)
        del frames,pose
    features,image_mask=get_image_features(video,boxes,str(root/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt'),preprocess_dir,device=device)
    rotation,angular,translation,intrinsics=get_camera_static(length,width,height)
    valid=lambda count:torch.arange(length)<count
    return dict(kp2d=keypoints,bbx_xys=boxes,K_fullimg=intrinsics,cam_angvel=angular,
        cam_tvel=translation,R_w2c=rotation,f_imgseq=features,has_text=torch.tensor([False]),
        length=torch.tensor(length),meta=[{'mode':'default'}],mask=dict(has_img_mask=image_mask,
        has_2d_mask=valid(length),has_cam_mask=valid(0 if static_camera else length),
        has_audio_mask=valid(0),has_music_mask=valid(0)))
