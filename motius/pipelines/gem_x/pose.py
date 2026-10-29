import numpy as np
import torch
from tqdm import tqdm
def keypoints_from_heatmaps(heatmaps, center, scale, use_udp=True):
    """Standard argmax + affine coordinate transform."""
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = flat.argmax(-1)
    px = (idx % W).astype(np.float32)
    py = (idx // W).astype(np.float32)
    if use_udp:
        for n in range(N):
            for k in range(K):
                hm = heatmaps[n, k]
                x, y = (int(px[n, k]), int(py[n, k]))
                if 1 < x < W - 1:
                    px[n, k] += np.sign(hm[y, x + 1] - hm[y, x - 1]) * 0.25
                if 1 < y < H - 1:
                    py[n, k] += np.sign(hm[y + 1, x] - hm[y - 1, x]) * 0.25
    preds = np.stack([px, py], axis=-1)
    preds[..., 0] = preds[..., 0] / W * (scale[:, [0]] * 200) + center[:, [0]] - scale[:, [0]] * 100
    preds[..., 1] = preds[..., 1] / H * (scale[:, [1]] * 200) + center[:, [1]] - scale[:, [1]] * 100
    maxvals = flat.max(-1, keepdims=True)
    return (preds, maxvals)
_VITPOSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VITPOSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def get_batch(video_or_np, bbx_xys, img_ds=1.0, path_type='np'):
    """Crop+resize+normalize video frames to (256, 256) input tensor."""
    import cv2
    if isinstance(video_or_np, np.ndarray):
        frames = video_or_np
    else:
        frames = [cv2.imread(p) for p in video_or_np]
    T = len(frames)
    out = np.zeros((T, 3, 256, 256), dtype=np.float32)
    for i, (frame, bxy) in enumerate(zip(frames, bbx_xys)):
        cx, cy, s = (float(bxy[0]), float(bxy[1]), float(bxy[2]))
        hs = s / 2
        src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
        dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_LINEAR)
        crop = crop[..., ::-1].astype(np.float32) / 255.0
        crop = (crop - _VITPOSE_MEAN) / _VITPOSE_STD
        out[i] = crop.transpose(2, 0, 1)
    return (torch.from_numpy(out), bbx_xys)

def flip_heatmap_soma77(output_flipped):
    assert output_flipped.ndim == 4
    batch_size, num_joints, _, _ = output_flipped.shape
    assert num_joints == 77, f'Expected 77 joints, got {num_joints}'
    x = output_flipped.reshape(batch_size, -1, 1, output_flipped.shape[2], output_flipped.shape[3])
    y = x.clone()
    pairs = [(9, 10), (11, 39), (12, 40), (13, 41), (14, 42), (15, 43), (16, 44), (17, 45), (18, 46), (19, 47), (20, 48), (21, 49), (22, 50), (23, 51), (24, 52), (25, 53), (26, 54), (27, 55), (28, 56), (29, 57), (30, 58), (31, 59), (32, 60), (33, 61), (34, 62), (35, 63), (36, 64), (37, 65), (38, 66), (67, 72), (68, 73), (69, 74), (70, 75), (71, 76)]
    for left, right in pairs:
        y[:, left, ...] = x[:, right, ...]
        y[:, right, ...] = x[:, left, ...]
    return y.reshape_as(output_flipped).flip(3)

class VitPoseExtractor:

    def __init__(self,checkpoint,device='cuda:0',pose_type='soma',tqdm_leave=True):
        from motius.models.vision.gem_x import GemXPoseBundle
        self.pose=GemXPoseBundle(checkpoint,device)
        self.device=device;self.flip_test=True;self.tqdm_leave=tqdm_leave

    @torch.no_grad()
    def extract(self, video_or_np, bbx_xys, img_ds=1.0, batch_size=16, path_type='np'):
        if isinstance(video_or_np, str | list | np.ndarray):
            imgs, bbx_xys = get_batch(video_or_np, bbx_xys, img_ds=img_ds, path_type=path_type)
        else:
            imgs = video_or_np
        total_frames = imgs.shape[0]
        results = []
        for j in tqdm(range(0, total_frames, batch_size), desc='ViTPose', leave=self.tqdm_leave):
            imgs_batch = imgs[j:j + batch_size, :, :, 32:224].to(self.device)
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_soma77(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
            else:
                heatmap = self.pose(imgs_batch.clone())
            bbx_xys_batch = bbx_xys[j:j + batch_size]
            heatmap_np = heatmap.clone().cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap_np, center=center, scale=scale, use_udp=True)
            kp2d = torch.from_numpy(np.concatenate((preds, maxvals), axis=-1))
            results.append(kp2d.detach().cpu())
        return torch.cat(results, dim=0).clone()
