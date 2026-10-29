import torch
import torch.nn.functional as F
import numpy as np
from motius.models.vision.gvhmr import GVHMRPoseBundle
from .features import get_batch
from tqdm import tqdm
from .heatmaps import keypoints_from_heatmaps
from .flipping import flip_heatmap_coco17

class VitPoseExtractor:

    def __init__(self, checkpoint, device='cuda', tqdm_leave=True):
        self.device = torch.device(device)
        self.pose = GVHMRPoseBundle(checkpoint, device=device)
        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path
        L, _, H, W = imgs.shape
        batch_size = 16
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc='ViTPose', leave=self.tqdm_leave):
            imgs_batch = imgs[j:j + batch_size, :, :, 32:224].to(self.device)
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())
            bbx_xys_batch = bbx_xys[j:j + batch_size]
            heatmap = heatmap.clone().cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
            kp2d = np.concatenate((preds, maxvals), axis=-1)
            kp2d = torch.from_numpy(kp2d)
            vitpose.append(kp2d.detach().cpu().clone())
        vitpose = torch.cat(vitpose, dim=0).clone()
        return vitpose
