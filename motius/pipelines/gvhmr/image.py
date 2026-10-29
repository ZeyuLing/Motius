"""Released affine crop and ImageNet normalization."""
import cv2
import numpy as np
import torch
IMAGE_MEAN=torch.tensor([.485,.456,.406])
IMAGE_STD=torch.tensor([.229,.224,.225])


def crop_and_resize(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (H, W, 3)
        bbx_xy: (2,)
        bbx_s: scalar
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack([bbx_xy - hs, bbx_xy + np.array([hs, -hs]), bbx_xy]).astype(np.float32)
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)
    A = cv2.getAffineTransform(src, dst)
    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return (img_crop, bbx_xys_final)
