"""Published camera-frame and foreground-mask preparation, without a CLI."""
import numpy as np
import torch
import cv2
from glob import glob
from torchvision.transforms import Resize

def est_calib(images):
    """ Roughly estimate intrinsics by image dimensions """
    image = images[0]
    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = (w0 / 2.0, h0 / 2.0)
    calib = [focal, focal, cx, cy]
    return calib

def get_dimention(imagedir):
    """ Get proper image dimension for DROID """
    if isinstance(imagedir, str):
        imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
        image = cv2.imread(imgfiles[0])
    elif isinstance(imagedir, np.ndarray):
        image = imagedir
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt(384 * 512 / (h0 * w0)))
    w1 = int(w0 * np.sqrt(384 * 512 / (h0 * w0)))
    image = cv2.resize(image, (w1, h1))
    image = image[:h1 - h1 % 8, :w1 - w1 % 8]
    H, W, _ = image.shape
    return (H, W)

def image_stream(images, calib, stride=1, max_frame=None, depths=None, depth_keyframes=None):
    """ Image generator for DROID """
    fx, fy, cx, cy = calib[:4]
    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    images = images[::stride]
    if max_frame is not None:
        images = images[:max_frame]
    for t, img in enumerate(images):
        image = img[..., ::-1]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt(384 * 512 / (h0 * w0)))
        w1 = int(w0 * np.sqrt(384 * 512 / (h0 * w0)))
        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0
        if depths is None:
            depth = None
        elif t in depth_keyframes:
            depth_idx = np.where(depth_keyframes == t)[0][0]
            depth = depths[depth_idx]
            depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_CUBIC)
            depth = depth[:h1 - h1 % 8, :w1 - w1 % 8]
            depth = torch.from_numpy(depth).float()
        else:
            depth = None
        size_factor = [w1 / w0, h1 / h0]
        yield (t, image[None], intrinsics, depth, size_factor)

def preprocess_masks(images, masks):
    """ Resize masks for masked droid """
    H, W = get_dimention(images[0])
    resize_1 = Resize((H, W), antialias=True)
    resize_2 = Resize((H // 8, W // 8), antialias=True)
    img_msks = []
    for i in range(0, len(masks), 500):
        m = resize_1(masks[i:i + 500])
        img_msks.append(m)
    img_msks = torch.cat(img_msks)
    conf_msks = []
    for i in range(0, len(masks), 500):
        m = resize_2(masks[i:i + 500])
        conf_msks.append(m)
    conf_msks = torch.cat(conf_msks)
    return (img_msks, conf_msks)
