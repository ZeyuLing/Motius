"""Published metric-depth processing and scale fitting; no neural modules."""
import cv2
import torch
import numpy as np
from torchmin import minimize

def est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=None,
                     far_thresh=10):
    """ Depth-align by iterative + robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (0<pred_depth) * (pred_depth<far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)

    # Stage 2: Robust optimization
    robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (0<pred_depth) * (pred_depth<far_thresh)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    def f(x):
        loss = sm * x - pm
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale])
    result = minimize(f, x0,  method='bfgs')
    scale = result.x.detach().cpu().item()

    return scale


def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)
