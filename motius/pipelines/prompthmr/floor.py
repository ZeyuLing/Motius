"""Original floor-height fitting without visualization dependencies."""
import torch
import numpy as np

def fit_floor_height(points, method='ransac', axis='y'):
    """
    Find height offset.
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    axis = {'x': 0, 'y': 1, 'z': 2}[axis]
    if method == 'lowest':
        offset = points[..., axis].min()
    elif method == 'average':
        z, _ = points[..., axis].min(dim=1)
        offset = z.mean()
    elif method == 'ransac':
        zs, _ = points[..., axis].min(dim=1)
        zs = np.sort(zs)
        alpha = 1.0
        min_z = np.min(zs)
        max_z = np.max(zs)
        zs = zs[zs <= min_z + (max_z - min_z) * alpha]
        inlier_thresh = 0.05
        best_inliers = 0
        best_z = 0.0
        for i in range(10000):
            z = np.random.choice(zs)
            inliers = np.abs(zs - z) < inlier_thresh
            inliers = np.sum(inliers)
            if inliers > best_inliers:
                best_z = z
                best_inliers = inliers
        offset = np.median(zs[np.abs(zs - best_z) < inlier_thresh])
    height_offset = torch.zeros([3])
    height_offset[axis] = offset.item()
    return height_offset
