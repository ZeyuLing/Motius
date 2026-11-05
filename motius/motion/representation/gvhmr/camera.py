import torch
import numpy as np
from motius.motion.representation.gvhmr.geometry import convert_bbx_xys_to_lurb, cvt_to_bi01_p2d

def estimate_focal_length(img_w, img_h):
    return (img_w ** 2 + img_h ** 2) ** 0.5

def estimate_K(img_w, img_h):
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K

def convert_K_to_K4(K):
    K4 = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float()
    return K4

def create_camera_sensor(width=None, height=None, f_fullframe=None):
    if width is None or height is None:
        if np.random.rand() < 0.5:
            width, height = (1200, 1600)
        else:
            width, height = (1600, 1200)
    if f_fullframe is None:
        f_fullframe_options = [24, 26, 28, 30, 35, 40, 50, 60, 70]
        f_fullframe = np.random.choice(f_fullframe_options)
    diag_fullframe = (24 ** 2 + 36 ** 2) ** 0.5
    diag_img = (width ** 2 + height ** 2) ** 0.5
    focal_length = diag_img / diag_fullframe * f_fullframe
    K_fullimg = torch.eye(3)
    K_fullimg[0, 0] = focal_length
    K_fullimg[1, 1] = focal_length
    K_fullimg[0, 2] = width / 2
    K_fullimg[1, 2] = height / 2
    return (width, height, K_fullimg)

def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """impl as in BEDLAM
    Args:
        bbx_xys: ((B), N, 3), in pixel space described by K_fullimg
        K_fullimg: ((B), (N), 3, 3)
    Returns:
        bbox_info: ((B), N, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    cx, cy, b = (bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2])
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl
    return bbox_info

def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = (pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2])
    focal_length = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-09)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-09)
    tz = 2 * focal_length / (sb + 1e-09)
    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t

def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """Inverse operation of compute_transl_full_cam"""
    assert transl.ndim == bbx_xys.ndim
    assert K_fullimg.ndim == bbx_xys.ndim + 1
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    gt_pred_cam = torch.stack([gt_s, gt_x, gt_y], dim=-1)
    return gt_pred_cam

def project_to_bi01(points, bbx_xys, K_fullimg):
    """
    points: (B, L, J, 3)
    bbx_xys: (B, L, 3)
    K_fullimg: (B, L, 3, 3)
    """
    p2d = perspective_projection(points, K_fullimg)
    bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys)
    p2d_bi01 = cvt_to_bi01_p2d(p2d, bbx_lurb)
    return p2d_bi01

def perspective_projection(points, K):
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.einsum('...ij,...kj->...ki', K, projected_points.float())
    return projected_points[..., :-1]

def normalize_kp2d(obs_kp2d, bbx_xys, clamp_scale_min=False):
    """
    Args:
        obs_kp2d: (B, L, J, 3) [x, y, c]
        bbx_xys: (B, L, 3)
    Returns:
        obs: (B, L, J, 3)  [x, y, c]
    """
    obs_xy = obs_kp2d[..., :2]
    obs_conf = obs_kp2d[..., 2]
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]
    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible_mask = (obs_xy[..., 0] < xy_min[..., None, 0]) + (obs_xy[..., 0] > xy_max[..., None, 0]) + (obs_xy[..., 1] < xy_min[..., None, 1]) + (obs_xy[..., 1] > xy_max[..., None, 1])
    obs_conf = obs_conf * ~invisible_mask
    if clamp_scale_min:
        scale = scale.clamp(min=1e-05)
    normalized_obs_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)

def get_bbx_xys(i_j2d, bbx_ratio=[192, 256], do_augment=False, base_enlarge=1.2):
    """Args: (B, L, J, 3) [x,y,c] -> Returns: (B, L, 3)"""
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    h = max_y - min_y
    w = max_x - min_x
    if True:
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio
    bbx_size = torch.max(h, w) * base_enlarge
    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8
        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)
    return torch.stack([center_x, center_y, bbx_size], dim=-1)

def get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2):
    """
    Args:
        bbx_xyxy: (N, 4) [x1, y1, x2, y2]
    Returns:
        bbx_xys: (N, 3) [center_x, center_y, size]
    """
    i_p2d = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)
    bbx_xys = get_bbx_xys(i_p2d[None], base_enlarge=base_enlarge)[0]
    return bbx_xys
