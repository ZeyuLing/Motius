import torch
def compute_transl_full_cam_metric3d(pred_trans_in_pixel_space, bbox_center_scale, cam_int):
    """
    pred_trans_in_pixel_space: (B, L, 3)
    bbox_center_scale: (B, L, 3)
    cam_int: (B, L, 3, 3)
    pz represents depth in a canonical camera with focal length 1000px.
    px and py are offsets of the human from the bbox center in pixel space.

    returns: pred_trans_in_metric_space (B, L, 3)
    """
    fx = cam_int[:, :, 0, 0]
    fy = cam_int[:, :, 1, 1]
    cx = cam_int[:, :, 0, 2]
    cy = cam_int[:, :, 1, 2]
    img_w, img_h = (cx * 2, cy * 2)
    bbox_cx = bbox_center_scale[:, :, 0]
    bbox_cy = bbox_center_scale[:, :, 1]
    bbox_scale = bbox_center_scale[:, :, 2]
    px = pred_trans_in_pixel_space[:, :, 0]
    py = pred_trans_in_pixel_space[:, :, 1]
    pz = pred_trans_in_pixel_space[:, :, 2]
    pz = 1 / (pz + 1e-06)
    px = px * img_w
    py = py * img_h
    u = bbox_cx + px
    v = bbox_cy + py
    Z = fx / 1000.0 * pz
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pred_trans_in_metric_space = torch.stack([X, Y, Z], dim=-1)
    return pred_trans_in_metric_space

def compute_transl_full_cam_prompthmr(transl, bbx_xys, K_fullimg):
    focal = K_fullimg[:, :, 0, 0]
    px, py, pz = transl.unbind(-1)
    pz = 1 / (pz + 1e-06)
    tx = px * pz
    ty = py * pz
    tz = pz * focal / 1000.0
    t_full = torch.stack([tx, ty, tz], dim=-1)
    return t_full

def inv_compute_transl_full_cam_metric3d(pred_trans_in_metric_space, bbox_center_scale, cam_int):
    '''
    pred_trans_in_metric_space: (B, L, 3) - Predicted translations [X, Y, Z] in metric space.
    bbox_center_scale: (B, L, 3) - Bounding box center coordinates and scale [bbox_cx, bbox_cy, bbox_scale].
    cam_int: (B, L, 3, 3) - Camera intrinsic matrices.

    Returns:
    pred_trans_in_pixel_space: (B, L, 3) - Predicted offsets [px, py, pz_in] in pixel space.
    '''
    import torch

    # Extract camera intrinsics
    fx = cam_int[:, :, 0, 0]  # Focal length in x (B, L)
    fy = cam_int[:, :, 1, 1]  # Focal length in y (B, L)
    cx = cam_int[:, :, 0, 2]  # Principal point x-coordinate (B, L)
    cy = cam_int[:, :, 1, 2]  # Principal point y-coordinate (B, L)

    img_w = cx * 2  # Image width (B, L)
    img_h = cy * 2  # Image height (B, L)

    # Extract bbox center and scale
    bbox_cx = bbox_center_scale[:, :, 0]  # Bounding box center x-coordinate (B, L)
    bbox_cy = bbox_center_scale[:, :, 1]  # Bounding box center y-coordinate (B, L)
    bbox_scale = bbox_center_scale[:, :, 2]  # Bounding box scale (B, L)

    # Extract predicted translations in metric space
    X = pred_trans_in_metric_space[:, :, 0]  # Coordinate X in metric space (B, L)
    Y = pred_trans_in_metric_space[:, :, 1]  # Coordinate Y in metric space (B, L)
    Z = pred_trans_in_metric_space[:, :, 2]  # Depth Z in metric space (B, L)

    # Avoid division by zero
    Z_safe = Z + 1e-6  # (B, L)

    # Adjust depth from actual camera to canonical camera
    f_c = 1000.0  # Canonical focal length
    s = fx / f_c  # Scaling factor (B, L)

    # Compute inverse depth in canonical camera space
    pz_in = (s / Z_safe) - 1e-6  # Predicted inverse depth (B, L)

    # Compute image coordinates u and v
    u = (X * fx) / Z_safe + cx  # Image coordinate x (B, L)
    v = (Y * fy) / Z_safe + cy  # Image coordinate y (B, L)

    # Compute offsets from bbox center
    px = u - bbox_cx  # Offset in x (B, L)
    py = v - bbox_cy  # Offset in y (B, L)

    px = px / img_w
    py = py / img_h

    # Stack the offsets to form the translation vector in pixel space
    pred_trans_in_pixel_space = torch.stack([px, py, pz_in], dim=-1)  # (B, L, 3)

    return pred_trans_in_pixel_space


def inv_compute_transl_full_cam_prompthmr(transl, bbx_xys, K_fullimg):
    focal = K_fullimg[:,:,0,0]
    tx, ty, tz = transl.unbind(-1)

    # Invert depth calculation
    pz = 1 / (tz * 1000. / focal + 1e-6)

    # Invert x,y calculations
    px = tx / tz
    py = ty / tz

    transl_out = torch.stack([px, py, pz], dim=-1)
    return transl_out
