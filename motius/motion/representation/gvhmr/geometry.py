import logging
Log = logging.getLogger(__name__)
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d
import motius.motion.representation.gvhmr.matrix as matrix
from einops import einsum

def apply_T_on_points(points, T):
    """
    Args:
        points: (..., N, 3)
        T: (..., 4, 4)
    Returns: (..., N, 3)
    """
    points_T = torch.einsum('...ki,...ji->...jk', T[..., :3, :3], points) + T[..., None, :3, 3]
    return points_T

def cvt_to_bi01_p2d(p2d, bbx_lurb):
    """
    p2d: (..., (N), 2)
    bbx_lurb: (..., 4)
    """
    if len(p2d.shape) == len(bbx_lurb.shape) + 1:
        bbx_lurb = bbx_lurb[..., None, :]
    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]
    bi01_p2d = (p2d - bbx_lurb[..., :2]) / bbx_wh
    return bi01_p2d

def transform_mat(R, t):
    """
    Args:
        R: Bx3x3 array of a batch of rotation matrices
        t: Bx3x(1) array of a batch of translation vectors
    Returns:
        T: Bx4x4 Transformation matrix
    """
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)

def convert_bbx_xys_to_lurb(bbx_xys):
    """
    Args: bbx_xys (..., 3) -> bbx_lurb (..., 4)
    """
    size = bbx_xys[..., 2:]
    center = bbx_xys[..., :2]
    lurb = torch.cat([center - size / 2, center + size / 2], dim=-1)
    return lurb

def compute_T_ayfz2ay(joints, inverse=False):
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        if inverse == False:
            T_ayfz2ay: (B, 4, 4)
        else :
            T_ay2ayfz: (B, 4, 4)
    """
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0
    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 0.0001
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))
    x_dir = torch.zeros_like(t_ayfz2ay)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)
    R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)
    if inverse:
        R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
        t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, 'b i j , b i -> b j')
        return transform_mat(R_ay2ayfz, t_ay2ayfz)
    else:
        return transform_mat(R_ayfz2ay, t_ayfz2ay)

def compute_cam_angvel(R_w2c, padding_last=True):
    """
    R_w2c : (F, 3, 3)
    """
    cam_angvel = matrix_to_rotation_6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))
    assert padding_last
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)
    return cam_angvel.float()

def get_sequence_cammat(w_j3d, c_j3d, cam_rot):
    L, J, _ = w_j3d.shape
    root_in_w = w_j3d[:, 0]
    root_in_c = c_j3d[:, 0]
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]
    cam_mat = matrix.set_position(cam_mat, cam_pos)
    return cam_mat
