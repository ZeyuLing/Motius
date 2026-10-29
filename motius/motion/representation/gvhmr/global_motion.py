import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from motius.motion.representation.gvhmr.tensor_ops import gaussian_smooth
tsf_axisangle = {'ay->ay': [0, 0, 0], 'any->ay': [0, 0, torch.pi], 'az->ay': [-torch.pi / 2, 0, 0], 'ay->any': [0, 0, torch.pi]}

def get_tgtcoord_rootparam(global_orient, transl, gravity_vec=None, tgt_gravity_vec=None, tsf='ay->ay'):
    """Rotate around the origin center, to match the new gravity direction
    Args:
        global_orient: torch.tensor, (*, 3)
        transl: torch.tensor, (*, 3)
        gravity_vec: torch.tensor, (3,)
        tgt_gravity_vec: torch.tensor, (3,)
    Returns:
        tgt_global_orient: torch.tensor, (*, 3)
        tgt_transl: torch.tensor, (*, 3)
        R_g2tg: (3, 3)
    """
    device = global_orient.device
    if gravity_vec is None and tgt_gravity_vec is None:
        aa = torch.tensor(tsf_axisangle[tsf]).to(device)
        R_g2tg = axis_angle_to_matrix(aa)
    else:
        raise NotImplementedError
    global_orient_R = axis_angle_to_matrix(global_orient)
    tgt_global_orient = matrix_to_axis_angle(R_g2tg @ global_orient_R)
    tgt_transl = torch.einsum('...ij,...j->...i', R_g2tg, transl)
    return (tgt_global_orient, tgt_transl, R_g2tg)

def get_local_transl_vel(transl, global_orient):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        transl: (*, L, 3)
        global_orient: (*, L, 3)
    Returns:
        transl_vel: (*, L, 3)
    """
    assert len(transl.shape) == len(global_orient.shape)
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]
    transl_vel = torch.cat([transl_vel, transl_vel[..., [-1], :]], dim=-2)
    local_transl_vel = torch.einsum('...lij,...li->...lj', global_orient_R, transl_vel)
    return local_transl_vel

def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        local_transl_vel: (*, L, 3)
        global_orient: (*, L, 3)
        transl_0: (*, 1, 3), if not provided, the start point is 0
    Returns:
        transl: (*, L, 3)
    """
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = torch.einsum('...lij,...lj->...li', global_orient_R, local_transl_vel)
    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)
    transl = torch.cumsum(transl_, dim=-2)
    return transl

def get_static_joint_mask(w_j3d, vel_thr=0.25, smooth=False, repeat_last=False):
    """
    w_j3d: (*, L, J, 3)
    vel_thr: HuMoR uses 0.15m/s
    """
    joint_v_ = (w_j3d[..., 1:, :, :] - w_j3d[..., :-1, :, :]).pow(2).sum(-1).sqrt() / 0.033
    if smooth:
        joint_v_ = gaussian_smooth(joint_v_, 3, -2)
    static_joint_mask = joint_v_ < vel_thr
    if repeat_last:
        static_joint_mask = torch.cat([static_joint_mask, static_joint_mask[..., [-1], :]], dim=-2)
    return static_joint_mask
