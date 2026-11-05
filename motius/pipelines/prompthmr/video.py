"""Prepared temporal prediction, camera decoding and world-motion assembly."""
import json
from pathlib import Path
import torch
from torch.amp import autocast
from torch.nn import functional as F
from motius.motion.representation.gvhmr.camera import compute_bbox_info_bedlam, compute_transl_full_cam
from motius.motion.representation.prompthmr.video_codec import PromptHMRVideoCodec
from motius.motion.representation.prompthmr.video_motion import rollout_local_transl_vel, get_tgtcoord_rootparam
from motius.motion.representation.gvhmr.tensor_ops import gaussian_smooth
from motius.motion.representation.prompthmr.camera import compute_transl_full_cam_prompthmr, compute_transl_full_cam_metric3d
from motius.motion.representation.prompthmr.video_rotation import rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle

class PromptHMRVideoPrediction:

    def __init__(self, bundle, body_model_path, *, camt_type=None, normalize_cam_angvel=True):
        self.bundle = bundle
        self.endecoder = PromptHMRVideoCodec(bundle.device)
        stats = json.loads((Path(__file__).resolve().parents[2] / 'configuration/prompthmr/stats.json').read_text())
        self.endecoder.mean.copy_(torch.tensor(stats['motion']['mean'], device=bundle.device))
        self.endecoder.std.copy_(torch.tensor(stats['motion']['std'], device=bundle.device))
        self.cam_angvel_mean = torch.tensor(stats['camera_angular_velocity']['mean'], device=bundle.device)
        self.cam_angvel_std = torch.tensor(stats['camera_angular_velocity']['std'], device=bundle.device)
        from types import SimpleNamespace
        self.args = SimpleNamespace(normalize_cam_angvel=normalize_cam_angvel)
        self.camt_type = camt_type or bundle.video_config.get('camt_type', 'cliff')

    def forward(self, inputs, static_cam=False):
        outputs = dict()
        length = inputs['length']
        f_cam_angvel = inputs['cam_angvel']
        if self.args.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        cliff_cam = compute_bbox_info_bedlam(inputs['bbx_xys'], inputs['K_fullimg'])
        f_condition = {'obs': inputs['obs'], 'f_cliffcam': cliff_cam, 'f_prhmrcam': inputs['f_imgseq'][:, :, 1], 'f_cam_angvel': f_cam_angvel, 'f_imgseq': inputs['f_imgseq'][:, :, 0]}
        model_output = self.bundle.predict_video(length=length, cam_int=inputs['K_fullimg'], **f_condition)
        decode_dict = self.endecoder.decode(model_output['pred_x'])
        outputs.update({'model_output': model_output, 'decode_dict': decode_dict})
        if self.camt_type == 'prompthmr':
            transl_incam = compute_transl_full_cam_prompthmr(model_output['pred_cam'], inputs['bbx_xys'], inputs['K_fullimg'])
        elif self.camt_type == 'cliff':
            transl_incam = compute_transl_full_cam(model_output['pred_cam'], inputs['bbx_xys'], inputs['K_fullimg'])
        elif self.camt_type == 'metric3d':
            transl_incam = compute_transl_full_cam_metric3d(model_output['pred_cam'], inputs['bbx_xys'], inputs['K_fullimg'])
        else:
            raise NotImplementedError('Not implemented')
        outputs['pred_smpl_params_incam'] = {'body_pose': decode_dict['body_pose'], 'betas': decode_dict['betas'], 'global_orient': decode_dict['global_orient'], 'transl': transl_incam}
        pred_smpl_params_global = get_smpl_params_w_Rt_v2(global_orient_gv=decode_dict['global_orient_gv'], local_transl_vel=decode_dict['local_transl_vel'], global_orient_c=decode_dict['global_orient'], cam_angvel=inputs['cam_angvel'])
        outputs['pred_smpl_params_global'] = {'body_pose': decode_dict['body_pose'], 'betas': decode_dict['betas'], **pred_smpl_params_global}
        outputs['static_conf_logits'] = model_output['static_conf_logits']
        return outputs

@autocast('cuda', enabled=False)
def get_smpl_params_w_Rt_v2(global_orient_gv, local_transl_vel, global_orient_c, cam_angvel):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
    """

    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-05
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R
    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)
    R_t_to_tp1 = as_identity(R_t_to_tp1)
    R_gv = axis_angle_to_matrix(global_orient_gv)
    R_c = axis_angle_to_matrix(global_orient_c)
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]
    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)
    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0))
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle
    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)
    R_t_to_0 = as_identity(R_t_to_0)
    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf='any->ay')
    smpl_params_w_Rt = {'global_orient': global_orient, 'transl': transl}
    return smpl_params_w_Rt
