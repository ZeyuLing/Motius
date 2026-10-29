import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from motius.motion.representation.gem_x.codec import EnDecoder
from motius.pipelines.gem_x.postprocessing import refine_pose_with_contact_ik, refine_translation_with_contacts
from motius.motion.representation.gem_x.body_params import get_pred_body_params_incam, set_intermediate_pred_body_params_global, set_pred_body_params_global, set_pred_body_params_incam
from motius.motion.representation.gem_x.camera import compute_transl_full_cam, get_a_pred_cam, project_to_bi01
from motius.motion.representation.gem_x.motion import get_tgtcoord_rootparam, rollout_local_transl_vel
from motius.motion.representation.gem_x.smoothing import gaussian_smooth
from motius.motion.representation.gem_x.rotation import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix
from motius.motion.conditioning.gem_x_augmentation import NVSKEL_LFINGERS_IDX, NVSKEL_RFINGERS_IDX
from motius.motion.body_models.gem_x import SomaLayer

def decode_prediction(codec, model_output, inputs):
    outputs = dict()
    if codec.obs_indices_dict is None:
        codec.build_obs_indices_dict()
    pred_x = model_output.get('pred_x', None)
    if pred_x is None:
        pred_x = model_output.get('pred_x_start', None)
    if pred_x is None:
        pred_x = model_output.get('pred_xstart', None)
    if pred_x is None:
        raise ValueError('No prediction tensor found in model output (`pred_x`/`pred_x_start`).')
    decode_dict = codec.decode(pred_x)
    if 'scale_params' in decode_dict:
        scale_params = decode_dict['scale_params'].clone()
        global_scale = scale_params[..., 0].clone()
        global_scale = torch.clamp(global_scale, min=0.7, max=1.0)
        scale_params[..., 0] = global_scale
        decode_dict['scale_params'] = scale_params
    outputs.update({'model_output': model_output, 'decode_dict': decode_dict})
    if 'body_pose' in decode_dict and 'global_orient' in decode_dict:
        pred_body_params_incam = {'body_pose': decode_dict['body_pose'], 'global_orient': decode_dict['global_orient'], 'transl': compute_transl_full_cam(model_output['pred_cam'], inputs['bbx_xys'], inputs['K_fullimg'])}
        if 'betas' in decode_dict:
            pred_body_params_incam['betas'] = decode_dict['betas']
        if 'identity_coeffs' in decode_dict:
            pred_body_params_incam['identity_coeffs'] = decode_dict['identity_coeffs']
        if 'scale_params' in decode_dict:
            pred_body_params_incam['scale_params'] = decode_dict['scale_params']
        set_pred_body_params_incam(outputs, pred_body_params_incam)
    return outputs

def finish_prediction(codec, args, model_output, inputs, outputs, postproc=True, static_cam=False):
    decode_dict = outputs['decode_dict']
    if 'global_orient_gv' in decode_dict and 'local_transl_vel' in decode_dict and ('global_orient' in decode_dict):
        if args.get('infer_version', 2) == 2:
            pred_body_params_global = get_body_params_w_Rt_v2(global_orient_gv=decode_dict['global_orient_gv'], local_transl_vel=decode_dict['local_transl_vel'], global_orient_c=decode_dict['global_orient'], cam_angvel=inputs['cam_angvel'])
            pred_body_params_global = {'body_pose': decode_dict['body_pose'], **pred_body_params_global}
            if 'betas' in decode_dict:
                pred_body_params_global['betas'] = decode_dict['betas']
            if 'identity_coeffs' in decode_dict:
                pred_body_params_global['identity_coeffs'] = decode_dict['identity_coeffs']
            if 'scale_params' in decode_dict:
                pred_body_params_global['scale_params'] = decode_dict['scale_params']
            set_pred_body_params_global(outputs, pred_body_params_global)
            if 'intermediate_decode_dict' in outputs:
                intermediate_pred_body_params_global = []
                for int_decode_dict in outputs['intermediate_decode_dict']:
                    pred_body_params_global = get_body_params_w_Rt_v2(global_orient_gv=int_decode_dict['global_orient_gv'], local_transl_vel=int_decode_dict['local_transl_vel'], global_orient_c=int_decode_dict['global_orient'], cam_angvel=inputs['cam_angvel'])
                    pred_body_params_global = {'body_pose': int_decode_dict['body_pose'], **pred_body_params_global}
                    if 'betas' in int_decode_dict:
                        pred_body_params_global['betas'] = int_decode_dict['betas']
                    if 'identity_coeffs' in int_decode_dict:
                        pred_body_params_global['identity_coeffs'] = int_decode_dict['identity_coeffs']
                    if 'scale_params' in int_decode_dict:
                        pred_body_params_global['scale_params'] = int_decode_dict['scale_params']
                    intermediate_pred_body_params_global.append(pred_body_params_global)
                set_intermediate_pred_body_params_global(outputs, intermediate_pred_body_params_global)
    elif 'body_pose' in decode_dict:
        pred_body_params_global = {'body_pose': decode_dict['body_pose'], 'global_orient': decode_dict['global_orient_w'], 'transl': decode_dict['transl_w']}
        if 'betas' in decode_dict:
            pred_body_params_global['betas'] = decode_dict['betas']
        if 'identity_coeffs' in decode_dict:
            pred_body_params_global['identity_coeffs'] = decode_dict['identity_coeffs']
        if 'scale_params' in decode_dict:
            pred_body_params_global['scale_params'] = decode_dict['scale_params']
        set_pred_body_params_global(outputs, pred_body_params_global)
    if 'static_conf_logits' in model_output:
        outputs['static_conf_logits'] = model_output['static_conf_logits']
    if postproc and 'static_conf_logits' in outputs and ('pred_body_params_global' in outputs):
        outputs['pred_body_params_global']['transl'] = refine_translation_with_contacts(outputs, codec)
        body_pose = refine_pose_with_contact_ik(outputs, codec)
        decode_dict['body_pose'] = body_pose
        outputs['pred_body_params_global']['body_pose'] = body_pose
        if 'pred_body_params_incam' in outputs:
            outputs['pred_body_params_incam']['body_pose'] = body_pose
    return outputs

@autocast(enabled=False)
def get_body_params_w_Rt_v2(global_orient_gv, local_transl_vel, global_orient_c, cam_angvel):
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
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf='ay->ay')
    body_params_w_Rt = {'global_orient': global_orient, 'transl': transl}
    return body_params_w_Rt
