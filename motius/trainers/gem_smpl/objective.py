import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from motius.motion.representation.gem_smpl.codec import EnDecoder
from motius.motion.representation.gem_smpl.body_params import get_pred_body_params_incam, set_intermediate_pred_body_params_global, set_pred_body_params_global, set_pred_body_params_incam
from motius.motion.representation.gem_smpl.camera import compute_transl_full_cam, get_a_pred_cam, project_to_bi01
from motius.motion.representation.gem_smpl.motion import get_tgtcoord_rootparam, rollout_local_transl_vel
from motius.motion.representation.gem_smpl.smoothing import gaussian_smooth
from motius.motion.representation.gem_smpl.rotation import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix

def compute_extra_incam_loss(inputs, outputs, ppl, mode):
    model_output = outputs['model_output']
    endecoder = ppl.endecoder
    weights = ppl.weights
    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs['mask']['valid'].clone()
    mask[inputs['mask']['2d_only']] = False
    mask_reproj = ~inputs['mask']['spv_incam_only']
    mask_reproj_17 = torch.zeros_like(inputs['mask']['valid']).bool()
    mask_reproj_17[inputs['mask']['2d_only']] = True
    pred_params_incam = get_pred_body_params_incam(outputs)
    smpl_model = ppl.endecoder.smplx_model
    smpl_fwd_params_incam = {k: v for k, v in pred_params_incam.items() if k in ('body_pose', 'global_orient', 'transl', 'betas')}
    pred_c_j3d = endecoder.fk_v2(**outputs['pred_body_params_incam'])
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]
    gt_c_j3d = endecoder.fk_v2(**inputs['smpl_params_c'])
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]
    jts_weights = torch.ones_like(gt_c_j3d[..., :1])
    if weights.cr_j3d > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction='none')
        cr_j3d_loss = (cr_j3d_loss * mask[..., None, None] * jts_weights).mean()
        extra_loss += cr_j3d_loss * weights.cr_j3d
        extra_loss_dict['cr_j3d_loss'] = cr_j3d_loss
    if weights.transl_c > 0.0:
        pred_cam = model_output['pred_cam']
        gt_transl = inputs['smpl_params_c']['transl']
        gt_pred_cam = get_a_pred_cam(gt_transl, inputs['bbx_xys'], inputs['K_fullimg'])
        gt_pred_cam[gt_pred_cam.isinf()] = -1
        gt_j3d_z_min = inputs['gt_j3d_coco17'][..., 2].min(dim=-1)[0]
        valid_mask = ((gt_j3d_z_min > 0.3) * (gt_pred_cam[..., 0] > 0.3) * (gt_pred_cam[..., 0] < 5.0) * (gt_pred_cam[..., 1] > -3.0) * (gt_pred_cam[..., 1] < 3.0) * (gt_pred_cam[..., 2] > -3.0) * (gt_pred_cam[..., 2] < 3.0) * (inputs['bbx_xys'][..., 2] > 0))[..., None]
        transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction='none')
        transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()
        extra_loss_dict['transl_c_loss'] = transl_c_loss
        extra_loss += transl_c_loss * weights.transl_c
    if weights.j2d > 0.0:
        reproj_z_thr = 0.3
        pred_c_j3d = pred_c_j3d.clone()
        pred_c_j3d_z0_mask = pred_c_j3d[..., 2].abs() <= reproj_z_thr
        pred_c_j3d[pred_c_j3d_z0_mask] = reproj_z_thr
        gt_c_j3d_z0_mask = gt_c_j3d[..., 2].abs() <= reproj_z_thr
        gt_c_j3d[gt_c_j3d_z0_mask] = reproj_z_thr
        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs['bbx_xys'], inputs['K_fullimg'])
        gt_j2d_01 = project_to_bi01(gt_c_j3d, inputs['bbx_xys'], inputs['K_fullimg'])
        valid_mask = ((gt_c_j3d[..., 2] > reproj_z_thr) * (pred_c_j3d[..., 2] > reproj_z_thr) * ~pred_j2d_01[..., 0].isnan() * ~pred_j2d_01[..., 0].isinf() * ~pred_j2d_01[..., 1].isnan() * ~pred_j2d_01[..., 1].isinf() * (gt_j2d_01[..., 0] > 0.0) * (gt_j2d_01[..., 0] < 1.0) * (gt_j2d_01[..., 1] > 0.0) * (gt_j2d_01[..., 1] < 1.0) * ~gt_j2d_01[..., 0].isnan() * ~gt_j2d_01[..., 0].isinf() * ~gt_j2d_01[..., 1].isnan() * ~gt_j2d_01[..., 1].isinf())[..., None]
        valid_mask[~mask_reproj] = False
        pred_j2d_01[pred_j2d_01.isnan()] = 0.0
        pred_j2d_01[pred_j2d_01.isinf()] = 0.0
        gt_j2d_01[gt_j2d_01.isnan()] = 0.0
        gt_j2d_01[gt_j2d_01.isinf()] = 0.0
        j2d_loss = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction='none')
        j2d_loss = (j2d_loss * mask[..., None, None] * valid_mask * jts_weights).mean()
        extra_loss += j2d_loss * weights.j2d
        extra_loss_dict['j2d_loss'] = j2d_loss
    if weights.get('cr_verts', 0) > 0:
        pred_c_verts437, pred_c_j17 = smpl_model(**smpl_fwd_params_incam)
        root_ = pred_c_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_
        gt_cr_verts437 = inputs['gt_cr_verts437']
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction='none')
        cr_vert_loss = (cr_vert_loss * mask[:, :, None, None]).mean()
        extra_loss += cr_vert_loss * weights.cr_verts
        extra_loss_dict['cr_vert_loss'] = cr_vert_loss
    if weights.get('verts2d', 0) > 0:
        gt_c_verts437 = inputs['gt_c_verts437']
        reproj_z_thr = 0.3
        pred_c_verts437 = pred_c_verts437.clone()
        pred_c_verts437_z0_mask = pred_c_verts437[..., 2].abs() <= reproj_z_thr
        pred_c_verts437[pred_c_verts437_z0_mask] = reproj_z_thr
        gt_c_verts437_z0_mask = gt_c_verts437[..., 2].abs() <= reproj_z_thr
        gt_c_verts437[gt_c_verts437_z0_mask] = reproj_z_thr
        pred_verts2d_01 = project_to_bi01(pred_c_verts437, inputs['bbx_xys'], inputs['K_fullimg'])
        gt_verts2d_01 = project_to_bi01(gt_c_verts437, inputs['bbx_xys'], inputs['K_fullimg'])
        valid_mask = ((gt_c_verts437[..., 2] > reproj_z_thr) * (pred_c_verts437[..., 2] > reproj_z_thr) * ~pred_verts2d_01[..., 0].isnan() * ~pred_verts2d_01[..., 0].isinf() * ~pred_verts2d_01[..., 1].isnan() * ~pred_verts2d_01[..., 1].isinf() * (gt_verts2d_01[..., 0] > 0.0) * (gt_verts2d_01[..., 0] < 1.0) * (gt_verts2d_01[..., 1] > 0.0) * (gt_verts2d_01[..., 1] < 1.0) * ~gt_verts2d_01[..., 0].isnan() * ~gt_verts2d_01[..., 0].isinf() * ~gt_verts2d_01[..., 1].isnan() * ~gt_verts2d_01[..., 1].isinf())[..., None]
        pred_verts2d_01[pred_verts2d_01.isnan()] = 0.0
        pred_verts2d_01[pred_verts2d_01.isinf()] = 0.0
        gt_verts2d_01[gt_verts2d_01.isnan()] = 0.0
        gt_verts2d_01[gt_verts2d_01.isinf()] = 0.0
        valid_mask[~mask_reproj] = False
        verts2d_loss = F.mse_loss(pred_verts2d_01, gt_verts2d_01, reduction='none')
        verts2d_loss = (verts2d_loss * mask[..., None, None] * valid_mask).mean()
        extra_loss += verts2d_loss * weights.verts2d
        extra_loss_dict['verts2d_loss'] = verts2d_loss
    if weights.get('shape_loss', 0.0) > 0:
        if 'betas' in pred_params_incam and 'smpl_params_c' in inputs:
            shape_loss = F.mse_loss(pred_params_incam['betas'], inputs['smpl_params_c']['betas'], reduction='none')
            shape_loss = (shape_loss * mask[..., None]).mean()
            extra_loss += shape_loss * weights.shape_loss
            extra_loss_dict['shape_loss'] = shape_loss
    return (extra_loss, extra_loss_dict)

def compute_extra_global_loss(inputs, outputs, ppl, mode):
    decode_dict = outputs['decode_dict']
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args
    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs['mask']['valid'].clone()
    mask[inputs['mask']['spv_incam_only']] = False
    mask[inputs['mask']['2d_only']] = False
    mask_contact = mask.clone()
    if 'static_gt_mask' in inputs:
        static_gt_mask = inputs['static_gt_mask']
        while static_gt_mask.ndim < mask_contact.ndim:
            static_gt_mask = static_gt_mask.unsqueeze(-1)
        if static_gt_mask.shape != mask_contact.shape:
            static_gt_mask = static_gt_mask.expand_as(mask_contact)
        mask_contact = mask_contact & static_gt_mask
    else:
        mask_contact[inputs['mask']['invalid_contact']] = False
    if weights.transl_w > 0:
        gt_transl_w = inputs['smpl_params_w']['transl']
        gt_global_orient_w = inputs['smpl_params_w']['global_orient']
        local_transl_vel = decode_dict['local_transl_vel']
        pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]])
        trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction='none')
        trans_w_loss = (trans_w_loss * mask[..., None]).mean()
        extra_loss += trans_w_loss * weights.transl_w
        extra_loss_dict['transl_w_loss'] = trans_w_loss
    if weights.static_conf_bce > 0:
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        static_gt = inputs['static_gt'] if 'static_gt' in inputs else endecoder.get_static_gt(inputs, vel_thr).float()
        pred_static_conf_logits = outputs['model_output']['static_conf_logits']
        static_conf_loss = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction='none')
        static_conf_loss = (static_conf_loss * mask_contact[..., None]).mean()
        extra_loss += static_conf_loss * weights.static_conf_bce
        extra_loss_dict['static_conf_loss'] = static_conf_loss
    return (extra_loss, extra_loss_dict)
