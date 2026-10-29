from motius.motion.conditioning.gem_processor import GEMProcessor
import os
import time
import numpy as np
import torch
import torch.nn as nn
from motius.models.gem_x.conditioning import nvskel77to33_idx
from motius.motion.representation.gem_x import stats_compose
from motius.motion.representation.gem_x.body_params import get_pred_body_params_global, get_pred_body_params_incam
from motius.motion.representation.gem_x.camera import compute_bbox_info_bedlam, perspective_projection, safely_render_x3d_K
from motius.motion.representation.gem_x.geometry import get_bbx_xys, normalize_kp2d
from motius.models.gem_x.positions import length_to_mask
from motius.motion.conditioning.gem_x_augmentation import get_invisible_legs_mask, get_visible_mask, get_wham_aug_kp3d, randomly_modify_hands_legs
from motius.motion.body_models.gem_x import SomaLayer

class GemXProcessor(GEMProcessor):
    def normalize_attr(self, x, key):
        """Normalize input tensor using stored statistics"""
        mean = self.normalizer_stats[key]['mean'].to(x)
        std = self.normalizer_stats[key]['std'].to(x)
        return (x - mean) / std

    def encode_text(self, raw_text, has_text=None):
        device = self.device
        if self.tokenizer is None or self.text_encoder is None:
            batch_size = len(raw_text)
            max_text_len = getattr(self, 'max_text_len', 16)
            text_dim = 1024
            denoiser = self.bundle._unwrap(self.bundle.denoiser)
            if denoiser is not None and hasattr(denoiser, 'encoded_text_dim'):
                text_dim = denoiser.encoded_text_dim
            encoded_text = torch.zeros((batch_size, max_text_len, text_dim), device=device, dtype=torch.float32)
            if has_text is not None:
                no_text = ~has_text.to(device)
                encoded_text[no_text] = 0
            return encoded_text
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                max_text_len = self.max_text_len
                encoded = self.tokenizer.batch_encode_plus(raw_text, return_tensors='pt', padding='max_length', max_length=max_text_len, truncation=True)
                input_ids = encoded.input_ids.to(device)
                attn_mask = encoded.attention_mask.to(device)
                with torch.no_grad():
                    output = self.bundle.encode_text_tokens(input_ids=input_ids, attention_mask=attn_mask)
                    encoded_text = output.detach()
                encoded_text = encoded_text[:, :max_text_len]
                attn_mask = attn_mask[:, :max_text_len]
                encoded_text *= attn_mask.unsqueeze(-1)
        if has_text is not None:
            no_text = ~has_text
            encoded_text[no_text] = 0
        return encoded_text

    def generate_mask(self, mask_cfg, orig_mask, length):
        _cfg = mask_cfg
        mask = torch.ones_like(orig_mask)
        njoints = orig_mask.shape[-1]
        drop_prob = _cfg.get('drop_prob', 0.0)
        if drop_prob <= 0:
            return mask
        max_num_drops = _cfg.get('max_num_drops', 1)
        min_drop_nframes = _cfg.get('min_drop_nframes', 1)
        max_drop_nframes = _cfg.get('max_drop_nframes', 30)
        joint_drop_prob = _cfg.get('joint_drop_prob', 0.0)
        for i in range(orig_mask.shape[0]):
            mlen = length[i].item()
            if np.random.rand() < drop_prob:
                num_drops = np.random.randint(1, max_num_drops + 1)
                for _ in range(num_drops):
                    drop_len = np.random.randint(min_drop_nframes, min(max_drop_nframes, mlen) + 1)
                    drop_start = np.random.randint(0, max(mlen - drop_len, 1))
                    if joint_drop_prob > 0:
                        drop_joints = np.random.rand(njoints) < joint_drop_prob
                        mask[i, drop_start:drop_start + drop_len, drop_joints] = False
                    else:
                        mask[i, drop_start:drop_start + drop_len] = False
        if joint_drop_prob > 0 and njoints == 17:
            COCO17_TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]
            for child in range(njoints):
                parent = COCO17_TREE[child]
                if parent == -1:
                    continue
                if isinstance(parent, list):
                    mask[..., child] *= mask[..., parent[0]] * mask[..., parent[1]]
                else:
                    mask[..., child] *= mask[..., parent]
        elif joint_drop_prob > 0:
            assert njoints == 77, f'njoints: {njoints} is not supported for body mask'
            SKELETON_TREE = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75]
            for child in range(njoints):
                parent = SKELETON_TREE[child]
                if parent == -1:
                    continue
                if isinstance(parent, list):
                    mask[..., child] *= mask[..., parent[0]] * mask[..., parent[1]]
                else:
                    mask[..., child] *= mask[..., parent]
        return mask

    def prepare_batch(self, batch, mode):
        target_x = self.endecoder.encode(batch)
        batch['sample_indices_dict'] = self.endecoder.obs_indices_dict
        if mode == 'diffusion':
            target_x[batch['mask']['2d_only']] = batch['regression_outputs']['model_output']['pred_x_start'][batch['mask']['2d_only']]
        else:
            target_x[batch['mask']['2d_only']] = 0
        valid_mask = batch['mask']['valid']
        target_x_mask = torch.ones_like(target_x).bool()
        target_x_mask[batch['mask']['2d_only']] = False
        global_obs_idx = self.endecoder.get_obs_indices('global_orient')
        global_sidx = int(global_obs_idx[0]) if isinstance(global_obs_idx, tuple) else int(global_obs_idx[0].item())
        target_x_mask[batch['mask']['spv_incam_only'], :, global_sidx:] = False
        target_x_mask = target_x_mask & valid_mask[:, :, None]
        batch['target_x'] = target_x
        batch['target_x_mask'] = target_x_mask
        batch['device'] = batch['target_x'].device
        batch['B'], batch['L'] = B, L = batch['target_x'].shape[:2]
        if 'text_embed' in batch:
            batch['encoded_text'] = batch['text_embed'].to(self.device)
        elif self.use_text_encoder:
            batch['encoded_text'] = self.encode_text(batch['caption'], batch['has_text'])
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                output = self.body_model(**batch['soma_params_c'])
                gt_verts, gt_j3d = (output['vertices'], output['joints'])
                root_ = gt_j3d[:, :, [0], :]
                batch['gt_j3d'] = gt_j3d
                batch['gt_cr_77'] = gt_j3d - root_
                batch['gt_c_verts437'] = gt_verts
                batch['gt_cr_verts437'] = gt_verts - root_
            i_x2d = safely_render_x3d_K(gt_verts, batch['K_fullimg'], thr=0.3)
        det_kp2d = batch['kp2d']
        assert det_kp2d.ndim == 4 and det_kp2d.shape[-1] == 3, f'det_kp2d.shape: {det_kp2d.shape}'
        det_kp2d_conf = det_kp2d[..., 2]
        batch['det_kp2d_conf'] = det_kp2d_conf
        det_kp2d = det_kp2d[..., :2]
        det_kp2d_mask = det_kp2d_conf > 0.5
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
        bbx_xys_detected = get_bbx_xys(det_kp2d, det_kp2d_mask, do_augment=True)
        bbx_xys[batch['mask']['2d_only']] = bbx_xys_detected[batch['mask']['2d_only']]
        if False:
            batch['bbx_xys'] = bbx_xys
        else:
            mask_bbx_xys = batch['mask']['bbx_xys']
            batch['bbx_xys'][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys].to(batch['bbx_xys'])
        with torch.cuda.amp.autocast(enabled=False):
            _gt_kp2d = perspective_projection(gt_j3d, batch['K_fullimg'])
            noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2], num_J=gt_j3d.shape[-2], device=gt_j3d.device)
            obs_i_j2d = perspective_projection(noisy_j3d, batch['K_fullimg'])
            noisy_det_j2d = det_kp2d.clone()
            if gt_j3d.shape[-2] in (17, 77):
                aug = get_wham_aug_kp3d(noisy_det_j2d.shape[:2], num_J=gt_j3d.shape[-2], device=noisy_det_j2d.device)[..., :2]
                f = torch.tensor([1024.0, 1024.0]).to(aug) / 4.0
                aug *= f * self.model_cfg.kp2d_noise_scale
                noisy_det_j2d = noisy_det_j2d + aug
        prob = 0.5
        mask_real_vitpose = (torch.rand(batch['B']).to(obs_i_j2d) < prob) * batch['mask']['vitpose']
        mask_real_vitpose = mask_real_vitpose | batch['mask']['2d_only']
        assert batch['mask']['2d_only'].sum() == 0, batch['mask']['2d_only'].sum()
        obs_i_j2d[mask_real_vitpose] = noisy_det_j2d[mask_real_vitpose]
        obs_i_j2d = randomly_modify_hands_legs(obs_i_j2d, num_J=obs_i_j2d.shape[-2])
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2], num_J=gt_j3d.shape[-2], device=gt_j3d.device)
        j2d_visible_mask = j2d_visible_mask & batch['mask']['has_2d_mask'][:, :, None]
        j2d_visible_mask[mask_real_vitpose] *= det_kp2d_mask[mask_real_vitpose]
        close_mask = (noisy_j3d[..., 2] < 0.3) & (~mask_real_vitpose)[:, None, None]
        j2d_visible_mask[close_mask] = False
        legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:2], num_J=j2d_visible_mask.shape[-1], device=gt_j3d.device)
        j2d_visible_mask[legs_invisible_mask] = False
        if 'mask_cfg' in self.model_cfg:
            mask = self.generate_mask(self.model_cfg.mask_cfg, j2d_visible_mask, batch['length'])
            j2d_visible_mask = j2d_visible_mask & mask
        if 'body_mask_cfg' in self.model_cfg:
            mask = self.generate_mask(self.model_cfg.body_mask_cfg, j2d_visible_mask, batch['length'])
            j2d_visible_mask = j2d_visible_mask & mask
        occluded_img_mask = j2d_visible_mask.sum(dim=-1) <= 3
        f_cliffcam = compute_bbox_info_bedlam(batch['bbx_xys'], batch['K_fullimg'])
        f_cliffcam[occluded_img_mask] = 0
        batch['f_cliffcam'] = f_cliffcam
        condition_mask = dict()
        condition_mask['has_img_mask'] = batch['mask']['has_img_mask'] & ~occluded_img_mask
        condition_mask['has_2d_mask'] = batch['mask']['has_2d_mask'] & ~occluded_img_mask
        condition_mask['has_cam_mask'] = batch['mask']['has_cam_mask'].clone()
        condition_mask['has_audio_mask'] = batch['mask']['has_audio_mask'].clone()
        condition_mask['has_music_mask'] = batch['mask']['has_music_mask'].clone()
        batch['condition_mask'] = condition_mask
        obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)
        obs = normalize_kp2d(obs_kp2d, batch['bbx_xys'])
        obs[~j2d_visible_mask] = 0
        j2d_visible_mask[~batch['mask']['valid']] = False
        batch['obs'] = obs
        condition_mask['j2d_visible_mask'] = j2d_visible_mask
        batch['obs'][~batch['mask']['valid']] = 0
        if 'static_gt' not in batch:
            batch['static_gt'] = self.endecoder.get_static_gt(batch, self.args.static_conf.vel_thr)
        batch['static_gt_mask'] = ~batch['mask']['invalid_contact']
        f_cam_angvel = batch['cam_angvel']
        if self.model_cfg.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        batch['f_cam_angvel'] = f_cam_angvel
        for k in self.normalizer_stats:
            if k in batch:
                batch[k] = self.normalize_attr(batch[k], k)
