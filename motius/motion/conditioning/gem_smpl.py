# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
import torch

def create_condition_mask(self, batch, cond_mask_cfg, mode, train, first_k_frames=None):
    B, L = (batch['B'], batch['L'])
    device = batch['device']
    has_text = batch['has_text']
    condition_mask = batch['condition_mask']
    has_img_mask = condition_mask['has_img_mask'].clone()
    has_2d_mask = condition_mask['has_2d_mask'].clone()
    has_cam_mask = condition_mask['has_cam_mask'].clone()
    has_audio_mask = condition_mask['has_audio_mask'].clone()
    has_music_mask = condition_mask['has_music_mask'].clone()
    j2d_visible_mask = condition_mask['j2d_visible_mask'].clone()
    if train:
        regression_no_img_mask = cond_mask_cfg.get('regression_no_img_mask', False)
        mask_text_prob = cond_mask_cfg.get('mask_text_prob', {}).get(mode, 0.0)
        mask_img_prob = cond_mask_cfg.get('mask_img_prob', 0.0)
        mask_cam_prob = cond_mask_cfg.get('mask_cam_prob', 0.0)
        mask_f_imgseq_prob = cond_mask_cfg.get('mask_f_imgseq_prob', 0.0)
        if mask_text_prob > 0:
            mask_text = (torch.rand(batch['B']) < mask_text_prob).to(device)
            batch['text_mask'] = mask_text
        else:
            batch['text_mask'] = None
        if batch.get('text_mask', None) is not None:
            batch['has_text'][batch['text_mask']] = False
        if regression_no_img_mask and mode == 'regression':
            mask_img_prob = 0
            mask_f_imgseq_prob = 0
            has_2d_mask[~batch['mask']['2d_only']] = True
        if mask_img_prob > 0:
            mask_img = (has_text[:, None] | has_audio_mask | has_music_mask) & (torch.rand(batch['B']) < mask_img_prob).to(device)[:, None]
            has_img_mask = has_img_mask & ~mask_img
            has_2d_mask = has_2d_mask & ~mask_img
            j2d_visible_mask = j2d_visible_mask & ~mask_img[..., None]
        if mask_cam_prob > 0:
            mask_cam = (has_text[:, None] | has_music_mask | has_audio_mask) & (torch.rand(batch['B']) < mask_cam_prob).to(device)[:, None]
            has_cam_mask = has_cam_mask & ~mask_cam
        has_music_mask = has_music_mask & (torch.rand((B,), device=device) > self.music_mask_prob)[:, None]
        has_audio_mask = has_audio_mask & (torch.rand((B,), device=device) > self.audio_mask_prob)[:, None]
    j2d_visible_mask = j2d_visible_mask & has_2d_mask[:, :, None]
    has_2d_mask = j2d_visible_mask.sum(dim=-1) > 3
    f_condition_exists = dict()
    for k in self.condition_source['image']:
        f_condition_exists[k] = has_img_mask.clone()
    for k in self.condition_source['2d']:
        if k == 'obs':
            f_condition_exists[k] = j2d_visible_mask.clone()
        else:
            f_condition_exists[k] = has_2d_mask.clone()
    for k in self.condition_source['camera']:
        f_condition_exists[k] = has_cam_mask.clone()
    for k in self.condition_source['audio']:
        f_condition_exists[k] = has_audio_mask.clone()
    for k in self.condition_source['music']:
        f_condition_exists[k] = has_music_mask.clone()
    if train and mask_f_imgseq_prob > 0:
        mask_f_imgseq = (torch.rand(batch['B']) < mask_f_imgseq_prob).to(device)
        f_condition_exists['f_imgseq'] = f_condition_exists['f_imgseq'] & ~mask_f_imgseq
    skip_keys = self.args.get('skip_keys_for_null_condition', [])
    uncond_prob = self.args.get('uncond_prob', 0.1)
    if train and (not self.args.get('disable_random_null_condition', False)):
        for k in self.args.in_attr:
            if k in skip_keys:
                continue
            mask = torch.rand(f_condition_exists[k].shape[:2]) < uncond_prob
            f_condition_exists[k][mask] = False
    return (f_condition_exists, j2d_visible_mask)
