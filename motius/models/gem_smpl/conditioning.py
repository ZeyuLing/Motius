# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
import torch
from torch import nn
from timm.models.vision_transformer import Mlp
from .layers import BasicBlock, zero_module

class ConditionEncoder(nn.Module):

    def __init__(self, args, model_config, denoiser_config, obs_num_joints):
        super().__init__()
        self.args = args
        self.model_cfg = model_config
        model_cfg = model_config
        self.no_exist_keys = ['obs', 'observed_motion_3d', 'multi_text_embed']
        self.remove_fingers = args.get('remove_fingers', False)
        self.f_condition_dim = {'obs': (obs_num_joints, 3), 'f_cliffcam': (3,), 'f_cam_angvel': (6,), 'f_cam_t_vel': (3,), 'f_imgseq': (1024,), 'encoded_music': (self.args.encoded_music_dim,), 'encoded_audio': (128,), 'observed_motion_3d': (151,)}
        self.not_add_features = ['obs', 'f_cliffcam', 'f_cam_angvel', 'f_cam_t_vel', 'f_imgseq', 'observed_motion_3d', 'multi_text_embed', 'encoded_music', 'encoded_audio']
        dropout = denoiser_config.get('dropout', 0.1)
        latent_dim = denoiser_config.get('latent_dim', 512)
        self.latent_dim = latent_dim
        if 'obs' in self.args.in_attr:
            self.learned_pos_linear = nn.Linear(2, 32)
            self.learned_pos_params = nn.Parameter(torch.randn(obs_num_joints, 32), requires_grad=True)
            self.embed_noisyobs = Mlp(obs_num_joints * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout)
        if 'f_cliffcam' in self.args.in_attr:
            self.cliffcam_embedder = nn.Sequential(nn.Linear(self.f_condition_dim['f_cliffcam'][0], latent_dim), nn.SiLU(), nn.Dropout(dropout), zero_module(nn.Linear(latent_dim, latent_dim)))
        if 'f_imgseq' in self.args.in_attr:
            self.imgseq_embedder = nn.Sequential(nn.LayerNorm(self.f_condition_dim['f_imgseq'][0]), zero_module(nn.Linear(self.f_condition_dim['f_imgseq'][0], latent_dim)))
        if 'f_cam_angvel' in self.args.in_attr:
            self.cam_angvel_embedder = nn.Sequential(nn.Linear(self.f_condition_dim['f_cam_angvel'][0], latent_dim), nn.SiLU(), nn.Dropout(dropout), zero_module(nn.Linear(latent_dim, latent_dim)))
        if 'f_cam_t_vel' in self.args.in_attr:
            self.cam_t_vel_embedder = nn.Sequential(nn.Linear(self.f_condition_dim['f_cam_t_vel'][0], latent_dim), nn.SiLU(), nn.Dropout(dropout), zero_module(nn.Linear(latent_dim, latent_dim)))
        if 'encoded_music' in self.args.in_attr:
            self.music_embedder = Mlp(self.f_condition_dim['encoded_music'][0], hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout)
            self.music_mask_prob = model_cfg.music_mask_prob
        if 'encoded_audio' in self.args.in_attr:
            self.audio_encoder = torch.nn.Sequential(BasicBlock(1, 32, 15, 5), BasicBlock(32, 32, 15, 6), BasicBlock(32, 32, 15, 1), BasicBlock(32, 64, 15, 5), BasicBlock(64, 64, 15, 1), BasicBlock(64, 128, 15, 4))
            self.audio_embedder = nn.Sequential(nn.LayerNorm(self.f_condition_dim['encoded_audio'][0]), zero_module(nn.Linear(self.f_condition_dim['encoded_audio'][0], latent_dim)))
            self.audio_mask_prob = model_cfg.audio_mask_prob
        if 'multi_text_embed' in self.args.in_attr:
            multi_text_module_cfg = model_cfg.get('multi_text_module_cfg', {})
            text_embed_dim = multi_text_module_cfg.get('text_embed_dim', 1024)
            self.multi_text_embedder = nn.Linear(text_embed_dim, latent_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=multi_text_module_cfg.get('nhead', 8), dim_feedforward=multi_text_module_cfg.get('dim_feedforward', 2048), dropout=dropout, batch_first=True)
            self.multi_text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=multi_text_module_cfg.get('num_layers', 3))
        self.condition_source = {'image': ['f_imgseq'], '2d': ['obs', 'f_cliffcam'], 'camera': ['f_cam_angvel', 'f_cam_tvel'], 'audio': ['encoded_audio'], 'music': ['encoded_music']}
        if self.model_cfg.use_cond_exists_as_input:
            if self.model_cfg.cond_merge_strategy == 'add':
                self.cond_exists_embedder = nn.ModuleDict()
                for k in self.args.in_attr:
                    if k not in self.no_exist_keys:
                        self.cond_exists_embedder[k] = nn.Sequential(nn.Linear(latent_dim + 1, latent_dim), nn.SiLU(), zero_module(nn.Linear(latent_dim, latent_dim)))
            elif self.model_cfg.cond_merge_strategy == 'concat':
                raise NotImplementedError('Concat is not implemented')

    def forward(self, batch, f_condition_exists, j2d_visible_mask, first_k_frames=None):
        B, L = (batch['B'], batch['L'])
        f_cond_dict = {}
        f_uncond_dict = {}
        f_uncond_exists = {k: f_condition_exists[k].clone() for k in f_condition_exists}
        f_empty_dict = {}
        f_empty_exists = {k: torch.zeros_like(f_condition_exists[k]) for k in f_condition_exists}
        length = batch['length']
        end_fr = first_k_frames if first_k_frames is not None else None
        if first_k_frames is not None:
            length = length.clamp(max=first_k_frames)
        for k in self.args.in_attr:
            if k == 'obs':
                obs = batch['obs'][:, :end_fr]
                B, L, J, C = obs.shape
                assert J in (17, 33) and C == 3
                obs = obs.clone()
                obs = obs * j2d_visible_mask[:, :, :, None]
                visible_mask = obs[..., [2]] > 0.5
                obs[~visible_mask[..., 0]] = 0
                f_obs = self.learned_pos_linear(obs[..., :2])
                f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
                f_obs_empty = self.learned_pos_params.repeat(B, L, 1, 1)
                f_obs = self.embed_noisyobs(f_obs.view(B, L, -1))
                f_obs_empty = self.embed_noisyobs(f_obs_empty.view(B, L, -1))
                f_cond_dict['obs'] = f_obs
                f_uncond_dict['obs'] = f_obs
                f_empty_dict['obs'] = f_obs_empty
            elif k == 'f_cliffcam':
                f_cliffcam = batch['f_cliffcam'][:, :end_fr]
                f_cliffcam = self.cliffcam_embedder(f_cliffcam)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict['f_cliffcam'] = f_cliffcam * mask.float()
                f_uncond_dict['f_cliffcam'] = f_cliffcam * mask.float()
                f_empty_dict['f_cliffcam'] = torch.zeros_like(f_cliffcam)
            elif k == 'f_cam_angvel':
                f_cam_angvel = batch['f_cam_angvel'][:, :end_fr]
                f_cam_angvel = self.cam_angvel_embedder(f_cam_angvel)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict['f_cam_angvel'] = f_cam_angvel * mask.float()
                f_uncond_dict['f_cam_angvel'] = f_cam_angvel * mask.float()
                f_empty_dict['f_cam_angvel'] = torch.zeros_like(f_cam_angvel)
            elif k == 'f_cam_t_vel':
                f_cam_t_vel = batch['f_cam_t_vel'][:, :end_fr]
                f_cam_t_vel = self.cam_t_vel_embedder(f_cam_t_vel)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict['f_cam_t_vel'] = f_cam_t_vel * mask.float()
                f_uncond_dict['f_cam_t_vel'] = f_cam_t_vel * mask.float()
                f_empty_dict['f_cam_t_vel'] = torch.zeros_like(f_cam_t_vel)
            elif k == 'f_imgseq':
                f_imgseq = batch['f_imgseq'][:, :end_fr]
                f_imgseq = self.imgseq_embedder(f_imgseq)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict['f_imgseq'] = f_imgseq * mask.float()
                f_uncond_dict['f_imgseq'] = f_imgseq * mask.float()
                f_empty_dict['f_imgseq'] = torch.zeros_like(f_imgseq)
            elif k == 'encoded_music':
                if 'music_embed' in batch:
                    f_encoded_music = batch['music_embed'][:, :end_fr]
                    f_encoded_music = self.music_embedder(f_encoded_music)
                    mask = f_condition_exists[k][:, :, None]
                    f_cond_dict['encoded_music'] = f_encoded_music * mask.float()
                else:
                    f_cond_dict['encoded_music'] = torch.zeros(B, L, self.latent_dim).to(batch['device'])
                f_uncond_dict['encoded_music'] = torch.zeros(B, L, self.latent_dim).to(batch['device'])
                f_uncond_exists['encoded_music'] = torch.zeros_like(f_condition_exists['encoded_music'])
                f_empty_dict['encoded_music'] = torch.zeros(B, L, self.latent_dim).to(batch['device'])
            elif k == 'encoded_audio':
                if 'audio_array' in batch:
                    encoded_audio = self.audio_encoder(batch['audio_array'].to(batch['device']).unsqueeze(1)).transpose(1, 2).contiguous()[:, :end_fr]
                    mask = f_condition_exists[k][:, :, None]
                    encoded_audio = self.audio_embedder(encoded_audio)
                    f_cond_dict['encoded_audio'] = encoded_audio * mask.float()
                else:
                    f_cond_dict['encoded_audio'] = torch.zeros(B, L, self.latent_dim).to(batch['device'])
                f_uncond_dict['encoded_audio'] = torch.zeros(B, L, self.latent_dim).to(batch['device'])
                f_uncond_exists['encoded_audio'] = torch.zeros_like(f_condition_exists['encoded_audio'])
                f_empty_dict['encoded_audio'] = torch.zeros(B, L, self.latent_dim).to(batch['device'])
            elif k == 'observed_motion_3d':
                motion_mask_3d = batch.get('motion_mask_3d', torch.zeros_like(batch['observed_motion_3d']))[:, :end_fr]
                f_observed_motion_3d = torch.cat([batch['observed_motion_3d'][:, :end_fr], motion_mask_3d], dim=-1)
                f_observed_motion_3d = self.observed_motion_3d_embedder(f_observed_motion_3d)
                f_cond_dict['observed_motion_3d'] = f_observed_motion_3d
                f_uncond_dict['observed_motion_3d'] = torch.zeros_like(f_observed_motion_3d)
                f_empty_dict['observed_motion_3d'] = torch.zeros_like(f_observed_motion_3d)
            else:
                raise AssertionError(f'Unknown condition key: {k}')
            if k not in self.not_add_features:
                f_cond_dict[k] = self.add_feature_embedders[k](batch[k][:, :end_fr])
            if self.model_cfg.use_cond_exists_as_input:
                if k not in self.no_exist_keys:
                    if k == 'obs':
                        exist_mask = f_condition_exists[k][:, :end_fr]
                        exist_mask = exist_mask.sum(dim=-1, keepdim=True) > 0
                        uncond_exist_mask = f_uncond_exists[k][:, :end_fr]
                        uncond_exist_mask = uncond_exist_mask.sum(dim=-1, keepdim=True) > 0
                        empty_exist_mask = f_empty_exists[k][:, :end_fr]
                        empty_exist_mask = empty_exist_mask.sum(dim=-1, keepdim=True) > 0
                    else:
                        exist_mask = f_condition_exists[k][:, :end_fr, None]
                        uncond_exist_mask = f_uncond_exists[k][:, :end_fr, None]
                        empty_exist_mask = f_empty_exists[k][:, :end_fr, None]
                    f_cond_dict[k] = torch.cat([f_cond_dict[k], exist_mask.float()], dim=-1)
                    f_cond_dict[k] = self.cond_exists_embedder[k](f_cond_dict[k])
                    f_uncond_dict[k] = torch.cat([f_uncond_dict[k], uncond_exist_mask.float()], dim=-1)
                    f_uncond_dict[k] = self.cond_exists_embedder[k](f_uncond_dict[k])
                    f_empty_dict[k] = torch.cat([f_empty_dict[k], empty_exist_mask.float()], dim=-1)
                    f_empty_dict[k] = self.cond_exists_embedder[k](f_empty_dict[k])
        f_cond = sum(f_cond_dict.values())
        f_uncond = sum(f_uncond_dict.values())
        f_empty = sum(f_empty_dict.values())
        batch['f_cond'] = f_cond
        batch['f_uncond'] = f_uncond
        batch['f_empty'] = f_empty
        return (f_cond, f_uncond, f_empty)
