"""Original stage objectives and augmentation expressed as a native Trainer."""
from __future__ import annotations
import numpy as np
import torch as t
import torch
from typing import Dict,Optional
from motius.registry import TRAINERS
from motius.motion.conditioning.motionbricks import extract_feature_from_motion_rep,sample_motion_segments_from_motion_clips,sample_keyframes
from .base import _MotionBricksStageTrainer
@TRAINERS.register_module()
class MotionBricksRootTrainer(_MotionBricksStageTrainer):
    STAGE = 'root'

    def training_step(self, batch, batch_idx, use_outside_training: bool=False, meta_info: Dict={}):
        """ @brief: the training step takes input a normalized motion representation.
            batch.keys() -> dict_keys(['motion', 'motion_len', 'motion_pad_mask', 'batch_size'])
        """
        assert self.training or use_outside_training, "It's possible to use training_step on evaluation set. But otherwise self.training should be true."
        batch_size, device = (batch['batch_size'], batch['motion'].device)
        raw_global_motions, motion_lengths, _ = (batch.pop('motion'), batch.pop('motion_len'), batch.pop('motion_pad_mask'))
        if self.network.ACCEPT_TEXT_EMB_INPUT:
            _, _, _ = (batch.pop('text'), batch.pop('text_len'), batch.pop('text_pad_mask'))
        augmented_batch_size = int(batch_size * self._args['batchsize_mul_factor'])
        num_token_position = np.random.choice(np.arange(self._args['min_tokens'], self._args['max_tokens'] + 1))
        num_token_position_off_range = np.random.choice(np.arange(self._args['min_off_target_tokens_to_sample'], self._args['max_off_target_tokens_to_sample'] + 1))
        sample_off_range_target_token = np.random.rand() < self._args['prob_off_range_target_token']
        num_token_position = num_token_position_off_range if sample_off_range_target_token else num_token_position
        batch['num_tokens'] = t.full([augmented_batch_size, 1], num_token_position).to(device)
        num_frames = num_token_position * self.network.get_num_frames_per_token()
        valid_samples_id = motion_lengths >= num_frames + 1
        num_invalid_samples = batch_size - valid_samples_id.sum()
        if num_invalid_samples > batch_size // 4 * 3:
            return None
        sample_info = {}
        global_motions = sample_motion_segments_from_motion_clips(raw_global_motions, motion_lengths, num_frames, self.args['batchsize_mul_factor'], info=sample_info, motion_rep=self.global_motion_rep)
        actual_batch_size = int(batch_size * self.args['batchsize_mul_factor'])
        batch['text_embeddings'] = None if not self.network.ACCEPT_TEXT_EMB_INPUT else batch.pop('text_feat')[sample_info['chosen_ids']].view([augmented_batch_size, -1])
        first_frame_heading_angle = t.rand(actual_batch_size).to(device) * np.pi * 2.0 if not self.motion_rep.compute_kwargs['removing_heading'] else 0.0
        global_motions = self.global_motion_rep.change_first_heading(global_motions, first_frame_heading_angle, is_normalized=True, to_normalize=True)
        local_motions = self.motion_rep.dual_rep.global_to_local(global_motions, is_normalized=True, to_normalize=True, lengths=t.full([actual_batch_size], global_motions.shape[1]).to(device))
        local_motions, global_motions = (local_motions[:, :num_frames, :], global_motions[:, :num_frames, :])
        batch['local_motions'], batch['global_motions'] = (local_motions, global_motions)
        batch['local_root_values'], batch['has_local_root_values'] = self._sample_the_conditions(extract_feature_from_motion_rep(batch['local_motions'], self.local_motion_rep, self._args['local_root_feature']), num_frames)
        batch['global_root_values'], batch['has_global_root_values'] = self._sample_the_conditions(extract_feature_from_motion_rep(batch['global_motions'], self.global_motion_rep, self._args['global_root_feature']), num_frames)
        batch['local_poses'], batch['has_local_poses'] = self._sample_the_conditions(extract_feature_from_motion_rep(batch['local_motions'], self.local_motion_rep, self._args['local_pose_feature']), num_frames)
        batch['text_embeddings'], batch['has_text_embeddings'] = self._sample_text_embedding(batch['text_embeddings'], batch['has_global_root_values'], batch['has_local_root_values'], batch['has_local_poses'])
        batch['num_tokens'] = batch['num_tokens'].clip(max=self.network.OUT_OF_REACH_NUM_TOKENS)
        batch['groundtruth_num_tokens'] = batch['num_tokens'].clone()
        provide_num_tokens = t.rand([augmented_batch_size, 1]) < self._args['prob_provide_num_tokens']
        batch['num_tokens'] = t.where(provide_num_tokens.to(device), batch['num_tokens'], t.full_like(batch['num_tokens'], self.network.MASKED_NUM_TOKENS))
        model_outputs = self._network_forward(batch['global_root_values'], batch['has_global_root_values'], batch['local_root_values'], batch['has_local_root_values'], batch['local_poses'], batch['has_local_poses'], batch['num_tokens'], text_embeddings=batch['text_embeddings'], has_text_embeddings=batch['has_text_embeddings'], groundtruth_num_tokens=batch['groundtruth_num_tokens'])
        losses = self.loss(batch, model_outputs)
        if use_outside_training:
            meta_info['losses'] = losses
            return None
        for key, val in losses.items():
            self.log(f'loss/train_{key}', val, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch['batch_size'])
        return losses['loss']

    def loss(self, batch: Dict, model_output_batch: Dict):
        """ @brief: calculate the root prediction loss and num-token classification loss.
        """
        losses = {}
        batch_size, num_valid_frames, device = (batch['global_motions'].shape[0], batch['global_motions'].shape[1], batch['global_motions'].device)
        pred_local_root_motions = self.motion_rep.dual_rep.global_to_local(model_output_batch['pred_global_root_values'][:, :num_valid_frames], is_normalized=True, to_normalize=True, lengths=t.full([batch_size], num_valid_frames).to(device))
        groundtruth_global_root_values = extract_feature_from_motion_rep(batch['global_motions'][:, :num_valid_frames], self.global_motion_rep, self._args['global_root_feature'])
        global_root_recons_loss = t.nn.SmoothL1Loss()(model_output_batch['pred_global_root_values'][:, :num_valid_frames, :], groundtruth_global_root_values)
        groundtruth_global_root_values = extract_feature_from_motion_rep(batch['local_motions'][:, :num_valid_frames - 1], self.local_motion_rep, self._args['local_root_feature'])
        local_root_recons_loss = t.nn.SmoothL1Loss()(pred_local_root_motions[:, :num_valid_frames - 1, :], groundtruth_global_root_values)
        num_tokens_idx = batch['groundtruth_num_tokens'].reshape([-1]) - self._args['min_tokens']
        num_token_loss = t.nn.functional.cross_entropy(model_output_batch['num_token_logits'], num_tokens_idx)
        pred_rank = model_output_batch['num_token_logits'].argsort(dim=1, descending=True)
        for i in [1, 3, 5]:
            losses[f'top_{i}_accuracy'] = (pred_rank[:, :i] == num_tokens_idx[:, None]).any(dim=-1).float().mean()
        total_loss = global_root_recons_loss * self._args['global_root_loss_coeff'] + local_root_recons_loss * self._args['local_root_loss_coeff'] + num_token_loss * self._args['num_token_loss_coeff']
        losses['num_token_loss'] = num_token_loss
        losses['global_root_recons_loss'] = global_root_recons_loss
        losses['local_root_recons_loss'] = local_root_recons_loss
        losses['loss'] = total_loss
        return losses

    def _construct_keyframe_prob(self, max_num_keyframes: int=None, no_keyframe_prob: float=0.0):
        """ @brief: generate the prob to sample each keyframe during training.
        """
        scheduled_max_num_keyframes = max_num_keyframes * (self.get_global_step() / self.args['keyframe_num_warmup_steps'])
        scheduled_max_num_keyframes = int(max(1, min(scheduled_max_num_keyframes, max_num_keyframes)))
        prob_num_keyframes = [1.0 if i > 0 and i <= scheduled_max_num_keyframes else 0.0 for i in range(max_num_keyframes + 1)]
        prob_no_keyframe = no_keyframe_prob
        prob_num_keyframes[0] = sum(prob_num_keyframes) / (1 - prob_no_keyframe) * prob_no_keyframe
        prob_num_keyframes = np.array(prob_num_keyframes)
        prob_num_keyframes /= prob_num_keyframes.sum()
        return prob_num_keyframes

    def _sample_the_conditions(self, feature: t.Tensor, num_frames: int):
        """ @brief: sample sparse conditions (start/end keyframes) during training.
        """
        NUM_START_FRAMES = NUM_END_FRAMES = self.network.get_num_frames_per_token()
        NUM_MIDDLE_FRAMES = num_frames - NUM_START_FRAMES - NUM_END_FRAMES
        assert NUM_MIDDLE_FRAMES > 0, 'The number of frames should be larger than the number of start and end frames.'
        feature_component = {'start': feature[:, :NUM_START_FRAMES], 'end': feature[:, -NUM_END_FRAMES:]}
        has_cond, cond = ({}, {})
        for component in ['start', 'end']:
            prob_num_keyframes = self._construct_keyframe_prob(self._args[f'max_num_{component}_keyframes'], no_keyframe_prob=self._args[f'no_{component}_keyframe_prob'])
            has_cond[component], cond[component] = sample_keyframes(feature_component[component], self._args[f'max_num_{component}_keyframes'], prob_num_keyframes)
        has_cond = t.cat([has_cond['start'], has_cond['end']], dim=1)
        cond = t.cat([cond['start'], cond['end']], dim=1)
        return (cond, has_cond)

    def _sample_text_embedding(self, text_embedding: t.Tensor=None, has_global_root_values: t.Tensor=None, has_local_root_values: t.Tensor=None, has_local_poses: t.Tensor=None):
        if text_embedding is not None:
            assert self.network.ACCEPT_TEXT_EMB_INPUT, 'The model does not accept text embedding as input.'
            has_text_emb = t.rand([text_embedding.shape[0], 1], device=text_embedding.device) < self._args.get('prob_provide_text_emb', 0.2)
            NUM_END_FRAMES = self.network.get_num_frames_per_token()
            has_end_target_info = t.logical_or(t.logical_or(has_global_root_values[:, -NUM_END_FRAMES:], has_local_root_values[:, -NUM_END_FRAMES:]), has_local_poses[:, -NUM_END_FRAMES:])
            has_text_emb = t.logical_or(has_text_emb, has_end_target_info.sum(dim=1, keepdims=True) < 1)
            text_embedding = t.where(has_text_emb, text_embedding, t.zeros_like(text_embedding))
        else:
            text_embedding = has_text_emb = None
        return (text_embedding, has_text_emb)
