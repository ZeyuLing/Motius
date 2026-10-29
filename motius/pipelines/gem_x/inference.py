import logging
Log=logging.getLogger(__name__)
import time
import torch
from motius.motion.representation.gem_x.geometry import normalize_kp2d
from motius.motion.representation.gem_x.camera import compute_bbox_info_bedlam

class GemXInference:
    @torch.no_grad()
    def predict(self, data, static_cam=False, postproc=True):
        now = time.time()
        test_mode = data['meta'][0].get('mode', 'default')
        if self.endecoder.obs_indices_dict is None:
            self.endecoder.build_obs_indices_dict()
        batch = {'length': data['length'][None].to(self.device), 'obs': normalize_kp2d(data['kp2d'], data['bbx_xys'])[None].to(self.device), 'bbx_xys': data['bbx_xys'][None].to(self.device), 'K_fullimg': data['K_fullimg'][None].to(self.device), 'cam_angvel': data['cam_angvel'][None].to(self.device), 'f_cam_angvel': data['cam_angvel'][None].to(self.device), 'cam_tvel': data['cam_tvel'][None].to(self.device), 'R_w2c': data['R_w2c'][None].to(self.device), 'f_imgseq': data['f_imgseq'][None].to(self.device), 'has_text': data['has_text'].to(self.device), 'B': 1, 'L': data['f_imgseq'].shape[0], 'mode': test_mode, 'target_x': torch.zeros(1, data['f_imgseq'].shape[0], self.endecoder.get_motion_dim()).to(self.device), 'sample_indices_dict': self.endecoder.obs_indices_dict}
        if 'music_embed' in data:
            batch['music_embed'] = data['music_embed'][None].to(self.device)
        if 'audio_array' in data:
            batch['audio_array'] = data['audio_array'][None].to(self.device)
        if 'fast_rollout' in data:
            batch['fast_rollout'] = data['fast_rollout']
        batch['device'] = batch['f_imgseq'].device
        if 'meta' in data:
            batch['meta'] = data['meta']
        else:
            batch['meta'] = None
        if 'text_embed' in batch:
            batch['encoded_text'] = batch['text_embed'].to(self.device)
        else:
            if 'caption' in data:
                batch['caption'] = [data['caption']]
            else:
                batch['caption'] = ['']
            batch['has_text'] = torch.tensor([True])
            batch['encoded_text'] = self.encode_text(batch['caption'], batch['has_text'])
        batch['f_cliffcam'] = compute_bbox_info_bedlam(batch['bbx_xys'], batch['K_fullimg']).to(self.device)
        condition_mask = dict()
        condition_mask['has_img_mask'] = data['mask']['has_img_mask'][None].to(self.device)
        condition_mask['has_2d_mask'] = data['mask']['has_2d_mask'][None].to(self.device)
        condition_mask['has_cam_mask'] = data['mask']['has_cam_mask'][None].to(self.device).clone()
        condition_mask['has_audio_mask'] = data['mask']['has_audio_mask'][None].to(self.device).clone()
        condition_mask['has_music_mask'] = data['mask']['has_music_mask'][None].to(self.device).clone()
        kp2d_conf = data['kp2d'][..., 2][None].to(self.device)
        condition_mask['j2d_visible_mask'] = kp2d_conf > 0.5
        batch['condition_mask'] = condition_mask
        if self.model_cfg.normalize_cam_angvel:
            batch['f_cam_angvel'] = (batch['f_cam_angvel'] - self.cam_angvel_mean) / self.cam_angvel_std
        for k in self.normalizer_stats:
            if k in batch:
                batch[k] = self.normalize_attr(batch[k], k)
        if 'multi_text_data' in batch['meta'][0]:
            if 'text_embed' not in batch['meta'][0]['multi_text_data']:
                multi_text_data = batch['meta'][0]['multi_text_data']
                num_text = len(multi_text_data['caption'])
                text_embed = self.encode_text(multi_text_data['caption'], torch.tensor([True] * num_text))
                batch['meta'][0]['multi_text_data']['text_embed'] = text_embed
        batch = self._encode_conditions(batch, cond_mask_cfg=None, mode=None, train=False)
        if self.args.infer_version == 3:
            postproc = False
        else:
            postproc = postproc
        Log.debug(f'Preproc taken: {time.time() - now}')
        now = time.time()
        outputs = self._sample_decode(batch, train=False, postproc=postproc, static_cam=static_cam, test_mode=test_mode)
        pred = {'body_params_global': {k: v[0] for k, v in outputs['pred_body_params_global'].items()}, 'body_params_incam': {k: v[0] for k, v in outputs['pred_body_params_incam'].items()}, 'K_fullimg': data['K_fullimg'], 'net_outputs': outputs}
        Log.debug(f'Demo taken: {time.time() - now}')
        return pred
