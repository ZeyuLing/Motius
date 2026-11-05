from copy import deepcopy
import torch
from motius.diffusion.gem_smpl import create_gaussian_diffusion
from motius.diffusion.gem_smpl.resample import create_named_schedule_sampler
from motius.models.gem_smpl.positions import length_to_mask
from .guidance import ClassifierFreeSampleModel

class GEMSampler:

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=True)
        self.test_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=False)
        gen_only_diffusion = deepcopy(self.model_cfg.diffusion)
        gen_only_diffusion.test_timestep_respacing = self.model_cfg.diffusion.get('gen_only_test_timestep_respacing', '50')
        print(f'Gen only test timestep respacing: {gen_only_diffusion.test_timestep_respacing}')
        self.test_gen_only_diffusion = create_gaussian_diffusion(gen_only_diffusion, training=False)
        self.schedule_sampler = create_named_schedule_sampler(self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion)
        return

    def forward_test(self, inputs, progress=False):
        assert not self.training, 'forward_test should only be called during inference'
        diffusion = self.test_gen_only_diffusion
        denoiser = self.denoiser
        length = inputs['length']
        B, L = (inputs['B'], inputs['L'])
        motion = inputs['motion']
        f_cond, f_uncond = (inputs['f_cond'], inputs['f_uncond'])
        f_empty = inputs['f_empty']
        vis_mask = length_to_mask(length, L)
        denoiser_kwargs = {'y': {'text': inputs.get('caption', [''] * B), 'f_cond': f_cond, 'f_uncond': f_uncond, 'f_empty': f_empty, 'mask': vis_mask, 'length': length}, 'inputs': inputs, 'sample_indices_dict': inputs['sample_indices_dict']}
        if 'encoded_text' in inputs:
            denoiser_kwargs['y']['encoded_text'] = inputs['encoded_text']
        if 'meta' in inputs and 'multi_text_data' in inputs['meta'][0]:
            denoiser_kwargs['y']['multi_text_data'] = inputs['meta'][0]['multi_text_data']
        if 'observed_motion_3d' in inputs:
            denoiser_kwargs['observed_motion_3d'] = inputs['observed_motion_3d']
            denoiser_kwargs['motion_mask_3d'] = inputs['motion_mask_3d']
            denoiser_kwargs['rm_text_flag'] = inputs.get('rm_text_flag', None)
        if self.args.get('use_cfg_sampler_for_gen', False):
            denoiser = ClassifierFreeSampleModel(denoiser, mask_localpose=self.mask_localpose, body_pose_indices=self.body_pose_indices)
            denoiser_kwargs['y']['scale'] = self.model_cfg.diffusion.guidance_param
        diff_sampler = self.model_cfg.diffusion.get('sampler', 'ddim')
        if diff_sampler == 'ddim':
            sample_fn = diffusion.ddim_sample_loop_with_aux
            kwargs = {'eta': self.model_cfg.diffusion.ddim_eta}
        else:
            raise NotImplementedError(f'Sampler {diff_sampler} not implemented')
        if self.args.get('force_zero_noise', False):
            noise = torch.zeros_like(motion)
        elif self.args.get('force_rand_noise', False):
            noise = torch.randn_like(motion)
        else:
            noise = torch.randn_like(motion)
        if self.args.get('return_mid', False):
            kwargs['return_mid'] = True
        if self.regression_only:
            x_t = torch.zeros_like(motion)
            t = (torch.ones(B) * (self.train_diffusion.original_num_steps - 1)).long().to(motion.device)
            denoise_out = self.denoiser(x_t, self.train_diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs)
            output = denoise_out.copy()
            if output.get('pred_x', None) is None and 'pred_x_start' in output:
                output['pred_x'] = output['pred_x_start']
        else:
            denoise_out = sample_fn(denoiser, motion.shape, clip_denoised=False, model_kwargs=denoiser_kwargs, skip_timesteps=0, init_image=None, progress=progress, dump_steps=None, noise=noise, const_noise=False, **kwargs)
            output = denoise_out.copy()
            if output.get('pred_x', None) is None and 'pred_x_start' in output:
                output['pred_x'] = output['pred_x_start']
        for x in self.args.out_attr:
            assert x in output, f'Output {x} not found in denoise_out'
        return output
