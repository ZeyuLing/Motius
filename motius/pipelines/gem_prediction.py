"""Shared native GEM condition/sampling/body-decoding composition."""
import importlib
from types import SimpleNamespace
import torch


class _Denoiser:
    def __init__(self,bundle):self.bundle=bundle
    def __call__(self,x,timesteps,y=None,**kwargs):return self.bundle.predict_denoise(x,timesteps,y=y,**kwargs)
    def parameters(self):return self.bundle.denoiser.parameters()
    def named_parameters(self):return self.bundle.denoiser.named_parameters()


class GEMPrediction:
    def initialize_prediction(self):
        variant=self.bundle.VARIANT
        self._selector=importlib.import_module('motius.motion.conditioning.'+variant).create_condition_mask
        self._decoding=importlib.import_module('motius.pipelines.'+variant+'.decoding')
        sampler=importlib.import_module('motius.pipelines.'+variant+'.sampling').GEMSampler()
        sampler.denoiser=_Denoiser(self.bundle)
        sampler.args=self.args
        sampler.model_cfg=SimpleNamespace(diffusion=self.bundle.config.diffusion)
        sampler.mask_localpose=self.bundle.config.network_options.get('mask_localpose',False)
        sampler.body_pose_indices=self.endecoder.obs_indices_dict['body_pose']
        sampler.regression_only=self.args.get('regression_only',False)
        sampler.training=False
        sampler.init_diffusion()
        self._sampler=sampler

    def _encode_conditions(self,batch,cond_mask_cfg,mode,train,first_k_frames=None):
        existence,visible=self._selector(self,batch,cond_mask_cfg,mode,train,first_k_frames)
        conditional,unconditional,empty=self.bundle.encode_conditions(batch,existence,visible,first_k_frames)
        batch.update(f_cond=conditional,f_uncond=unconditional,f_empty=empty)
        if batch.get('text_mask') is not None:batch['encoded_text']=batch['encoded_text']*(1-batch['text_mask'][:,None,None].float())
        length=batch['length'];length=length.clamp(max=first_k_frames) if first_k_frames is not None else length
        visible_frames=torch.arange(conditional.shape[1],device=length.device).expand(len(length),conditional.shape[1])<length.unsqueeze(1)
        end=first_k_frames
        batch['motion']=(batch['target_x']*visible_frames[...,None])[:,:end]
        return batch

    def _sample_decode(self,inputs,*,train=False,postproc=True,static_cam=False,test_mode=None,**kwargs):
        if train:raise ValueError('Use GemSmplTrainer/GemXTrainer for training graphs')
        model_output=self._sampler.forward_test(inputs)
        decoded=self._decoding.decode_prediction(self.endecoder,model_output,inputs)
        return self._decoding.finish_prediction(self.endecoder,self.args,model_output,inputs,decoded,postproc,static_cam)
