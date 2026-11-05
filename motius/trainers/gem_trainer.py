"""Shared native GEM regression/diffusion optimization graph."""
import importlib
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from motius.trainers.base_trainer import BaseTrainer


class GEMTrainer(BaseTrainer):
    def __init__(self,bundle,processor=None,stage='joint',**kwargs):
        super().__init__(bundle,**kwargs)
        if stage not in {'regression','diffusion','joint'}:raise ValueError(stage)
        if processor is None:
            cls=getattr(importlib.import_module('motius.motion.conditioning.'+bundle.VARIANT+'_processor'),'GemSmplProcessor' if bundle.VARIANT=='gem_smpl' else 'GemXProcessor')
            processor=cls(bundle)
        self.processor=processor
        self.stage=stage
        self.args=bundle.config.pipeline_args
        cfg=bundle.config
        diffusion=importlib.import_module('motius.diffusion.'+bundle.VARIANT)
        self.diffusion=diffusion.create_gaussian_diffusion(cfg.diffusion,training=True)
        samplers=importlib.import_module('motius.diffusion.'+bundle.VARIANT+'.resample')
        self.schedule_sampler=samplers.create_named_schedule_sampler(cfg.diffusion.schedule_sampler_type,self.diffusion)
        self.decoder=importlib.import_module('motius.pipelines.'+bundle.VARIANT+'.decoding')
        self.objective=importlib.import_module('motius.trainers.'+bundle.VARIANT+'.objective')
        self.selector=importlib.import_module('motius.motion.conditioning.'+bundle.VARIANT).create_condition_mask
        # Retain architectural frozen parameters, including positional encodings.
        bundle.conditioner.train();bundle.denoiser.train()
        bundle._trainable_modules=['conditioner','denoiser'];bundle._save_ckpt_modules=['conditioner','denoiser']
        bundle._frozen_modules=['text_encoder']

    def train_step(self,batch):
        if self.stage!='joint':return self._step(batch,self.stage)
        inputs=dict(batch)
        results={'loss':0}
        stages=self.bundle.config.pipeline_args.get('train_modes',['regression'])
        for stage in stages:
            # The original augments the same raw supervision anew for each mode.
            self.processor.prepare_batch(inputs,stage)
            outputs=self._step(inputs,stage,prepared=True,full_output=True)
            results['loss']=results['loss']+outputs['loss']
            results.update({'Loss_'+stage+'/'+key:value for key,value in outputs.items()
                            if isinstance(value,torch.Tensor) and ('_loss' in key or key=='loss')})
            if stage=='regression' and 'diffusion' in stages:inputs['regression_outputs']=outputs
        return results

    def _step(self,batch,stage,*,prepared=False,full_output=False):
        # A processor owns raw-data augmentation and representation preparation;
        # callers may also supply the exact prepared dataset fields directly.
        inputs=dict(batch)
        if not prepared and 'target_x' not in inputs:self.processor.prepare_batch(inputs,stage)
        # Source train_step snapshots top-level tensors before the next mode's
        # preparation mutates fields such as bounding boxes in place.
        inputs={k:(v.detach().clone() if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
        inputs={k:(v.to(self.bundle.device) if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
        existence,visible=self.selector(self.processor,inputs,self.bundle.config.model_config.condition_mask,stage,True)
        conditional,unconditional,empty=self.bundle.encode_conditions(inputs,existence,visible)
        inputs.update(f_cond=conditional,f_uncond=unconditional,f_empty=empty)
        if inputs.get('text_mask') is not None:inputs['encoded_text']=inputs['encoded_text']*(1-inputs['text_mask'][:,None,None].float())
        length=inputs['length'];frames=conditional.shape[1]
        mask=torch.arange(frames,device=length.device).expand(len(length),frames)<length.unsqueeze(1)
        motion=inputs['target_x']*mask[...,None];inputs['motion']=motion
        y=dict(text=inputs.get('caption',['']*len(length)),f_cond=conditional,mask=mask,length=length)
        if 'encoded_text' in inputs:y['encoded_text']=inputs['encoded_text']
        extra=dict(inputs=inputs,sample_indices_dict=inputs['sample_indices_dict'])
        if 'observed_motion_3d' in inputs:
            extra.update(observed_motion_3d=inputs['observed_motion_3d'],motion_mask_3d=inputs['motion_mask_3d'],rm_text_flag=inputs['rm_text_flag'])
        if stage=='regression':
            timestep=(torch.ones(len(length))*(self.diffusion.original_num_steps-1)).long().to(motion.device)
            time_weight=torch.ones(len(length)).to(motion.device);target=motion
            kind=self.args.get('regression_input_type','zero')
            if kind=='zero':noisy=torch.zeros_like(motion)
            elif kind=='normal':noisy=torch.randn_like(motion)
            else:raise ValueError(kind)
        else:
            timestep,time_weight=self.schedule_sampler.sample(len(length),motion.device)
            if 'regression_outputs' not in inputs:raise ValueError('Diffusion training requires detached regression_outputs for 2D-only targets')
            regression=inputs['regression_outputs']['model_output']['pred_x_start'].detach()
            target=motion.clone();target[inputs['mask']['2d_only']]=regression[inputs['mask']['2d_only']]
            noisy=self.diffusion.q_sample(target.clone(),timestep,noise=torch.randn_like(target))
            if self.bundle.config.network_options.get('mask_localpose',False):
                indices=self.processor.endecoder.obs_indices_dict['body_pose']
                selected=torch.rand((len(length),),device=noisy.device)<self.bundle.config.network_options.get('mask_localpose_prob',0.)
                noisy[:,:,indices]=target[:,:,indices]*selected[:,None,None].float()+noisy[:,:,indices]*(1-selected[:,None,None].float())
                y['f_cond']=conditional*(1-selected[:,None,None].float())+empty*selected[:,None,None].float()
        # This independent wrapped call follows the noise draw, preserving the
        # source's conditioner/dropout/noise/denoiser RNG ordering.
        prediction=self.bundle.predict_denoise(noisy,self.diffusion._scale_timesteps(timestep),return_aux=False,y=y,**extra)
        prediction.update(target_x_start=target,t_weights=time_weight)
        outputs=self.decoder.decode_prediction(self.processor.endecoder,prediction,inputs)
        total=0
        if self.args.weights.get('simple',1.)>0:
            dimension=151 if self.bundle.VARIANT=='gem_smpl' else self.processor.endecoder.get_motion_dim()
            error=F.mse_loss(prediction['pred_x'][...,:dimension],inputs['target_x'][...,:dimension],reduction='none')
            simple=(error*inputs['target_x_mask'][...,:dimension]).mean()
            total+=simple*self.args.weights.get('simple',1.);outputs['simple_loss']=simple
        owner=SimpleNamespace(endecoder=self.processor.endecoder,args=self.args,weights=self.args.weights)
        if self.bundle.VARIANT=='gem_x':owner._get_soma_body_model=self.processor.endecoder.get_soma_body_model
        if self.objective.get_pred_body_params_incam(outputs) is not None:
            for function in [self.objective.compute_extra_incam_loss,self.objective.compute_extra_global_loss]:
                loss,terms=function(inputs,outputs,owner,stage)
                total+=loss;outputs.update(terms)
        outputs['loss']=total
        if full_output:return outputs
        return {k:v for k,v in outputs.items() if isinstance(v,torch.Tensor)}

    @torch.no_grad()
    def val_step(self,batch):
        return {'val_'+key:value.detach() for key,value in self.train_step(batch).items()}
