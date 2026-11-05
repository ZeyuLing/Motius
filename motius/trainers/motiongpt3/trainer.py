"""Native MotionGPT3 language, diffusion-head and VAE training graphs."""
import numpy as np
import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.conditioning.motiongpt3 import MotionGPT3Processor
from motius.diffusion.motiongpt3 import create_diffusion
from .objective import prepare_language_batch,language_loss


@TRAINERS.register_module()
class MotionGPT3Trainer(BaseTrainer):
    def __init__(self,bundle,stage='lm_pretrain',language_weight=1.,diffusion_weight=1.,
                 feature_weight=1.,velocity_weight=.5,kl_weight=1e-5,**kwargs):
        super().__init__(bundle,**kwargs)
        if stage not in ['vae','lm_pretrain','lm_instruct','lm_finetune']:raise ValueError(stage)
        self.stage=stage;self.language_weight=language_weight;self.diffusion_weight=diffusion_weight
        self.feature_weight=feature_weight;self.velocity_weight=velocity_weight;self.kl_weight=kl_weight
        self._processor=None;self._gradient_hooks=[]
        bundle._trainable_modules=['vae'] if stage=='vae' else ['lm','diffusion_head']
        bundle._save_ckpt_modules=list(bundle._trainable_modules)
        bundle._frozen_modules=['lm','diffusion_head'] if stage=='vae' else ['vae']
        bundle.vae.requires_grad_(stage=='vae')
        if stage=='vae':bundle.lm.requires_grad_(False);bundle.diffusion_head.requires_grad_(False)
        else:
            # Keep the published modality-specific text/motion freezing policy.
            model=bundle.unwrap_network(bundle.lm).language_model
            if 'finetune' in stage:model.requires_grad_(True)
            else:
                start=model.config.text_vocab_size-2
                def new_rows_only(gradient):
                    mask=torch.zeros_like(gradient);mask[start:]=1
                    return gradient*mask
                for module in [model.pre_processors[0],model.post_processors[0]]:
                    self._gradient_hooks.append(module.weight.register_hook(new_rows_only))
        self.train_diffusion=create_diffusion(timestep_respacing='',**bundle.diffusion_options)

    def processor(self):
        if self._processor is None:
            from transformers import AutoTokenizer
            self._processor=MotionGPT3Processor(AutoTokenizer.from_pretrained(self.bundle.tokenizer_path,legacy=True,local_files_only=True),self.bundle.options,self.bundle.device)
        self._processor.device=self.bundle.device
        return self._processor

    def train_step(self,batch):
        motion=batch['motion'].to(self.bundle.device);lengths=batch['length']
        if self.stage=='vae':
            reconstruction,latent,distribution=self.bundle.vae(motion,lengths=lengths,reencode=True)
            reference=torch.distributions.Normal(torch.zeros_like(distribution.loc),torch.ones_like(distribution.scale))
            feature=F.smooth_l1_loss(reconstruction,motion)
            kl=torch.distributions.kl_divergence(distribution,reference).mean()
            velocity=F.smooth_l1_loss(reconstruction[...,4:67],motion[...,4:67])
            # The order matches the published stage-one aggregate.
            loss=feature*self.feature_weight+kl*self.kl_weight+velocity*self.velocity_weight
            return {'loss':loss,'reconstruction':feature,'kl':kl,'velocity':velocity}
        inputs,labels,targets=prepare_language_batch(self.processor(),self.bundle.unwrap_network(self.bundle.vae),batch['text'],motion,lengths,batch['tasks'])
        output,hidden=self.bundle.predict_language(**inputs)
        cls=language_loss(output.logits,labels,inputs['type_ids'])
        if hidden is None or not targets:
            diffusion=cls.new_zeros(())
        else:
            target=torch.cat(targets,dim=0)
            dimension=self.bundle.unwrap_network(self.bundle.lm).language_model.mot_embed_dim
            if not self.bundle.options['multi_hidden']:dimension*=self.bundle.options['motion_holder_repeat']
            hidden=hidden.view(len(target),-1,dimension)
            multiplier=self.bundle.options['diffusion_batch_mul']
            target=target.repeat(multiplier,1,1);hidden=hidden.repeat(multiplier,1,1)
            drop_mask=None
            if self.bundle.options['guidance_scale']>1.:
                drop_mask=[bool(np.random.rand(1)<self.bundle.options['guidance_uncondp']) for _ in hidden]
            timestep=torch.randint(0,self.train_diffusion.num_timesteps,(target.shape[0],),device=target.device)
            values=self.train_diffusion.training_losses(self.bundle.predict_diffusion,target,timestep,
                dict(c=hidden,drop_mask=drop_mask))
            diffusion=values['loss'].mean()
        return {'loss':cls*self.language_weight+diffusion*self.diffusion_weight,'language_loss':cls,'diffusion_loss':diffusion}

    @torch.no_grad()
    def val_step(self,batch):
        return {'val_'+key:value.detach() for key,value in self.train_step(batch).items()}
