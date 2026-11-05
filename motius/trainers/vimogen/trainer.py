"""Native ViMoGen masked, channel-weighted flow-matching objective."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.diffusion.vimogen import FlowMatchScheduler
from motius.motion.conditioning.vimogen import maybe_corrupt_ref_motion


@TRAINERS.register_module()
class ViMoGenTrainer(BaseTrainer):
    def __init__(self,bundle,ref_corruption=None,gradient_checkpointing=True,**kwargs):
        super().__init__(bundle,**kwargs)
        self.ref_corruption=ref_corruption;self.gradient_checkpointing=gradient_checkpointing
        self.scheduler=FlowMatchScheduler();self.scheduler.set_timesteps(1000,training=True)
        bundle.model.requires_grad_(True)
        bundle._trainable_modules=['model'];bundle._save_ckpt_modules=['model'];bundle._frozen_modules=['text_encoder']

    def train_step(self,batch):
        device=self.bundle.device;dtype=self.bundle.dtype
        motion=batch['motion'].to(device=device,dtype=dtype);mask=batch['motion_mask'].to(device=device,dtype=dtype)
        context=batch['prompt_emb'].to(device=device,dtype=dtype)
        reference=batch['ref_motion'].to(device=device,dtype=dtype).clone();reference_mask=batch['ref_motion_mask'].to(device=device,dtype=dtype).clone()
        dimension_mask=batch['motion_dim_mask'].to(device=device);attend=batch['attend_to_text_mask'].to(device=device)
        if self.ref_corruption is not None and (~attend.bool()).any():
            corrupted,corrupted_mask=maybe_corrupt_ref_motion(reference,reference_mask,self.ref_corruption,is_test=False)
            selected=~attend.bool();reference[selected]=corrupted.to(reference)[selected];reference_mask[selected]=corrupted_mask.to(reference_mask)[selected]
        noise=batch['noise'].to(motion) if 'noise' in batch else torch.randn_like(motion)
        indices=batch['timestep_ids'].cpu() if 'timestep_ids' in batch else torch.randint(0,self.scheduler.num_train_timesteps,(motion.shape[0],))
        timestep=self.scheduler.timesteps[indices].to(device=device,dtype=dtype)
        noisy=self.scheduler.add_noise(motion,noise,timestep).to(dtype)
        target=self.scheduler.training_target(motion,noise,timestep)
        context_manager=torch.amp.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context_manager:
            # The original graph derives the per-example attention branch from
            # reference content; the provided mask partitions logged losses only.
            prediction=self.bundle.predict_velocity(x=noisy,timestep=timestep,context=context,x_mask=mask,
                ref_motion=reference,ref_motion_mask=reference_mask,use_gradient_checkpointing=self.gradient_checkpointing)
            loss=F.mse_loss(prediction.float(),target.float(),reduction='none')
            dimension_mask=dimension_mask.unsqueeze(1).repeat(1,motion.shape[1],1)
            expanded=mask.unsqueeze(-1).expand(-1,-1,motion.shape[-1])
            weight=self.scheduler.training_weight(timestep).unsqueeze(-1).unsqueeze(-1).expand(-1,motion.shape[1],motion.shape[-1])
            loss_mask=dimension_mask*expanded
            channels=torch.ones(loss.shape[-1],device=device,dtype=dtype).view(1,1,-1);channels[:,:,258:]=3.
            values=loss*loss_mask*weight*channels
            total=values.reshape(-1).sum()/loss_mask.reshape(-1).sum()
        result={'loss':total}
        for name,selection in [('text',attend==1),('reference',attend==0)]:
            denominator=loss_mask[selection].reshape(-1).sum()
            if denominator!=0:result['loss_'+name]=values[selection].reshape(-1).sum()/denominator
        return result

    @torch.no_grad()
    def val_step(self,batch):
        return {'val_'+key:value.detach() for key,value in self.train_step(batch).items()}
