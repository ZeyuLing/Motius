"""Native UniMuMo dual-stream and caption objectives on encoded conditions."""
import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.conditioning.unimumo_pattern import DelayedPattern,restore_prediction_logits


def codebook_cross_entropy(logits,targets,mask):
    total=torch.zeros([],device=targets.device);values=[]
    for book in range(targets.shape[1]):
        projected=logits[:,book].contiguous().view(-1,logits.size(-1))
        labels=targets[:,book].contiguous().view(-1);valid=mask[:,book].contiguous().view(-1)
        loss=F.cross_entropy(projected[valid],labels[valid]);total+=loss;values.append(loss.detach())
    return total/targets.shape[1],values


@TRAINERS.register_module()
class UniMuMoTrainer(BaseTrainer):
    def __init__(self,bundle,stage='train_music_motion',motion_weight=.15,**kwargs):
        super().__init__(bundle,**kwargs)
        if stage not in ['train_music_motion','train_caption']:raise ValueError(stage)
        self.stage=stage;self.motion_weight=motion_weight
        bundle.requires_grad_(False)
        core=bundle.unwrap_network(bundle.core)
        if stage=='train_music_motion':
            core.generator.requires_grad_(True);core.text_condition_projection.requires_grad_(True)
            for layer in core.generator.layers:layer.caption_attention.requires_grad_(False)
            bundle._trainable_modules=['core'];bundle._frozen_modules=['audio_codec','text_encoder','captioner']
        else:
            for layer in core.generator.layers:layer.caption_attention.requires_grad_(True)
            core.caption_context_projection.requires_grad_(True);bundle.captioner.requires_grad_(True)
            bundle._trainable_modules=['core','captioner'];bundle._frozen_modules=['audio_codec','text_encoder']
        bundle._save_ckpt_modules=list(bundle._trainable_modules)

    def train_step(self,batch):
        music=batch['music_codes'].to(self.bundle.device);motion=batch['motion_codes'].to(self.bundle.device)
        model=self.bundle.unwrap_network(self.bundle.core).generator
        pattern=DelayedPattern(music.shape[-1],tuple(range(model.num_codebooks)))
        music_input,_=pattern.build(music,special_token=model.special_token_id,valid_only=True)
        motion_input,_=pattern.build(motion,special_token=model.special_token_id,valid_only=True)
        condition=batch['condition'].to(self.bundle.device);condition_mask=batch['condition_mask'].to(self.bundle.device)
        project_condition=batch.get('project_condition',True)
        if self.stage=='train_music_motion':
            logits_music,logits_motion=self.bundle.predict_codes(music_input,motion_input,condition=condition,
                condition_mask=condition_mask,project_condition=project_condition)
            logits_music,mask_music=restore_prediction_logits(logits_music,music.shape[-1],pattern.delays)
            logits_motion,mask_motion=restore_prediction_logits(logits_motion,motion.shape[-1],pattern.delays)
            loss_music,_=codebook_cross_entropy(logits_music,music,mask_music)
            loss_motion,_=codebook_cross_entropy(logits_motion,motion,mask_motion)
            return {'loss':loss_music*(1-self.motion_weight)+loss_motion*self.motion_weight,'music_loss':loss_music,'motion_loss':loss_motion}
        mode=batch['caption_mode']
        if mode not in ['music','motion']:raise ValueError(mode)
        if mode=='music':motion_input[:]=model.special_token_id
        else:music_input[:]=model.special_token_id
        context=self.bundle.predict_codes(music_input,motion_input,condition=condition,condition_mask=condition_mask,
            project_condition=project_condition,caption_mode=True)
        mask=torch.zeros(context.shape[:2],dtype=torch.long,device=context.device);middle=context.shape[1]//2
        mask[:,:middle] = int(mode=='music');mask[:,middle:]=int(mode=='motion')
        labels=batch['labels'].to(self.bundle.device).clone();labels[labels==0]=-100
        from transformers.modeling_outputs import BaseModelOutput
        result=self.bundle.captioner(encoder_outputs=BaseModelOutput(last_hidden_state=context),attention_mask=mask,
            labels=labels,decoder_attention_mask=batch['decoder_attention_mask'].to(self.bundle.device))
        return {'loss':result.loss,'caption_loss':result.loss}

    @torch.no_grad()
    def val_step(self,batch):
        return {'val_'+key:value.detach() for key,value in self.train_step(batch).items()}
