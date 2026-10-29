"""Native MotionGPT instruction and VQ training through prepared networks."""
import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .objective import forward_encdec


@TRAINERS.register_module()
class MotionGPTTrainer(BaseTrainer):
    def __init__(self,bundle,stage='lm_instruct',feature_weight=1.,velocity_weight=.5,
                 commitment_weight=.02,language_weight=1.,reconstruction_loss='l1_smooth',**kwargs):
        super().__init__(bundle,**kwargs)
        if stage not in ['vae','lm_pretrain','lm_instruct']:raise ValueError(stage)
        self.stage=stage;self.feature_weight=feature_weight;self.velocity_weight=velocity_weight
        self.commitment_weight=commitment_weight;self.language_weight=language_weight
        self.reconstruction_loss=reconstruction_loss;self._processor=None
        name='vae' if stage=='vae' else 'lm'
        bundle._trainable_modules=[name];bundle._save_ckpt_modules=[name]
        bundle._frozen_modules=['lm' if name=='vae' else 'vae']
        for key in ['lm','vae']:getattr(bundle,key).requires_grad_(key==name)

    def processor(self):
        if self._processor is None:
            from transformers import AutoTokenizer
            from motius.motion.conditioning.motiongpt import MotionGPTTextProcessor
            tokenizer=AutoTokenizer.from_pretrained(self.bundle.t5_model_path,legacy=True,local_files_only=self.bundle.local_files_only)
            tokenizer.add_tokens([f'<motion_id_{i}>' for i in range(self.bundle.codebook_size+3)])
            self._processor=MotionGPTTextProcessor(tokenizer,self.bundle.device)
        self._processor.device=self.bundle.device
        return self._processor

    def train_step(self,batch):
        if self.stage=='vae':
            motion=batch['motion'].to(self.bundle.device)
            reconstructed,commitment,perplexity=self.bundle.vae(motion)
            fn={'l1':F.l1_loss,'l2':F.mse_loss,'l1_smooth':F.smooth_l1_loss}[self.reconstruction_loss]
            feature=fn(reconstructed,motion);velocity=fn(reconstructed[...,4:67],motion[...,4:67])
            return {'loss':feature*self.feature_weight+velocity*self.velocity_weight+commitment*self.commitment_weight,
                'reconstruction':feature,'velocity':velocity,'commitment':commitment,'perplexity':perplexity}
        if 'input_ids' in batch:
            output=self.bundle.predict_language(**{k:batch[k].to(self.bundle.device) for k in ['input_ids','attention_mask','labels','decoder_attention_mask'] if k in batch})
        else:
            output=forward_encdec(self.processor(),self.bundle.predict_language,batch['text'],
                batch['motion_tokens'].to(self.bundle.device),batch['token_lens'],batch['tasks'])
        return {'loss':output.loss*self.language_weight,'language_loss':output.loss}

    @torch.no_grad()
    def val_step(self,batch):
        return {'val_'+k:v.detach() for k,v in self.train_step(batch).items()}
