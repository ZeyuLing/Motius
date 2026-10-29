"""VerMo language training with multimodal preparation outside the neural bundle."""
import torch
from torch import nn
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.conditioning.vermo.processor import build_vermo_processor
from .objective import chunked_logit_loss, modal_boundary_diagnostics


@TRAINERS.register_module()
class VermoTrainer(BaseTrainer):
    def __init__(self,bundle,processor=None,loss_chunk_size=128,**kwargs):
        super().__init__(bundle,**kwargs)
        from motius.configuration.vermo import configure_attention_runtime
        configure_attention_runtime()
        self.processor=processor if processor is not None else build_vermo_processor(bundle)
        self.loss_chunk_size=int(loss_chunk_size)
        if self.loss_chunk_size<=0:raise ValueError('loss_chunk_size must be positive')
        bundle.lm.requires_grad_(True).train()
        bundle.motion_tokenizer.requires_grad_(False).eval()
        bundle._trainable_modules=['lm'];bundle._save_ckpt_modules=['lm']
        bundle._frozen_modules=['motion_tokenizer']

    def train_step(self,batch):
        prepared=dict(batch) if 'input_ids' in batch else self.processor.process_train(batch)
        prepared={k:(v.to(self.bundle.device) if isinstance(v,torch.Tensor) else v) for k,v in prepared.items()}
        labels=prepared.pop('labels')
        weights=prepared.pop('loss_token_weights',None);boundary=prepared.pop('loss_boundary_mask',None)
        metadata={k:prepared.pop(k,None) for k in ['loss_fsq_aux_mask','motion_token_range','motion_fsq_code_levels','motion_fsq_canonical_to_code','motion_fsq_level_sizes','motion_fsq_aux_weight']}
        # Chunk projection occurs inside the wrapped neural graph, including
        # the tied vocabulary head. No trainable submodule bypasses DDP.
        output=self.bundle.forward_lm(dict(prepared,projection_chunk_size=self.loss_chunk_size,use_cache=False))
        identity=nn.Identity()
        loss,sums,counts=chunked_logit_loss(output.logits,labels,identity,output.logits.shape[-1],self.loss_chunk_size,
            token_weights=weights,fsq_aux_mask=metadata['loss_fsq_aux_mask'],motion_token_range=metadata['motion_token_range'],
            fsq_code_levels=metadata['motion_fsq_code_levels'],fsq_canonical_to_code=metadata['motion_fsq_canonical_to_code'],
            fsq_level_sizes=metadata['motion_fsq_level_sizes'],fsq_aux_weight=metadata['motion_fsq_aux_weight'] or 0.,return_per_sample=True)
        correct,count,boundary_loss=modal_boundary_diagnostics(output.logits,labels,boundary,identity)
        return dict(loss=loss,sample_loss_sums=sums,sample_loss_weights=counts,
                    boundary_correct=correct,boundary_count=count,boundary_loss_sum=boundary_loss)

    @torch.no_grad()
    def val_step(self,batch):
        return {'val_'+key:value.detach() for key,value in self.train_step(batch).items()}
