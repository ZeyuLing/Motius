"""Native auxiliary and spatial-temporal MoGenTS token training."""
import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from . import objective


@TRAINERS.register_module()
class MoGenTSTrainer(BaseTrainer):
    MODULES={"aux_base":"mask_transformer_aux","ts_base":"mask_transformer_ts",
             "aux_residual":"res_transformer_aux","ts_residual":"res_transformer_ts"}

    def __init__(self,bundle,stage="ts_base",**kwargs):
        super().__init__(bundle,**kwargs)
        if stage not in self.MODULES:raise ValueError(f"Unknown MoGenTS stage: {stage}")
        self.stage=stage
        name=self.MODULES[stage]
        for key in [*self.MODULES.values(),"vq_model","fusion","length_estimator"]:
            module=getattr(bundle,key,None)
            if module is not None:
                module.requires_grad_(key==name)
                model=bundle.unwrap_network(module)
                if hasattr(model,"clip_model"):model.clip_model.requires_grad_(False).eval()
        bundle._trainable_modules=[name];bundle._save_ckpt_modules=[name]
        bundle._frozen_modules=[key for key in [*self.MODULES.values(),"vq_model","fusion","length_estimator"]
                                if key!=name and getattr(bundle,key,None) is not None]

    def train_step(self,batch):
        network=self.bundle.unwrap_network(getattr(self.bundle,self.MODULES[self.stage]))
        predict=lambda *args,**kwargs:self.bundle.predict_tokens(self.stage,*args,**kwargs)
        loss,_,accuracy=getattr(objective,f"compute_{self.stage}_objective")(
            network,predict,batch["tokens"].to(self.bundle.device),batch["text"],batch["token_lens"].to(self.bundle.device))
        return {"loss":loss,"mogents_accuracy":torch.as_tensor(accuracy,device=loss.device)}

    @torch.no_grad()
    def val_step(self,batch):
        return {"val_"+key:value.detach() for key,value in self.train_step(batch).items()}
