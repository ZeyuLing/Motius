"""DROID-Calib feature/context encoders and recurrent update network only."""
from pathlib import Path
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.utils.torch_checkpoint import load_torch_checkpoint
from .encoders import BasicEncoder
from .update import UpdateModule

class PromptDroidBundle(ModelBundle):
    def __init__(self,checkpoint=None,device='cuda'):
        super().__init__()
        self.fnet=BasicEncoder(output_dim=128,norm_fn='instance')
        self.cnet=BasicEncoder(output_dim=256,norm_fn='none')
        self.update=UpdateModule()
        if checkpoint is not None:
            state=load_torch_checkpoint(checkpoint,weights_only=False)
            state={key.replace('module.',''):value for key,value in state.items()}
            for key in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']:
                state[key]=state[key][:2]
            self.load_state_dict(state,strict=True)
        self._trainable_modules=['fnet','cnet','update'];self._save_ckpt_modules=['fnet','cnet','update']
        self.to(device).eval()

    def forward_features(self,images):return self.fnet(images)
    def forward_context(self,images):return self.cnet(images)
    def forward_update(self,*args,**kwargs):return self.update(*args,**kwargs)
