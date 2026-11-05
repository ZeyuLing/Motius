"""Metric3D neural graph and strict released-checkpoint import."""
from pathlib import Path
import json
import torch
from mmengine import Config
from motius.models.base_model_bundle import ModelBundle
from motius.utils.torch_checkpoint import load_torch_checkpoint
from .network import MetricDepthNetwork

class PromptMetricDepthBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda',dtype=torch.float16):
        super().__init__()
        path=Path(__file__).resolve().parents[4]/'configuration/prompthmr/metric3d.json'
        self.config=Config(json.loads(path.read_text()))
        self.model=MetricDepthNetwork(self.config)
        state=load_torch_checkpoint(checkpoint,weights_only=False)['model_state_dict']
        # The released file omits the unused pretraining mask token. The
        # original architecture initializes it to exact zeros; retain that
        # declared constant while requiring every learned inference tensor.
        key='depth_model.encoder.mask_token'
        if key not in state:
            token=self.model.depth_model.encoder.mask_token
            if torch.count_nonzero(token).item()!=0:raise ValueError('Unexpected source mask-token initialization')
            state={**state,key:token.detach().clone()}
        self.model.load_state_dict(state,strict=True)
        self._save_ckpt_modules=['model'];self._trainable_modules=['model']
        self.to(device=device,dtype=dtype).eval()

    @property
    def device(self):return next(self.model.parameters()).device
    def forward(self,data):return self.model(data)
