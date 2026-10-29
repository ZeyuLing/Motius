"""GEM-X frontend neural networks; video transforms live in Pipeline."""
from pathlib import Path
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .pose import GemXHeatmapNetwork

@MODEL_BUNDLES.register_module()
class GemXPoseBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        self.model=GemXHeatmapNetwork()
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.to(device).eval()
        self._save_ckpt_modules=['model']
    def forward(self,x):return self.model(x)

class GemXDetectorBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        import onnxruntime as ort
        providers=[]
        target=torch.device(device)
        if target.type=='cuda':providers.append(('CUDAExecutionProvider',{'device_id':target.index or 0}))
        import sys
        if sys.platform=='darwin':providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')
        options=ort.SessionOptions();options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(checkpoint),sess_options=options,providers=providers)
    def forward(self,images):return self.session.run(None,{self.session.get_inputs()[0].name:images})
