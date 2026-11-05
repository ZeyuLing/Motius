"""GEM-SMPL's exact released COCO-17 heatmap network."""
from pathlib import Path
import torch
from motius.models.base_model_bundle import ModelBundle
from .pose import ViT,TopdownHeatmapSimpleHead,VitPoseModel

class GemSmplPoseBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        backbone=ViT(img_size=(256,192),patch_size=16,embed_dim=1280,depth=32,num_heads=16,ratio=1,mlp_ratio=4,qkv_bias=True,drop_path_rate=.55)
        head=TopdownHeatmapSimpleHead(in_channels=1280,out_channels=17,num_deconv_layers=2,num_deconv_filters=(256,256),num_deconv_kernels=(4,4),extra=dict(final_conv_kernel=1))
        self.model=VitPoseModel(backbone,head)
        state=torch.load(Path(checkpoint),map_location='cpu',weights_only=False)
        state=state.get('state_dict',state.get('model',state))
        self.model.load_state_dict(state,strict=True)
        self.model.to(device).eval()
        self._save_ckpt_modules=['model'];self._frozen_modules=['model']

    def forward(self,images):return self.model(images)
