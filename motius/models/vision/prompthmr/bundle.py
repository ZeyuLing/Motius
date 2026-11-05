"""PromptHMR detection, segmentation, keypoint and temporal-mask networks."""
from pathlib import Path
import torch
from motius.utils.torch_checkpoint import load_torch_checkpoint
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.vision.gvhmr.networks import huge_vit
from motius.models.vision.gvhmr.heatmap import TopdownHeatmapSimpleHead


class PromptPoseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=huge_vit()
        self.keypoint_head=TopdownHeatmapSimpleHead(in_channels=1280,out_channels=25,
            num_deconv_layers=2,num_deconv_filters=(256,256),num_deconv_kernels=(4,4),extra=dict(final_conv_kernel=1))

    def forward(self,images):return self.keypoint_head(self.backbone(images))


class PromptPoseBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__();self.model=PromptPoseNetwork()
        state=load_torch_checkpoint(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        self.model.load_state_dict(state.get('state_dict',state),strict=True)
        self.to(device).eval();self._save_ckpt_modules=['model']

    def forward(self,images):return self.model(images)


class PromptDetectorBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        cfg=get_cfg()
        config=Path(__file__).resolve().parents[3]/'configuration/prompthmr/detectron2/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
        cfg.merge_from_file(str(config));cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=.25
        cfg.MODEL.DEVICE=str(device)
        self.config=cfg
        self.model=build_model(cfg)
        if Path(checkpoint).name=='detector_native.pt':
            self.model.load_state_dict(load_torch_checkpoint(checkpoint,weights_only=True),strict=True)
        else:DetectionCheckpointer(self.model).load(str(checkpoint))
        self.to(device).eval();self._save_ckpt_modules=['model']

    def forward(self,inputs):return self.model.inference(inputs,do_postprocess=False)


class PromptSegmentationBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50
        self.model=deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=True)
        self.model.load_state_dict(load_torch_checkpoint(checkpoint,map_location='cpu',weights_only=True,mmap=True),strict=True)
        self.to(device).eval();self._save_ckpt_modules=['model']

    def forward(self,images):return self.model(images)


class PromptMaskBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        config=Path(__file__).resolve().parents[3]/'configuration/prompthmr/sam2_hiera_t.yaml'
        cfg=OmegaConf.load(config).model
        cfg._target_='motius.models.vision.prompthmr.sam2.PromptSAM2Network'
        cfg.sam_mask_decoder_extra_args=dict(dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=.05,dynamic_multimask_stability_thresh=.98)
        cfg.binarize_mask_from_pts_for_mem_enc=True
        self.model=instantiate(cfg,_recursive_=True)
        state=load_torch_checkpoint(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        self.model.load_state_dict(state['model'],strict=True)
        self.to(device).eval();self._save_ckpt_modules=['model']

    @property
    def network(self):return getattr(self.model,'module',self.model)
    @property
    def device(self):return next(self.model.parameters()).device
    def forward(self,*args,**kwargs):return self.model(*args,**kwargs)


class PromptCameraBundle(ModelBundle):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__()
        from .camera import CameraRegressorNetwork
        self.model=CameraRegressorNetwork()
        state=load_torch_checkpoint(checkpoint,map_location='cpu',weights_only=False,mmap=True)['state_dict']
        if any(not key.startswith('model.') for key in state):raise ValueError('Unexpected camera checkpoint component')
        self.model.load_state_dict({key.removeprefix('model.'):value for key,value in state.items()},strict=True)
        self.to(device).eval();self._save_ckpt_modules=['model']

    def forward(self,images):return self.model(images)
