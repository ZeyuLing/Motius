"""Released GVHMR vision features and keypoint heatmap networks."""
import torch
from torch import nn
from einops import rearrange
from .vit import ViT
from .token_transformer import TransformerDecoder
from .heatmap import TopdownHeatmapSimpleHead


def huge_vit():
    return ViT(img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32,
               num_heads=16, ratio=1, use_checkpoint=False, mlp_ratio=4,
               qkv_bias=True, drop_path_rate=.55)


class FeatureTokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=TransformerDecoder(num_tokens=1,token_dim=1,dim=1024,
            depth=6,heads=8,mlp_dim=1024,dim_head=64,dropout=0.,emb_dropout=0.,
            norm="layer",context_dim=1280)

    def forward(self,features):
        context=rearrange(features,"b c h w -> b (h w) c")
        token=torch.zeros(len(features),1,1,device=features.device)
        return self.transformer(token,context=context).squeeze(1)


class HMR2FeatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=huge_vit()
        self.smpl_head=FeatureTokenHead()

    def forward(self,images):
        return self.smpl_head(self.backbone(images))


class ViTPoseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=huge_vit()
        self.keypoint_head=TopdownHeatmapSimpleHead(in_channels=1280,out_channels=17,
            num_deconv_layers=2,num_deconv_filters=(256,256),num_deconv_kernels=(4,4),
            extra=dict(final_conv_kernel=1))

    def forward(self,images):
        return self.keypoint_head(self.backbone(images))
