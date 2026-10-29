# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license
import os
import math
import einops
import numpy as np
import torch
from torch import nn

from .common import LayerNorm2d
from ..dinov2.vision_transformer import vit_giant2, vit_large


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = 'dinov2_vitb14',
        out_chans: int = 256,
    ):
        super().__init__()
        self.backbone = self.get_backbone(backbone)
        self.backbone.encoder.mask_token.requires_grad = False

        self.neck = nn.Sequential(
            nn.Conv2d(
                self.backbone.embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        b, s, d = x.shape
        h = w = np.sqrt(s).astype(int)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.neck(x)
        return x

    def get_backbone(self, backbone:str):
        # To allow for: DINOV2, SAM, RADIO, etc
        if backbone == 'dinov2_vitb14':
            net = Dinov2Backbone(name='dinov2_vitb14', pretrained=True)
        elif backbone == 'dinov2_vitl14':
            net = Dinov2Backbone(name='dinov2_vitl14', pretrained=True)
        elif backbone == 'dinov2_vitg14':
            net = Dinov2Backbone(name='dinov2_vitg14', pretrained=True)
        else:
            raise Exception('Backbone not implemented.')
        return net




class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=True, *args, **kwargs):
        super().__init__()

        if name != 'dinov2_vitl14':
            raise ValueError('The released PromptHMR architecture uses dinov2_vitl14')
        vit = vit_large(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)

        self.name = name
        self.encoder = vit
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim



    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        y = self.encoder.get_intermediate_layers(x)[0]
        return y
