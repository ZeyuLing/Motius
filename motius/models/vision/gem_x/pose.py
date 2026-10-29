# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
import torch
from torch import nn
from motius.models.vision.dinov3.models.vision_transformer import DinoVisionTransformer
class TopdownHeatmapSimpleHead(nn.Module):

    def __init__(self, in_channels, out_channels, num_deconv_layers=2, num_deconv_filters=(256, 256), num_deconv_kernels=(4, 4), extra=None):
        super().__init__()
        layers = []
        in_ch = in_channels
        for out_ch, k in zip(num_deconv_filters, num_deconv_kernels):
            padding = 1 if k == 4 else 1 if k == 3 else 0
            output_padding = 0 if k == 4 else 1 if k == 3 else 0
            layers += [nn.ConvTranspose2d(in_ch, out_ch, k, stride=2, padding=padding, output_padding=output_padding, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.deconv_layers = nn.Sequential(*layers)
        final_k = extra.get('final_conv_kernel', 1) if extra else 1
        pad = 1 if final_k == 3 else 0
        self.final_layer = nn.Conv2d(in_ch, out_channels, final_k, padding=pad)

    def forward(self, x):
        if isinstance(x, list | tuple):
            x = x[-1]
        return self.final_layer(self.deconv_layers(x))

class GemXBackbone(DinoVisionTransformer):
    def forward(self,x):
        return self.get_intermediate_layers(x,n=1,reshape=True,return_class_token=False,return_extra_tokens=False,norm=True)[0]

class GemXHeatmapNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=GemXBackbone(img_size=(256,192),patch_size=16,embed_dim=1280,depth=32,num_heads=20,ffn_ratio=6,n_storage_tokens=4,layerscale_init=1e-5,mask_k_bias=True,ffn_layer='swiglu')
        self.keypoint_head=TopdownHeatmapSimpleHead(in_channels=1280,out_channels=77)
    def forward(self,x):return self.keypoint_head(self.backbone(x))
