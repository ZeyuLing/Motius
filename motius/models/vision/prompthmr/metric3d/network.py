"""Metric3D's released neural encoder/recurrent depth-normal decoder graph."""
from torch import nn
from .backbone import vit_large_reg
from .decoder import RAFTDepthNormalDPT5

class MetricEncoderDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg.model.backbone.type!='vit_large_reg' or cfg.model.decode_head.type!='RAFTDepthNormalDPT5':
            raise ValueError('This checkpoint architecture is Metric3D ViT-Large with the released depth/normal decoder')
        self.encoder=vit_large_reg(**cfg.model.backbone)
        self.decoder=RAFTDepthNormalDPT5(cfg)

    def forward(self,input,**kwargs):return self.decoder(self.encoder(input),**kwargs)

class MetricDepthNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.depth_model=MetricEncoderDecoder(cfg)

    def forward(self,data):
        output=self.depth_model(**data)
        return output['prediction'],output['confidence'],output
