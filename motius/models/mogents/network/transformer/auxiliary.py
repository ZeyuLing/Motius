"""Auxiliary MoGenTS networks with the released trainable position encoding."""
from torch import nn
from motius.models.momask.network.mask_transformer.transformer import (
    MaskTransformer as _MaskTransformer,
    ResidualTransformer as _ResidualTransformer,
)


def _trainable_positions(network):
    positions=network.position_enc.pe
    del network.position_enc.pe
    network.position_enc.register_parameter('pe',nn.Parameter(positions))


class MaskTransformer(_MaskTransformer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        _trainable_positions(self)


class ResidualTransformer(_ResidualTransformer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        _trainable_positions(self)
