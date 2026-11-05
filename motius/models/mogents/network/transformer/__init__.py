"""MoGenTS neural architectures with shared auxiliary token networks."""
from .auxiliary import MaskTransformer,ResidualTransformer
from .transformer_ts import MaskTransformer2D,ResidualTransformer2D
__all__=["MaskTransformer","ResidualTransformer","MaskTransformer2D","ResidualTransformer2D"]
