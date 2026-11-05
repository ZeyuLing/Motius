"""SOMA decoder with the published numeric-mask behavior."""

from .encdec_double_cond import DoubleCondDecoder


class SOMADecoder(DoubleCondDecoder):
    def forward(
        self,
        x,
        external_cond=None,
        target_cond=None,
        has_target_cond=None,
        token_mask=None,
    ):
        if token_mask is not None:
            token_mask = token_mask.to(dtype=x.dtype)
        return super().forward(
            x, external_cond, target_cond, has_target_cond, token_mask
        )
