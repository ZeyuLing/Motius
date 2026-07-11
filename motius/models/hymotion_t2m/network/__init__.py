"""HYMotion-T2M network implementation and registry surface."""

from motius.models.hymotion_t2m.network.hymotion_dit import HunyuanMotionDiT
from motius.models.hymotion_t2m.network.hymotion_mmdit import HunyuanMotionMMDiT
from motius.registry import HF_MODELS


class HunyuanMotionT2MMMDiT(HunyuanMotionMMDiT):
    """T2M-namespaced HYMotion MMDiT backbone."""


if not HF_MODELS.get("HunyuanMotionT2MMMDiT"):
    HF_MODELS.register_module(
        name="HunyuanMotionT2MMMDiT",
        module=HunyuanMotionT2MMMDiT,
        force=True,
    )

if not HF_MODELS.get("HunyuanMotionT2MDiT"):
    HF_MODELS.register_module(
        name="HunyuanMotionT2MDiT",
        module=HunyuanMotionDiT,
        force=True,
    )

__all__ = ["HunyuanMotionT2MMMDiT", "HunyuanMotionMMDiT", "HunyuanMotionDiT"]
