"""Native OmniControl guidance-aware denoising objective."""

from motius.registry import TRAINERS
from motius.trainers.gaussian_motion import GaussianMotionTrainer


@TRAINERS.register_module()
class OmniControlTrainer(GaussianMotionTrainer):
    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, method="omnicontrol", **kwargs)
