"""Native MDM objective, prepared-motion input contract."""

from motius.registry import TRAINERS
from motius.trainers.gaussian_motion import GaussianMotionTrainer


@TRAINERS.register_module()
class MDMTrainer(GaussianMotionTrainer):
    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, method="mdm", **kwargs)
