"""Native CondMDI objective and prepared observation conditioning."""

from motius.registry import TRAINERS
from motius.trainers.gaussian_motion import GaussianMotionTrainer


@TRAINERS.register_module()
class CondMDITrainer(GaussianMotionTrainer):
    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, method="condmdi", **kwargs)
