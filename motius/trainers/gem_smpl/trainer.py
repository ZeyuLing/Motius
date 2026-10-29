from motius.registry import TRAINERS
from motius.trainers.gem_trainer import GEMTrainer

@TRAINERS.register_module()
class GemSmplTrainer(GEMTrainer):
    """Native SMPL regression/diffusion primitive objectives."""
