from motius.registry import TRAINERS
from motius.trainers.gem_trainer import GEMTrainer

@TRAINERS.register_module()
class GemXTrainer(GEMTrainer):
    """Native SOMA regression/diffusion primitive objectives."""
