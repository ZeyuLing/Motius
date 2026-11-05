"""Native ProtoMotions PPO/AMP objectives and training entry point."""

from .train import ProtoMotionsTrainer
from .trainer import ProtoMotionsPPOTrainer

__all__ = ["ProtoMotionsTrainer", "ProtoMotionsPPOTrainer"]
