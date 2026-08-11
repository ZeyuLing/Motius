"""GenTrack online, offline, GRPO, and DPO trainers."""

from motius.registry import TRAINERS
from motius.trainers.gentrack.physflow_g1_dpo_trainer import PhysFlowG1DPOTrainer
from motius.trainers.gentrack.physflow_g1_grpo_trainer import (
    PhysFlowG1GRPOTrainer,
    PhysFlowG1RewardWeightedSFTTrainer,
)
from motius.trainers.gentrack.physflow_g1_offline_trainer import (
    PhysFlowG1OfflineTrainer,
)
from motius.trainers.gentrack.physflow_g1_trainer import PhysFlowG1Trainer


@TRAINERS.register_module(name="GenTrackFlowGRPOTrainer", force=True)
class GenTrackFlowGRPOTrainer(PhysFlowG1GRPOTrainer):
    """Public GenTrack Flow-GRPO trainer name."""


_ALIASES = {
    "GenTrackG1Trainer": PhysFlowG1Trainer,
    "GenTrackFlowDPOTrainer": PhysFlowG1DPOTrainer,
    "GenTrackRewardWeightedSFTTrainer": PhysFlowG1RewardWeightedSFTTrainer,
    "GenTrackOfflineTrainer": PhysFlowG1OfflineTrainer,
}
for _name, _module in _ALIASES.items():
    TRAINERS.register_module(name=_name, module=_module, force=True)

GenTrackG1Trainer = PhysFlowG1Trainer
GenTrackFlowDPOTrainer = PhysFlowG1DPOTrainer
GenTrackRewardWeightedSFTTrainer = PhysFlowG1RewardWeightedSFTTrainer
GenTrackOfflineTrainer = PhysFlowG1OfflineTrainer

__all__ = [
    "GenTrackFlowDPOTrainer",
    "GenTrackFlowGRPOTrainer",
    "GenTrackG1Trainer",
    "GenTrackOfflineTrainer",
    "GenTrackRewardWeightedSFTTrainer",
]
