"""Trainer classes for motius."""

from motius.trainers.base_trainer import BaseTrainer
from motius.trainers.gentrack import GenTrackFlowGRPOTrainer
from motius.trainers.hymotion_t2m import HyMotionT2MTrainer
from motius.trainers.prism import PrismTrainer
from motius.trainers.protomotions import ProtoMotionsTrainer
from motius.trainers.sonic import SonicTrainer
from motius.trainers.tmr import TMRTrainer

__all__ = [
    "BaseTrainer",
    "GenTrackFlowGRPOTrainer",
    "HyMotionT2MTrainer",
    "PrismTrainer",
    "ProtoMotionsTrainer",
    "SonicTrainer",
    "TMRTrainer",
]
