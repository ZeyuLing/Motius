"""Trainer classes for motius."""

from motius.trainers.base_trainer import BaseTrainer
from motius.trainers.hymotion_t2m import HyMotionT2MTrainer
from motius.trainers.prism import PrismTrainer
from motius.trainers.tmr import TMRTrainer

__all__ = [
    "BaseTrainer",
    "HyMotionT2MTrainer",
    "PrismTrainer",
    "TMRTrainer",
]
