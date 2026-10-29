"""Trainer exports, loaded on demand."""
from importlib import import_module
from motius.trainers.base_trainer import BaseTrainer

_EXPORTS = {
    "FlowHMRTrainer": "flowhmr",
    "MotionCLRTrainer": "motionclr",
    "MDMTrainer": "mdm",
    "CondMDITrainer": "condmdi",
    "GenTrackFlowGRPOTrainer": "gentrack",
    "HyMotionT2MTrainer": "hymotion_t2m",
    "PrismTrainer": "prism",
    "ProtoMotionsTrainer": "protomotions",
    "SonicTrainer": "sonic",
    "TMRTrainer": "tmr",
}
__all__ = ["BaseTrainer", *_EXPORTS]

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"motius.trainers.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

def register_all_trainers():
    for module in _EXPORTS.values():
        import_module(f"motius.trainers.{module}")
