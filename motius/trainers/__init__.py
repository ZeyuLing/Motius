"""Native trainer exports, imported only when the corresponding method is used.

Explicit framework-wide registration is available through register_all_trainers.
A method-specific launch does not import unrelated methods or their dependencies.
"""
from importlib import import_module
from .base_trainer import BaseTrainer

_EXPORTS = {
    "TM2DTrainer": ("motius.trainers.tm2d", "TM2DTrainer"),
    "EDGETrainer": ("motius.trainers.edge", "EDGETrainer"),
    "MotionLCMTrainer": ("motius.trainers.motionlcm", "MotionLCMTrainer"),
    "MLDTrainer": ("motius.trainers.mld", "MLDTrainer"),
    "MotionCLRTrainer": ("motius.trainers.motionclr", "MotionCLRTrainer"),
    "ProjFlowTrainer": ("motius.trainers.projflow", "ProjFlowTrainer"),
    "MDMTrainer": ("motius.trainers.mdm", "MDMTrainer"),
    "CondMDITrainer": ("motius.trainers.condmdi", "CondMDITrainer"),
    "OmniControlTrainer": ("motius.trainers.omnicontrol", "OmniControlTrainer"),
    "MotionMillionTrainer": ("motius.trainers.motionmillion", "MotionMillionTrainer"),
    "TM2TTrainer": ("motius.trainers.tm2t", "TM2TTrainer"),
    "T2MGPTTrainer": ("motius.trainers.t2mgpt", "T2MGPTTrainer"),
    "MotionStreamerTrainer": ("motius.trainers.motionstreamer", "MotionStreamerTrainer"),
    'MotionCLIPTrainer': ('motius.trainers.motion_clip', 'MotionCLIPTrainer'),
    'BailandoTrainer': ('motius.trainers.bailando', 'BailandoTrainer'),
    'MoMaskTrainer': ('motius.trainers.momask', 'MoMaskTrainer'),
    'MoGenTSTrainer': ('motius.trainers.mogents', 'MoGenTSTrainer'),
    'MaskControlTrainer': ('motius.trainers.maskcontrol', 'MaskControlTrainer'),
    'FlowHMRTrainer': ('motius.trainers.flowhmr', 'FlowHMRTrainer'),
    'FlowMDMTrainer': ('motius.trainers.flowmdm', 'FlowMDMTrainer'),
    'GVHMRTrainer': ('motius.trainers.gvhmr', 'GVHMRTrainer'),
    'GenTrackFlowGRPOTrainer': ('motius.trainers.gentrack', 'GenTrackFlowGRPOTrainer'),
    'HyMotionT2MTrainer': ('motius.trainers.hymotion_t2m', 'HyMotionT2MTrainer'),
    'PrismTrainer': ('motius.trainers.prism', 'PrismTrainer'),
    'ProtoMotionsTrainer': ('motius.trainers.protomotions', 'ProtoMotionsTrainer'),
    'SonicTrainer': ('motius.trainers.sonic', 'SonicTrainer'),
    'SONICOnlinePPOTrainer': ('motius.trainers.sonic', 'SONICOnlinePPOTrainer'),
    'SONICPPOTrainer': ('motius.trainers.sonic', 'SONICPPOTrainer'),
    'ProtoMotionsPPOTrainer': ('motius.trainers.protomotions', 'ProtoMotionsPPOTrainer'),
    'TMRTrainer': ('motius.trainers.tmr', 'TMRTrainer'),
    'InterGenTrainer': ('motius.trainers.intergen', 'InterGenTrainer'),
    'InterMaskTrainer': ('motius.trainers.intermask', 'InterMaskTrainer'),
    'MotionGPTTrainer': ('motius.trainers.motiongpt', 'MotionGPTTrainer'),
    'MotionGPT3Trainer': ('motius.trainers.motiongpt3', 'MotionGPT3Trainer'),
    'UniMuMoTrainer': ('motius.trainers.unimumo', 'UniMuMoTrainer'),
    'ViMoGenTrainer': ('motius.trainers.vimogen', 'ViMoGenTrainer'),
    'DARTTrainer': ('motius.trainers.dart', 'DARTTrainer'),
    'VermoTrainer': ('motius.trainers.vermo', 'VermoTrainer'),
    'GemSmplTrainer': ('motius.trainers.gem_smpl', 'GemSmplTrainer'),
    'GemXTrainer': ('motius.trainers.gem_x', 'GemXTrainer'),
    'MotionBricksPoseTrainer': ('motius.trainers.motionbricks', 'MotionBricksPoseTrainer'),
    'MotionBricksRootTrainer': ('motius.trainers.motionbricks', 'MotionBricksRootTrainer'),
    'MotionBricksVQTrainer': ('motius.trainers.motionbricks', 'MotionBricksVQTrainer'),
    'PromptHMRVideoTrainer': ('motius.trainers.prompthmr', 'PromptHMRVideoTrainer'),
    'BeyondMimicTrainer': ('motius.trainers.beyondmimic', 'BeyondMimicTrainer'),
}

__all__ = ['BaseTrainer', *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def register_all_trainers():
    """Register all method trainers for explicit framework-wide discovery."""
    for module, _ in dict.fromkeys(_EXPORTS.values()):
        import_module(module)
