"""Dataset exports, loaded on demand to avoid unrelated method dependencies."""
from importlib import import_module

_EXPORTS = {
    "PreparedMotionDataset": ("motius.datasets.prepared_motion", "PreparedMotionDataset"),
    "PreparedTM2DDataset": ("motius.datasets.prepared_tm2d", "PreparedTM2DDataset"),
    "PhysFlowPromptDataset": ("motius.datasets.gentrack", "PhysFlowPromptDataset"),
    "GenTrackPromptDataset": ("motius.datasets.gentrack", "GenTrackPromptDataset"),
    "MotionBricksSyntheticDataset": ("motius.datasets.motionbricks", "SyntheticMotionDataset"),
    "AISTPPMusicDanceDataset": ("motius.datasets.aistpp_music_to_dance", "AISTPPMusicDanceDataset"),
    "MotionStreamerTextTokenDataset": ("motius.datasets.motionstreamer", "MotionStreamerTextTokenDataset"),
    "MotionStreamerStreamingTokenDataset": ("motius.datasets.motionstreamer", "MotionStreamerStreamingTokenDataset"),
    "MotionStreamerTAEDataset": ("motius.datasets.motionstreamer", "MotionStreamerTAEDataset"),
    "PromptHMRVideoDataset": ("motius.datasets.prompthmr", "PromptHMRVideoDataset"),
    "CalibrationImageDataset": ("motius.datasets.prompthmr", "CalibrationImageDataset"),
    "MoGenDiTDataset": ("motius.datasets.mogendit", "MoGenDiTDataset"),
    "PreparedDARTDataset": ("motius.datasets.prepared_dart", "PreparedDARTDataset"),
    "PreparedEDGEDataset": ("motius.datasets.prepared_dance", "PreparedEDGEDataset"),
    "PipelineDataset": ("motius.datasets.base_dataset", "PipelineDataset"),
    "TMRTextMotionDataset": ("motius.datasets.motion", "TMRTextMotionDataset"),
    "ManifestTextMotionDataset": ("motius.datasets.text_motion", "ManifestTextMotionDataset"),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

def register_all_datasets():
    for module, _ in _EXPORTS.values():
        import_module(module)
