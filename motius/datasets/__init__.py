"""Dataset exports loaded per method; explicit full registration remains available."""
from importlib import import_module

_EXPORTS = {
    "PreparedEDGEDataset": ("motius.datasets.prepared_dance", "PreparedEDGEDataset"),
    "PreparedMotionDataset": ("motius.datasets.prepared_motion", "PreparedMotionDataset"),
    "MotionStreamerTextTokenDataset": ("motius.datasets.motionstreamer", "MotionStreamerTextTokenDataset"),
    "MotionStreamerStreamingTokenDataset": ("motius.datasets.motionstreamer", "MotionStreamerStreamingTokenDataset"),
    "MotionStreamerTAEDataset": ("motius.datasets.motionstreamer", "MotionStreamerTAEDataset"),
    'AISTPPMusicDanceDataset': ('motius.datasets.aistpp_music_to_dance', 'AISTPPMusicDanceDataset'),
    'PipelineDataset': ('motius.datasets.base_dataset', 'PipelineDataset'),
    'TMRTextMotionDataset': ('motius.datasets.motion', 'TMRTextMotionDataset'),
    'ManifestTextMotionDataset': ('motius.datasets.text_motion', 'ManifestTextMotionDataset'),
    'PhysFlowPromptDataset': ('motius.datasets.gentrack', 'PhysFlowPromptDataset'),
    'GenTrackPromptDataset': ('motius.datasets.gentrack', 'GenTrackPromptDataset'),
    'MotionBricksSyntheticDataset': ('motius.datasets.motionbricks', 'SyntheticMotionDataset'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def register_all_datasets():
    for module, _ in dict.fromkeys(_EXPORTS.values()):
        import_module(module)
    from .motion import register_motion_datasets
    register_motion_datasets()
