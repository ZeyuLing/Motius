"""Dataset exports, loaded on demand to avoid unrelated method dependencies."""
from importlib import import_module

_EXPORTS = {
    "AISTPPMusicDanceDataset": ("motius.datasets.aistpp_music_to_dance", "AISTPPMusicDanceDataset"),
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
