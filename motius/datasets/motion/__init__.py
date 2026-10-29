"""Dataset exports loaded per method; explicit full registration remains available."""
from importlib import import_module

_EXPORTS = {
    'MotionDatasetUnion': ('motius.datasets.motion.dataset_union', 'MotionDatasetUnion'),
    'HYMotionOfficialT2MDataset': ('motius.datasets.motion.hymotion_official_t2m_dataset', 'HYMotionOfficialT2MDataset'),
    'TMRTextMotionDataset': ('motius.datasets.motion.tmr_text_motion_dataset', 'TMRTextMotionDataset'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def register_motion_datasets():
    for module, _ in dict.fromkeys(_EXPORTS.values()):
        import_module(module)
