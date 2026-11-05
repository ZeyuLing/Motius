"""MotionBricks bundles; runtime dependencies are loaded on demand."""
from importlib import import_module

__all__ = ['MotionBricksBundle', 'MotionBricksReleaseBundle']


def __getattr__(name):
    modules = {'MotionBricksBundle': 'bundle', 'MotionBricksReleaseBundle': 'release_bundle'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f'{__name__}.{modules[name]}'), name)
    globals()[name] = value
    return value
