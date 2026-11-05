"""MotionBricks pipelines; legacy navigation dependencies remain optional."""
from importlib import import_module

__all__ = ['MotionBricksPipeline', 'MotionBricksReleasePipeline']


def __getattr__(name):
    modules = {'MotionBricksPipeline': 'pipeline', 'MotionBricksReleasePipeline': 'release_pipeline'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f'{__name__}.{modules[name]}'), name)
    globals()[name] = value
    return value
