"""Motius core framework.

The public core release keeps imports lightweight. Method packages are opened
incrementally and register their own models, trainers, datasets, and pipelines.
"""

from motius.registry import (
    DATASETS,
    EVALUATORS,
    HF_MODELS,
    HOOKS,
    MODEL_BUNDLES,
    MODELS,
    PIPELINES,
    TRAINERS,
    TRANSFORMS,
    VISUALIZERS,
    build_hf_model_from_cfg,
)
from motius.pipelines.auto import Pipeline


def register_all_modules() -> None:
    """Register framework-level components.

    Lightweight datasets and trainers are registered here. Method-specific
    model packages remain opt-in through each config's ``custom_imports`` list.
    """

    import motius.hooks.checkpoint_hook  # noqa: F401
    import motius.hooks.ema_hook  # noqa: F401
    import motius.hooks.logger_hook  # noqa: F401
    import motius.hooks.lr_scheduler_hook  # noqa: F401
    import motius.visualization.file_visualizer  # noqa: F401
    import motius.visualization.tensorboard_visualizer  # noqa: F401
    import motius.datasets.transforms  # noqa: F401
    from motius.datasets import register_all_datasets
    register_all_datasets()
    import motius.evaluation.evaluators  # noqa: F401
    import motius.models.tmr  # noqa: F401
    from motius.trainers import register_all_trainers
    register_all_trainers()


__all__ = [
    "DATASETS",
    "EVALUATORS",
    "HF_MODELS",
    "HOOKS",
    "MODEL_BUNDLES",
    "MODELS",
    "PIPELINES",
    "Pipeline",
    "TRAINERS",
    "TRANSFORMS",
    "VISUALIZERS",
    "build_hf_model_from_cfg",
    "register_all_modules",
]
