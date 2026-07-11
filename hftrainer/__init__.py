"""Motius core framework.

The first public release keeps the historical ``hftrainer`` namespace for
compatibility while the repository brand moves to Motius. Method packages are
added incrementally; importing the package should stay lightweight.
"""

from hftrainer.registry import (
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


def register_all_modules() -> None:
    """Register framework-level components.

    Method-specific models, trainers, and pipelines are intentionally not
    imported by the core release. They will be registered by each method package
    as it is opened.
    """

    import hftrainer.hooks.checkpoint_hook  # noqa: F401
    import hftrainer.hooks.ema_hook  # noqa: F401
    import hftrainer.hooks.logger_hook  # noqa: F401
    import hftrainer.hooks.lr_scheduler_hook  # noqa: F401
    import hftrainer.visualization.file_visualizer  # noqa: F401
    import hftrainer.visualization.tensorboard_visualizer  # noqa: F401
    import hftrainer.datasets.transforms  # noqa: F401


__all__ = [
    "DATASETS",
    "EVALUATORS",
    "HF_MODELS",
    "HOOKS",
    "MODEL_BUNDLES",
    "MODELS",
    "PIPELINES",
    "TRAINERS",
    "TRANSFORMS",
    "VISUALIZERS",
    "build_hf_model_from_cfg",
    "register_all_modules",
]
