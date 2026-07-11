"""Base pipeline class."""

from abc import ABC
import importlib
from typing import Any, Optional

import torch

from motius.models.base_model_bundle import ModelBundle


class BasePipeline(ABC):
    """
    Abstract base class for all inference pipelines.

    Pipelines hold a ModelBundle and assemble the inference forward graph.
    All atomic forward functions are called from the bundle (shared with Trainer).
    """
    BUNDLE_CLS = None

    def __init__(self, bundle: ModelBundle, **kwargs):
        self.bundle = bundle
        self.bundle.eval()

    @classmethod
    def _resolve_bundle_cls(cls, bundle_cls=None):
        bundle_cls = bundle_cls or getattr(cls, "BUNDLE_CLS", None)
        if isinstance(bundle_cls, str):
            module_name, _, attr_name = bundle_cls.rpartition(".")
            if not module_name or not attr_name:
                raise ValueError(
                    f"{cls.__name__}.BUNDLE_CLS must be an import path, got {bundle_cls!r}"
                )
            bundle_cls = getattr(importlib.import_module(module_name), attr_name)
        if bundle_cls is None:
            raise TypeError(
                f"{cls.__name__}.from_pretrained(path) requires {cls.__name__}.BUNDLE_CLS "
                "or the legacy form from_pretrained(BundleCls, path)."
            )
        return bundle_cls

    @classmethod
    def from_config(cls, bundle_cls=None, bundle_cfg: Optional[dict] = None, **kwargs):
        """Build a pipeline from a bundle config dict."""
        if bundle_cfg is None and isinstance(bundle_cls, dict):
            bundle_cfg = bundle_cls
            bundle_cls = None
        bundle_cls = cls._resolve_bundle_cls(bundle_cls)
        bundle = bundle_cls.from_config(bundle_cfg)
        bundle.eval()
        return cls(bundle=bundle, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        bundle_cls_or_path,
        pretrained_model_name_or_path: Optional[str] = None,
        bundle_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Build a pipeline from a HuggingFace/diffusers-style pretrained artifact.

        Supported forms:

        - ``SomePipeline.from_pretrained("repo-or-path")`` when the pipeline
          defines ``BUNDLE_CLS``;
        - legacy ``SomePipeline.from_pretrained(BundleCls, "repo-or-path")``.
        """
        if pretrained_model_name_or_path is None:
            bundle_cls = cls._resolve_bundle_cls()
            pretrained_model_name_or_path = bundle_cls_or_path
        else:
            bundle_cls = cls._resolve_bundle_cls(bundle_cls_or_path)
        bundle = bundle_cls.from_pretrained(
            pretrained_model_name_or_path,
            **(bundle_kwargs or {}),
        )
        bundle.eval()
        return cls(bundle=bundle, **kwargs)

    @classmethod
    def from_checkpoint(cls, bundle_cls, bundle_cfg: dict, ckpt_path: str, **kwargs):
        """
        Build pipeline by loading a checkpoint into a ModelBundle.

        Args:
            bundle_cls: the ModelBundle subclass to instantiate
            bundle_cfg: config dict for the bundle
            ckpt_path: path to checkpoint file or directory
            **kwargs: additional args passed to cls.__init__
        """
        from motius.utils.checkpoint_utils import load_checkpoint

        if hasattr(bundle_cfg, 'to_dict'):
            bundle_cfg = bundle_cfg.to_dict()

        bundle = bundle_cls.from_config(bundle_cfg)
        state_dict = load_checkpoint(ckpt_path, map_location='cpu')
        bundle.load_state_dict_selective(state_dict)
        bundle.eval()
        return cls(bundle=bundle, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Run inference.

        T2M-style pipelines historically expose ``infer_t2m`` as their public
        task method. Keep ``__call__`` concrete so those pipelines can still be
        instantiated and called like a standard pipeline.
        """
        if hasattr(self, "infer_t2m"):
            return self.infer_t2m(*args, **kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")
