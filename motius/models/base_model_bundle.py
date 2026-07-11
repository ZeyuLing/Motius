"""
ModelBundle base class.

ModelBundle holds all sub-modules for a task and serves as the shared core
between Trainer and Pipeline. It handles:
  - Module instantiation via HF_MODELS registry
  - Per-module trainable / save_ckpt control
  - Per-module precision and gradient-checkpointing control
  - LoRA injection via peft
  - Selective checkpoint save / load
  - Bundle-level construction helpers aligned with HuggingFace-style APIs
"""

import copy
import importlib
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ModelBundle(nn.Module):
    """
    Base class for all task-specific ModelBundles.

    Subclasses should call self._build_modules(modules_cfg) in __init__
    to instantiate and configure all sub-modules.

    Sub-modules declared with trainable=False are kept in eval mode during
    training (overridden in train()). Sub-modules with save_ckpt=True are
    included in state_dict_to_save() / load_state_dict_selective().
    """

    def __init__(self):
        super().__init__()
        self._save_ckpt_modules: List[str] = []
        self._trainable_modules: List[str] = []
        self._frozen_modules: List[str] = []
        self._lora_modules: List[str] = []
        self._module_checkpoint_formats: Dict[str, str] = {}
        self._module_build_configs: Dict[str, Dict[str, Any]] = {}
        # Non-nn.Module attributes (e.g. tokenizers)
        self._extra_attributes: Dict[str, Any] = {}

    _DTYPE_ALIASES = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'torch.float32': torch.float32,
        'fp16': torch.float16,
        'float16': torch.float16,
        'torch.float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'torch.bfloat16': torch.bfloat16,
    }
    _PRETRAINED_PATH_SENTINEL = '__pretrained__'

    # Optional declarative specs for HF-native bundles. Common single-model and
    # diffusers-style bundles can set these instead of hand-writing
    # _bundle_config_from_pretrained() / save_pretrained().
    #
    # HF_PRETRAINED_SPEC describes how one pretrained artifact maps to bundle
    # component configs. HF_SAVE_PRETRAINED_SPEC describes how the bundle
    # exports back to an inference artifact.
    HF_PRETRAINED_SPEC: Optional[Dict[str, Any]] = None
    HF_SAVE_PRETRAINED_SPEC: Optional[Dict[str, Any]] = None

    @staticmethod
    def _to_plain_dict(cfg: Optional[dict]) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, 'to_dict'):
            cfg = cfg.to_dict()
        return copy.deepcopy(dict(cfg))

    @classmethod
    def _resolve_pretrained_default(cls, value, pretrained_model_name_or_path: str):
        if value == cls._PRETRAINED_PATH_SENTINEL:
            return pretrained_model_name_or_path
        return copy.deepcopy(value)

    @staticmethod
    def _import_object(obj_or_path):
        if not isinstance(obj_or_path, str):
            return obj_or_path
        module_name, _, attr_name = obj_or_path.rpartition('.')
        if not module_name or not attr_name:
            raise ValueError(f"Expected import path like 'pkg.mod.Class', got: {obj_or_path!r}")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    @classmethod
    def _build_bundle_config_from_spec(
        cls,
        pretrained_model_name_or_path: str,
        spec: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        shared_pretrained_kwargs = kwargs.pop(
            spec.get('shared_pretrained_kwargs_arg', 'shared_pretrained_kwargs'),
            None,
        ) or {}

        bundle_cfg: Dict[str, Any] = {}

        for component_name, component_spec in spec.get('components', {}).items():
            component_type = kwargs.pop(
                component_spec.get('type_arg', f'{component_name}_type'),
                component_spec['default_type'],
            )
            component_overrides = kwargs.pop(
                component_spec.get('overrides_arg', f'{component_name}_overrides'),
                None,
            )
            component_pretrained_kwargs = kwargs.pop(
                component_spec.get('pretrained_kwargs_arg', f'{component_name}_kwargs'),
                None,
            )

            component_cfg = {'type': component_type}
            if component_spec.get('load_mode', 'from_pretrained') == 'from_pretrained':
                from_pretrained = {
                    'pretrained_model_name_or_path': pretrained_model_name_or_path,
                }
                subfolder = component_spec.get('subfolder')
                if subfolder:
                    from_pretrained['subfolder'] = subfolder
                cls._merge_nested_dict(from_pretrained, shared_pretrained_kwargs)
                cls._merge_nested_dict(
                    from_pretrained,
                    component_spec.get('from_pretrained_defaults'),
                )
                cls._merge_nested_dict(from_pretrained, component_pretrained_kwargs)
                component_cfg['from_pretrained'] = from_pretrained
            cls._merge_nested_dict(component_cfg, component_spec.get('cfg_defaults'))
            cls._merge_nested_dict(component_cfg, component_overrides)
            bundle_cfg[component_name] = component_cfg

        for field_name, field_spec in spec.get('init_args', {}).items():
            if isinstance(field_spec, dict):
                arg_name = field_spec.get('arg', field_name)
                default_value = field_spec.get('default')
            else:
                arg_name = field_name
                default_value = field_spec
            value = kwargs.pop(arg_name, default_value)
            bundle_cfg[field_name] = cls._resolve_pretrained_default(
                value,
                pretrained_model_name_or_path,
            )

        if kwargs:
            unexpected = ', '.join(sorted(kwargs))
            raise TypeError(
                f"Unexpected from_pretrained kwargs for {cls.__name__}: {unexpected}"
            )

        return bundle_cfg

    def _save_pretrained_from_spec(
        self,
        save_directory: str,
        spec: Dict[str, Any],
        merge_lora: bool = True,
        safe_serialization: bool = True,
        **kwargs,
    ):
        from motius.utils.hf_export import safe_hf_export

        os.makedirs(save_directory, exist_ok=True)

        merge_modules = [
            name for name in spec.get('merge_lora_modules', [])
            if self.is_lora_module(name)
        ]
        if merge_lora and merge_modules:
            self.merge_lora_weights(merge_modules)

        export_kind = spec.get('kind', 'module')
        if export_kind == 'module':
            module = getattr(self, spec['module'])
            with safe_hf_export():
                module.save_pretrained(
                    save_directory,
                    safe_serialization=safe_serialization,
                    **kwargs,
                )
        elif export_kind == 'pipeline':
            pipeline_cls = self._import_object(spec['pipeline_class'])
            pipeline_kwargs = {}
            for ctor_key, attr_name in spec.get('components', {}).items():
                pipeline_kwargs[ctor_key] = getattr(self, attr_name)
            pipeline_kwargs.update(copy.deepcopy(spec.get('pipeline_kwargs', {})))
            pipeline = pipeline_cls(**pipeline_kwargs)
            with safe_hf_export():
                pipeline.save_pretrained(
                    save_directory,
                    safe_serialization=safe_serialization,
                    **kwargs,
                )
        else:
            raise ValueError(
                f"Unsupported HF_SAVE_PRETRAINED_SPEC kind '{export_kind}' "
                f"for {type(self).__name__}."
            )

        for extra_spec in spec.get('extra_artifacts', []):
            if isinstance(extra_spec, str):
                extra_spec = {'attr': extra_spec}
            attr_name = extra_spec['attr']
            artifact = getattr(self, attr_name, None)
            if artifact is None or not hasattr(artifact, 'save_pretrained'):
                continue
            subdir = extra_spec.get('subdir')
            artifact_dir = (
                save_directory
                if subdir in (None, '', '.')
                else os.path.join(save_directory, subdir)
            )
            artifact.save_pretrained(artifact_dir)

    @classmethod
    def _resolve_module_dtype(cls, dtype_spec) -> torch.dtype:
        if isinstance(dtype_spec, torch.dtype):
            return dtype_spec
        if isinstance(dtype_spec, str):
            dtype = cls._DTYPE_ALIASES.get(dtype_spec)
            if dtype is not None:
                return dtype
        raise ValueError(
            "module_dtype must be one of: fp32, fp16, bf16, float32, float16, "
            "bfloat16, torch.float32, torch.float16, torch.bfloat16, or torch.dtype."
        )

    @staticmethod
    def _normalize_gradient_checkpointing_cfg(cfg_value) -> Dict[str, Any]:
        if cfg_value in (False, None):
            return {}
        if cfg_value is True:
            return {}
        if isinstance(cfg_value, dict):
            return copy.deepcopy(cfg_value)
        raise ValueError(
            "gradient_checkpointing must be a bool or a dict of keyword arguments."
        )

    @classmethod
    def _enable_gradient_checkpointing(cls, module: nn.Module, module_name: str, cfg_value):
        kwargs = cls._normalize_gradient_checkpointing_cfg(cfg_value)

        if hasattr(module, 'gradient_checkpointing_enable'):
            try:
                module.gradient_checkpointing_enable(**kwargs)
                return
            except TypeError:
                if kwargs:
                    try:
                        module.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs=kwargs
                        )
                        return
                    except TypeError as exc:
                        raise ValueError(
                            f"Module '{module_name}' exposes gradient_checkpointing_enable(), "
                            f"but rejected kwargs {kwargs}."
                        ) from exc
                raise

        if hasattr(module, 'enable_gradient_checkpointing'):
            try:
                module.enable_gradient_checkpointing(**kwargs)
                return
            except TypeError as exc:
                raise ValueError(
                    f"Module '{module_name}' exposes enable_gradient_checkpointing(), "
                    f"but rejected kwargs {kwargs}."
                ) from exc

        raise ValueError(
            f"Module '{module_name}' does not expose gradient checkpointing hooks. "
            "Expected gradient_checkpointing_enable() or enable_gradient_checkpointing()."
        )

    @classmethod
    def _bundle_type_matches(cls, cfg_type) -> bool:
        if cfg_type is cls:
            return True
        if isinstance(cfg_type, str):
            if cfg_type == cls.__name__:
                return True
            from motius.registry import MODEL_BUNDLES
            return MODEL_BUNDLES.get(cfg_type) is cls
        return False

    @staticmethod
    def _merge_nested_dict(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not overrides:
            return base

        for key, value in overrides.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                ModelBundle._merge_nested_dict(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        """
        Construct a bundle from an MMEngine/HF-Trainer-style config dict.

        If ``cfg`` contains a ``type`` field pointing at another registered
        bundle subclass, dispatch through the registry. Otherwise instantiate
        ``cls`` directly with the remaining keyword arguments.
        """
        cfg_dict = cls._to_plain_dict(cfg)
        if kwargs:
            cfg_dict = cls._merge_nested_dict(cfg_dict, kwargs)

        if 'type' in cfg_dict:
            cfg_type = cfg_dict.get('type')
            if not cls._bundle_type_matches(cfg_type):
                from motius.registry import MODEL_BUNDLES
                return MODEL_BUNDLES.build(cfg_dict)
            cfg_dict.pop('type')

        return cls(**cfg_dict)

    @classmethod
    def from_cfg(cls, cfg: Optional[dict] = None, **kwargs):
        """Alias kept for MMEngine-style naming."""
        return cls.from_config(cfg, **kwargs)

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        if cls.HF_PRETRAINED_SPEC is not None:
            return cls._build_bundle_config_from_spec(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                spec=cls.HF_PRETRAINED_SPEC,
                **kwargs,
            )
        raise NotImplementedError(
            f"{cls.__name__}.from_pretrained() is only available for bundles that "
            "define how a HuggingFace/diffusers pretrained artifact maps to bundle "
            "sub-modules. Override _bundle_config_from_pretrained(), or use "
            f"{cls.__name__}.from_config(...) for custom/self-developed models."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Construct a bundle from a HuggingFace/diffusers-style pretrained artifact.

        The public API is generic, but subclasses can override
        ``_bundle_config_from_pretrained()`` to describe how one artifact should
        be split into the bundle's sub-module config.
        """
        bundle_cfg = cls._bundle_config_from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )
        bundle_cfg = cls._merge_nested_dict(bundle_cfg, config_overrides)
        return cls.from_config(bundle_cfg)

    def save_pretrained(self, save_directory: str, **kwargs):
        if type(self).HF_SAVE_PRETRAINED_SPEC is not None:
            return self._save_pretrained_from_spec(
                save_directory=save_directory,
                spec=type(self).HF_SAVE_PRETRAINED_SPEC,
                **kwargs,
            )
        raise NotImplementedError(
            f"{type(self).__name__}.save_pretrained() is task-specific. "
            "HF-native bundles should override it to export an artifact that "
            "official diffusers/transformers APIs can read. Custom bundles can "
            "continue to rely on checkpoint save/load, or implement save_pretrained() "
            "for their own artifact format."
        )

    def _build_modules(self, modules_cfg: dict):
        """
        Instantiate all sub-modules from config dict.

        Each entry in modules_cfg is a sub-module config with optional keys:
          - trainable: bool | 'lora' (default True)
          - save_ckpt: bool (default: True if trainable else False)
          - module_dtype: torch dtype override applied after construction
          - gradient_checkpointing: bool | dict
          - All other keys are passed to HF_MODELS.build()
        """
        from motius.registry import HF_MODELS
        from motius.models.peft_utils import apply_lora

        self._save_ckpt_modules = []
        self._trainable_modules = []
        self._frozen_modules = []
        self._lora_modules = []
        self._module_checkpoint_formats = {}
        self._module_build_configs = {}

        for name, sub_cfg in modules_cfg.items():
            sub_cfg = copy.deepcopy(sub_cfg)
            trainable = sub_cfg.pop('trainable', True)
            trainable_spec = trainable
            lora_cfg = sub_cfg.pop('lora_cfg', None)
            save_ckpt = sub_cfg.pop('save_ckpt', True if trainable else False)
            gradient_checkpointing = sub_cfg.pop('gradient_checkpointing', False)
            module_dtype_spec = sub_cfg.pop('module_dtype', None)
            checkpoint_format = sub_cfg.pop('checkpoint_format', None)
            if checkpoint_format is None:
                checkpoint_format = 'lora' if trainable == 'lora' else 'full'
            if checkpoint_format not in {'full', 'lora'}:
                raise ValueError(
                    f"Unsupported checkpoint_format '{checkpoint_format}' for module '{name}'. "
                    "Expected 'full' or 'lora'."
                )
            if checkpoint_format == 'lora' and trainable != 'lora':
                raise ValueError(
                    f"Module '{name}' sets checkpoint_format='lora' but trainable={trainable!r}. "
                    "Only modules declared with trainable='lora' can save adapter-only checkpoints."
                )
            normalized_cfg = copy.deepcopy(sub_cfg)
            normalized_cfg['trainable'] = trainable_spec
            normalized_cfg['save_ckpt'] = save_ckpt
            normalized_cfg['checkpoint_format'] = checkpoint_format
            if lora_cfg is not None:
                normalized_cfg['lora_cfg'] = copy.deepcopy(lora_cfg)
            if gradient_checkpointing:
                normalized_cfg['gradient_checkpointing'] = copy.deepcopy(
                    gradient_checkpointing
                )
            if module_dtype_spec is not None:
                normalized_cfg['module_dtype'] = module_dtype_spec
            self._module_build_configs[name] = normalized_cfg

            # Build the module
            module = HF_MODELS.build(sub_cfg)

            if isinstance(module, nn.Module) and gradient_checkpointing:
                self._enable_gradient_checkpointing(module, name, gradient_checkpointing)

            # Apply LoRA if requested
            if trainable == 'lora':
                module = apply_lora(module, lora_cfg or {})
                trainable = True
                save_ckpt = True  # always save lora weights
                self._lora_modules.append(name)

            if isinstance(module, nn.Module):
                if module_dtype_spec is not None:
                    module_dtype = self._resolve_module_dtype(module_dtype_spec)
                    module = module.to(dtype=module_dtype)

                if not trainable:
                    module.requires_grad_(False)
                    self._frozen_modules.append(name)
                else:
                    self._trainable_modules.append(name)

                setattr(self, name, module)
                self._module_checkpoint_formats[name] = checkpoint_format

                if save_ckpt:
                    self._save_ckpt_modules.append(name)
            else:
                # Non-nn.Module (e.g. tokenizer, scheduler with no params)
                # Store as plain attribute
                if gradient_checkpointing:
                    raise ValueError(
                        f"Module '{name}' is not an nn.Module, so gradient_checkpointing "
                        "cannot be applied."
                    )
                if module_dtype_spec is not None:
                    raise ValueError(
                        f"Module '{name}' is not an nn.Module, so module_dtype cannot be applied."
                    )
                self._extra_attributes[name] = module
                object.__setattr__(self, name, module)

    def get_module_checkpoint_format(self, name: str) -> str:
        return self._module_checkpoint_formats.get(name, 'full')

    def is_lora_module(self, name: str) -> bool:
        return name in self._lora_modules

    def get_module_build_cfg(self, name: str) -> Dict[str, Any]:
        return copy.deepcopy(self._module_build_configs.get(name, {}))

    def get_module_pretrained_path(self, name: str) -> Optional[str]:
        build_cfg = self.get_module_build_cfg(name)
        from_pretrained = build_cfg.get('from_pretrained') or {}
        if isinstance(from_pretrained, dict):
            return from_pretrained.get('pretrained_model_name_or_path')
        return None

    def checkpoint_metadata(self) -> Dict[str, Any]:
        modules = {}
        for name in self._save_ckpt_modules:
            modules[name] = {
                'checkpoint_format': self.get_module_checkpoint_format(name),
                'is_lora': self.is_lora_module(name),
                'trainable': name in self._trainable_modules,
            }
        return {'format_version': 2, 'modules': modules}

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all trainable parameters for optimizer construction.

        Includes:
          - Parameters from all ``_trainable_modules`` sub-modules.
          - **Bundle-level** ``nn.Parameter`` attributes (e.g. ``null_vtxt_feat``,
            ``null_ctxt_input``) that are direct children of the bundle but not
            inside any registered sub-module.  These are collected via
            ``self.named_parameters(recurse=False)``.

        Historical note: before this fix, bundle-level Parameters were silently
        excluded from the optimizer, so they were never trained.  This caused
        inference divergence when the null embeddings (loaded from a pretrained
        checkpoint) were not saved in M2M/T2M checkpoints.
        """
        params = []
        for name in self._trainable_modules:
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                params.extend(
                    param for param in module.parameters() if param.requires_grad
                )
        # Bundle-level nn.Parameters (direct children, not in any sub-module)
        for _name, param in self.named_parameters(recurse=False):
            if param.requires_grad:
                params.append(param)
        return params

    def trainable_named_parameters(self):
        """Yield (name, param) for all trainable parameters."""
        for mod_name in self._trainable_modules:
            module = getattr(self, mod_name)
            if isinstance(module, nn.Module):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        yield f"{mod_name}.{param_name}", param
        # Bundle-level nn.Parameters
        for param_name, param in self.named_parameters(recurse=False):
            if param.requires_grad:
                yield param_name, param

    def get_module_parameters(self, *names: str) -> List[torch.nn.Parameter]:
        """
        Return parameters from specified sub-modules.

        Useful for building per-optimizer parameter groups when optimizer
        names don't match module names (e.g. 'student' optimizer for
        'student_unet' module).

        Args:
            *names: one or more module names registered in this bundle

        Returns:
            Flat list of parameters from the specified modules.

        Raises:
            ValueError: if a module name is not found or is not nn.Module
        """
        params = []
        for name in names:
            module = getattr(self, name, None)
            if module is None:
                raise ValueError(
                    f"Module '{name}' not found in bundle. "
                    f"Available: {self._trainable_modules + self._frozen_modules}"
                )
            if isinstance(module, nn.Module):
                params.extend(
                    param for param in module.parameters() if param.requires_grad
                )
            else:
                raise ValueError(
                    f"Module '{name}' is not an nn.Module (type: {type(module).__name__}). "
                    f"Cannot extract parameters."
                )
        return params

    def state_dict_to_save(self) -> Dict[str, dict]:
        """
        Return a nested state dict containing only save_ckpt=True modules,
        plus any bundle-level nn.Parameters and registered buffers.

        Format: {module_name: state_dict, '__bundle_params__': {...}}

        Automatically unwraps DDP/FSDP wrappers so checkpoint keys are clean
        (without ``module.`` prefix).
        """
        from motius.models.peft_utils import get_lora_state_dict

        sd = {}
        meta = self.checkpoint_metadata()
        if meta['modules']:
            sd['__motius_meta__'] = meta
        for name in self._save_ckpt_modules:
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                # Unwrap DDP / FSDP wrappers to get clean keys
                save_target = module
                while hasattr(save_target, 'module'):
                    save_target = save_target.module
                if self.get_module_checkpoint_format(name) == 'lora':
                    sd[name] = get_lora_state_dict(save_target)
                else:
                    sd[name] = save_target.state_dict()

        # Save bundle-level nn.Parameters and buffers (direct children only).
        # These live outside any sub-module and would otherwise be lost.
        bundle_params = {}
        for pname, param in self.named_parameters(recurse=False):
            bundle_params[pname] = param.data.clone()
        for bname, buf in self.named_buffers(recurse=False):
            bundle_params[bname] = buf.clone()
        if bundle_params:
            sd['__bundle_params__'] = bundle_params

        return sd

    def load_state_dict_selective(
        self,
        state_dict: Dict[str, dict],
        strict: bool = False,
        exclude_bundle_keys: Optional[list] = None,
        exclude_module_keys: Optional[list] = None,
        skip_frozen: bool = False,
    ):
        """
        Load only modules that are present in state_dict.
        Modules not present in state_dict are left unchanged.

        Also restores bundle-level nn.Parameters and buffers from
        ``'__bundle_params__'`` if present in the checkpoint.

        Args:
            state_dict: {module_name: module_state_dict} or flat state dict
            strict: whether to enforce strict key matching per module
            exclude_bundle_keys: list of bundle-level parameter/buffer names
                to skip when loading from ``__bundle_params__``.  Use this
                when loading a checkpoint whose mean/std (or other buffers)
                were computed for a different representation and should NOT
                overwrite the values already initialised by the bundle's
                ``__init__``.  Example: ``['mean', 'std']`` to preserve
                KIMODO Root stats when loading an SMPL Root checkpoint.
            exclude_module_keys: list of top-level module names to skip when
                loading. Use this when warm-starting most of a bundle while
                keeping config-initialised modules such as a newly trained VAE.
            skip_frozen: if True, skip loading weights for parameters that
                have ``requires_grad=False``.  Use this when resuming from
                a checkpoint that has degraded/collapsed values for modules
                that have already been frozen (e.g. caption_freeze_strategy
                freezes text encoders from T2M pretrained, but the resume
                checkpoint has collapsed encoder weights from unconditioned
                training).
        """
        if not state_dict:
            return

        from motius.models.peft_utils import (
            looks_like_lora_state_dict,
            set_lora_state_dict,
        )

        checkpoint_meta = {}
        if '__motius_meta__' in state_dict:
            meta = state_dict.pop('__motius_meta__')
            if isinstance(meta, dict):
                checkpoint_meta = meta.get('modules', {}) or {}

        # Restore bundle-level parameters / buffers saved by state_dict_to_save().
        bundle_params = state_dict.pop('__bundle_params__', None)
        if bundle_params and isinstance(bundle_params, dict):
            _exclude = set(exclude_bundle_keys or [])
            for pname, pval in bundle_params.items():
                if pname in _exclude:
                    from motius.utils.logger import get_logger
                    get_logger().info(
                        f"Skipping excluded bundle key '{pname}' "
                        f"(shape {tuple(pval.shape)}) — preserving "
                        f"config-initialised value."
                    )
                    continue
                if hasattr(self, pname):
                    attr = getattr(self, pname)
                    if isinstance(attr, nn.Parameter):
                        # skip_frozen: don't overwrite frozen bundle params
                        if skip_frozen and not attr.requires_grad:
                            from motius.utils.logger import get_logger
                            get_logger().info(
                                f"skip_frozen: skipping frozen bundle param "
                                f"'{pname}' (shape {tuple(pval.shape)})"
                            )
                            continue
                        if attr.shape == pval.shape:
                            attr.data.copy_(pval)
                        else:
                            from motius.utils.logger import get_logger
                            get_logger().warning(
                                f"Shape mismatch for bundle param '{pname}': "
                                f"ckpt {tuple(pval.shape)} vs model {tuple(attr.shape)}, skipped"
                            )
                    elif isinstance(attr, torch.Tensor):
                        if attr.shape == pval.shape:
                            attr.copy_(pval)

        if not state_dict:
            return

        # Detect format: nested {module_name: {weight_name: tensor}} or flat
        first_val = next(iter(state_dict.values()))
        if isinstance(first_val, torch.Tensor):
            # Flat state dict — try to split by module name
            nested = {}
            for key, val in state_dict.items():
                parts = key.split('.', 1)
                if len(parts) == 2 and hasattr(self, parts[0]):
                    mod_name, param_name = parts
                    if mod_name not in nested:
                        nested[mod_name] = {}
                    nested[mod_name][param_name] = val
                else:
                    # Try direct load
                    nested[key] = {key: val}
            state_dict = nested

        exclude_modules = set(exclude_module_keys or [])
        for name, sd in state_dict.items():
            if name in exclude_modules:
                from motius.utils.logger import get_logger
                get_logger().info(
                    f"Skipping excluded module '{name}' from checkpoint; "
                    "preserving config-initialised module."
                )
                continue
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    checkpoint_format = checkpoint_meta.get(name, {}).get('checkpoint_format')
                    if checkpoint_format is None and self.is_lora_module(name):
                        checkpoint_format = 'lora' if looks_like_lora_state_dict(sd) else 'full'

                    if checkpoint_format == 'lora':
                        set_lora_state_dict(module, sd)
                        continue

                    # Unwrap DDP / FSDP wrappers so checkpoint keys (without
                    # ``module.`` prefix) match the underlying model's state_dict.
                    load_target = module
                    while hasattr(load_target, 'module'):
                        load_target = load_target.module

                    # Filter out shape-mismatched parameters to avoid RuntimeError.
                    # PyTorch's load_state_dict with strict=False still raises on
                    # shape mismatches; we skip them gracefully and warn.
                    if not strict:
                        target_sd = load_target.state_dict()
                        shape_mismatched = []
                        for k, v in list(sd.items()):
                            if k in target_sd and isinstance(v, torch.Tensor):
                                if v.shape != target_sd[k].shape:
                                    shape_mismatched.append(
                                        f"{k}: ckpt {tuple(v.shape)} vs model {tuple(target_sd[k].shape)}"
                                    )
                                    del sd[k]
                        if shape_mismatched:
                            from motius.utils.logger import get_logger
                            logger = get_logger()
                            logger.warning(
                                f"Skipped {len(shape_mismatched)} shape-mismatched params "
                                f"in module '{name}': {shape_mismatched[:5]}"
                            )

                    # Skip frozen parameters: don't overwrite params that have
                    # requires_grad=False (e.g. encoders frozen via
                    # caption_freeze_strategy).  This lets us resume from an
                    # intermediate checkpoint without destroying the T2M-pretrained
                    # encoder weights that were loaded and frozen during __init__.
                    if skip_frozen:
                        frozen_skipped = []
                        for pname, param in load_target.named_parameters():
                            if not param.requires_grad and pname in sd:
                                frozen_skipped.append(pname)
                                del sd[pname]
                        if frozen_skipped:
                            from motius.utils.logger import get_logger
                            logger = get_logger()
                            logger.info(
                                f"skip_frozen: skipped {len(frozen_skipped)} frozen "
                                f"params in module '{name}': "
                                f"{frozen_skipped[:5]}{'...' if len(frozen_skipped) > 5 else ''}"
                            )

                    missing, unexpected = load_target.load_state_dict(sd, strict=strict)
                    if missing:
                        from motius.utils.logger import get_logger
                        logger = get_logger()
                        logger.warning(f"Missing keys in module '{name}': {missing[:5]}...")
                    if unexpected:
                        from motius.utils.logger import get_logger
                        logger = get_logger()
                        logger.warning(f"Unexpected keys in module '{name}': {unexpected[:5]}...")
                elif isinstance(module, (nn.Parameter, torch.Tensor)):
                    # Handle direct nn.Parameter or buffer attributes.
                    # sd is a dict like {name: tensor} from the flat→nested conversion.
                    tensor_val = sd.get(name) if isinstance(sd, dict) else sd
                    if isinstance(tensor_val, torch.Tensor):
                        if isinstance(module, nn.Parameter):
                            if module.shape == tensor_val.shape:
                                module.data.copy_(tensor_val)
                        else:
                            # buffer
                            if module.shape == tensor_val.shape:
                                module.copy_(tensor_val)

    def merge_lora_weights(self, module_names: Optional[List[str]] = None):
        """Merge LoRA adapters into their base modules for inference/export."""
        from motius.models.peft_utils import is_peft_model, merge_lora

        target_names = module_names or list(self._lora_modules)
        for name in target_names:
            module = getattr(self, name, None)
            if module is None or not isinstance(module, nn.Module):
                continue
            if not is_peft_model(module):
                continue

            merged_module = merge_lora(module)
            if name in self._frozen_modules:
                merged_module.requires_grad_(False)
                merged_module.eval()
            setattr(self, name, merged_module)
            self._module_checkpoint_formats[name] = 'full'
            if name in self._lora_modules:
                self._lora_modules.remove(name)

    def train(self, mode: bool = True):
        """
        Override train() to keep frozen modules always in eval mode.
        This ensures BatchNorm / Dropout in frozen modules behave correctly during training.
        """
        super().train(mode)
        if mode:
            for name in self._frozen_modules:
                module = getattr(self, name, None)
                if isinstance(module, nn.Module):
                    module.eval()
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ModelBundle.forward() is not implemented directly. "
            "Use the atomic forward methods (encode_text, predict_noise, etc.) instead."
        )
