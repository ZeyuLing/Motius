"""GenTrack neural generator and immutable reference-network checkpoints."""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from motius.models.hymotion_t2m.bundle import HyMotionT2MBundle
from motius.registry import MODEL_BUNDLES




@MODEL_BUNDLES.register_module()
class PhysFlowG1Bundle(HyMotionT2MBundle):
    """G1-native flow-matching generator for the PhysFlow online loop."""

    # Frames the generator was trained at; sampling pads to >= this like eval.
    TRAIN_FRAMES = 360

    def __init__(
        self,
        *args,
        sample_steps: int = 50,
        sample_guidance: float = 1.0,
        immutable_anchor_checkpoint: Optional[str] = None,
        require_immutable_anchor: bool = False,
        trainable_motion_parameter_prefixes: Optional[Sequence[str]] = None,
        freeze_condition_embeddings_when_restricted: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_steps = int(sample_steps)
        self.sample_guidance = float(sample_guidance)
        self.immutable_anchor_checkpoint = (
            str(immutable_anchor_checkpoint)
            if immutable_anchor_checkpoint not in (None, "")
            else None
        )
        self.require_immutable_anchor = bool(require_immutable_anchor)
        self.trainable_motion_parameter_prefixes = tuple(
            str(prefix).strip()
            for prefix in (trainable_motion_parameter_prefixes or ())
            if str(prefix).strip()
        )
        self.freeze_condition_embeddings_when_restricted = bool(
            freeze_condition_embeddings_when_restricted
        )
        self._restricted_trainable_motion_parameters: Tuple[str, ...] = ()
        if self.trainable_motion_parameter_prefixes:
            self._restrict_trainable_motion_parameters()
        # Optional immutable G0 reference (explicit checkpoint). When set, this
        # replaces the legacy lazy deepcopy anchor for reference KL / distill.
        self._immutable_g0_transformer: Optional[nn.Module] = None
        # Legacy lazy anchor (captured on first use when immutable G0 is off).
        self._anchor_transformer: Optional[nn.Module] = None

    def config_dict(self):
        config = super().config_dict()
        config.update(
            sample_steps=self.sample_steps,
            sample_guidance=self.sample_guidance,
        )
        return config

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        from motius.configuration.gentrack import specialize_artifact

        specialize_artifact(save_directory)
        return result

    def _device(self) -> torch.device:
        return next(self.motion_transformer.parameters()).device

    def _restrict_trainable_motion_parameters(self) -> None:
        """Freeze the generator except for an explicit parameter-prefix scope."""
        core = self._core_transformer
        core.requires_grad_(False)
        matched = []
        for name, parameter in core.named_parameters():
            if any(
                name == prefix.rstrip(".") or name.startswith(prefix)
                for prefix in self.trainable_motion_parameter_prefixes
            ):
                parameter.requires_grad_(True)
                matched.append(name)

        if not matched:
            available = [name for name, _ in core.named_parameters()]
            raise ValueError(
                "trainable_motion_parameter_prefixes matched no transformer "
                f"parameters: {self.trainable_motion_parameter_prefixes!r}; "
                f"examples={available[:8]!r}"
            )

        if self.freeze_condition_embeddings_when_restricted:
            for name in (
                "null_vtxt_feat",
                "null_ctxt_input",
                "special_game_vtxt_feat",
                "special_game_ctxt_feat",
            ):
                parameter = getattr(self, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.requires_grad_(False)

        self._restricted_trainable_motion_parameters = tuple(matched)

    def trainable_motion_parameter_scope(self) -> Dict[str, object]:
        """Return the configured post-training scope for logging and tests."""
        names = self._restricted_trainable_motion_parameters
        name_set = set(names)
        numel = sum(
            parameter.numel()
            for name, parameter in self._core_transformer.named_parameters()
            if name in name_set
        )
        return {
            "prefixes": self.trainable_motion_parameter_prefixes,
            "parameter_tensors": len(names),
            "parameter_numel": numel,
            "parameter_names": names,
        }

    @property
    def _core_transformer(self):
        """The underlying MMDiT, unwrapping accelerate's DDP/FSDP wrapper.

        Under multi-GPU the runner replaces ``bundle.motion_transformer`` with a
        ``DistributedDataParallel`` wrapper. Calling it (``forward``) routes
        through DDP fine, but ATTRIBUTE access (``output_dim``) and ``deepcopy``
        must target the wrapped module, not the DDP shell.
        """
        mt = self.motion_transformer
        return getattr(mt, "module", mt)

    @staticmethod
    def _extract_motion_transformer_state(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not state_dict:
            raise ValueError("empty checkpoint state dict")
        if "motion_transformer" in state_dict and isinstance(
            state_dict["motion_transformer"], dict
        ):
            return dict(state_dict["motion_transformer"])
        nested = {
            key.split(".", 1)[1]: value
            for key, value in state_dict.items()
            if key.startswith("motion_transformer.") and isinstance(value, Tensor)
        }
        if nested:
            return nested
        if all(isinstance(value, Tensor) for value in state_dict.values()):
            return dict(state_dict)
        raise ValueError(
            "checkpoint does not contain motion_transformer weights "
            "(expected nested 'motion_transformer' or 'motion_transformer.*' keys)"
        )

    def init_immutable_g0_anchor(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        force: bool = False,
    ) -> None:
        """Load a frozen G0 transformer from an explicit checkpoint path.

        Idempotent across resume/round boundaries: once initialized, the G0
        reference is never replaced unless ``force=True``.
        """
        if self._immutable_g0_transformer is not None and not force:
            return

        path = checkpoint_path or self.immutable_anchor_checkpoint
        if not path:
            if self.require_immutable_anchor:
                raise RuntimeError(
                    "immutable G0 anchor is required but no checkpoint path was configured"
                )
            return

        resolved = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.exists(resolved):
            raise RuntimeError(f"immutable G0 anchor checkpoint not found: {resolved}")

        from motius.utils.checkpoint_utils import load_checkpoint

        try:
            state_dict = load_checkpoint(resolved, map_location="cpu")
            mt_state = self._extract_motion_transformer_state(state_dict)
            g0_core = deepcopy(self._core_transformer)
            g0_core.load_state_dict(mt_state, strict=True)
            g0_core.requires_grad_(False)
            g0_core.eval()
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize immutable G0 anchor from {resolved}"
            ) from exc

        object.__setattr__(self, "_immutable_g0_transformer", g0_core)
        object.__setattr__(self, "_anchor_transformer", g0_core)

    def immutable_g0_anchor_fingerprint(self) -> Optional[float]:
        """Cheap checksum for tests; None when immutable G0 is not initialized."""
        if self._immutable_g0_transformer is None:
            return None
        first = next(self._immutable_g0_transformer.parameters(), None)
        if first is None:
            return 0.0
        return float(first.detach().float().sum().item())

    def _reference_transformer(self) -> Optional[nn.Module]:
        if self._immutable_g0_transformer is not None:
            anc = self._immutable_g0_transformer
        else:
            if self.require_immutable_anchor:
                raise RuntimeError(
                    "immutable G0 anchor is required but not initialized; "
                    f"expected checkpoint {self.immutable_anchor_checkpoint!r}"
                )
            self._maybe_init_anchor()
            anc = self._anchor_transformer
        if anc is None:
            return None
        # Trainer construction precedes checkpoint restore and accelerator
        # preparation. The immutable reference intentionally stays outside the
        # module registry so resume cannot overwrite it, therefore move it on
        # demand once the live policy's final device is known.
        first = next(anc.parameters(), None)
        live_device = self._device()
        if first is not None and first.device != live_device:
            anc.to(live_device)
        anc.eval()
        return anc

    def _maybe_init_anchor(self) -> None:
        if self._immutable_g0_transformer is not None:
            return
        if self.require_immutable_anchor:
            raise RuntimeError(
                "lazy anchor initialization is disabled when require_immutable_anchor=True; "
                "call init_immutable_g0_anchor() after loading the live checkpoint"
            )
        if self._anchor_transformer is not None:
            return
        try:
            anc = deepcopy(self._core_transformer)
            anc.requires_grad_(False)
            anc.eval()
            object.__setattr__(self, "_anchor_transformer", anc)
        except Exception as exc:
            raise RuntimeError("failed to initialize legacy lazy anchor") from exc

    # ------------------------------------------------------------ conditioning

    # ---------------------------------------------------------------- sampling






    # -------------------------------------------------------------------- loss
