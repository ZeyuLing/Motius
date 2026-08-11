"""PhysFlowG1Bundle: the G1-native HyMotion flow-matching generator wrapped for
the PhysFlow online-adversarial closed loop.

It subclasses :class:`HyMotionT2MBundle` so warm-start / ``from_config`` /
``load_state_dict_selective`` / ``predict_flow`` / ``denormalize_motion`` /
``null_vtxt_feat`` are all inherited unchanged.  On top of that it exposes the
four atomic methods the PhysFlow trainer/reward path expects, but in the 38-d
G1 flow-matching space instead of the generic diffusion space:

  * ``sample_motion``   -- flow-matching ODE sampling from cached dual text
    embeddings (CLIP-L 768 ``vtxt`` + Qwen3 4096 ``ctxt``), returns NORMALIZED
    38-d motion (network space), no grad.
  * ``latents_to_qpos`` -- denormalize + :func:`decode_g1_to_qpos` -> MuJoCo
    qpos numpy ``[B, T, 36]`` (pos3 + quat_wxyz4 + 29 dof), exact, no SMPL
    retarget -- the same qpos layout the frozen judge consumes.
  * ``save_qpos_csv``   -- write qpos ``[T, 36]`` as a header-less, frame-column
    -less CSV, exactly what ``convert_g1_csv_to_proto.py`` parses
    (``--pos-units m --rot-format quat_wxyz --joint-units rad``).
  * ``sft_loss_g1``     -- reward-filtered flow-matching velocity loss toward
    the selected (trackable) sample, with an anchor MSE to an explicit immutable
    G0 checkpoint or the legacy first-use snapshot.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from motius.models.hymotion_t2m.bundle import HyMotionT2MBundle
from motius.models.gentrack.flow_grpo import (
    clipped_grpo_loss,
    flow_dpo_pair_loss,
    flow_grpo_transition,
    reverse_kl_from_log_probs,
    sample_unique_timestep_indices,
)
from motius.motion.representation.g1 import decode_g1_to_qpos
from motius.registry import MODEL_BUNDLES


def _len_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    return (torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
            < lengths.unsqueeze(1))


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
    def _pack_ctxt(self, text_ctxt: List[Tensor], ctxt_len: Tensor, device, dtype):
        """List of (seq_i, 4096) -> padded (B, max_seq, 4096) + bool mask."""
        B = len(text_ctxt)
        max_seq = max(int(c.shape[0]) for c in text_ctxt)
        ctxt = torch.zeros(B, max_seq, self._ctxt_input_dim, dtype=dtype, device=device)
        for i, c in enumerate(text_ctxt):
            ctxt[i, :c.shape[0]] = c.to(device, dtype)
        return ctxt, _len_to_mask(ctxt_len.to(device), max_seq)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_motion(
        self,
        text_vec: Tensor,            # (B, 1, 768)
        text_ctxt: List[Tensor],     # list of (seq_i, 4096)
        ctxt_len: Tensor,            # (B,)
        lengths: Tensor,             # (B,) int target frames
        num_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        initial_noise: Optional[Tensor] = None,
        transformer=None,
        return_initial_noise: bool = False,
    ) -> Tensor:
        """Flow-matching ODE -> NORMALIZED 38-d motion (B, Lmax, 38)."""
        device = self._device()
        dtype = torch.float32
        num_steps = int(num_steps or self.sample_steps)
        guidance = float(guidance if guidance is not None else self.sample_guidance)

        vtxt = text_vec.to(device, dtype)
        ctxt, ctxt_mask = self._pack_ctxt(text_ctxt, ctxt_len, device, dtype)
        lengths = lengths.to(device)
        L = int(lengths.max().item())
        Lp = max(L, self.TRAIN_FRAMES)
        B = vtxt.shape[0]
        core = transformer
        motion_dim = getattr(core, "output_dim", self._core_transformer.output_dim)
        x_mask = _len_to_mask(lengths, Lp)

        def predict(x_input, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal):
            if core is None:
                return self.predict_flow(
                    x_input=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                    timesteps=timesteps, x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal)
            return core(
                x=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                timesteps=timesteps, x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal, mask_density=None,
                task_emb=None)

        do_cfg = guidance > 1.0
        if do_cfg:
            null_vtxt = self.null_vtxt_feat.to(device, dtype).expand_as(vtxt)
            vtxt_cfg = torch.cat([null_vtxt, vtxt], 0)
            ctxt_cfg = torch.cat([ctxt, ctxt], 0)
            ctxt_mask_cfg = torch.cat([ctxt_mask, ctxt_mask], 0)
            x_mask_cfg = x_mask.repeat(2, 1)

        def fn(t_val, x):
            if do_cfg:
                xd = torch.cat([x, x], 0)
                xp = predict(
                    x_input=xd, ctxt_input=ctxt_cfg, vtxt_input=vtxt_cfg,
                    timesteps=t_val.expand(2 * B), x_mask_temporal=x_mask_cfg,
                    ctxt_mask_temporal=ctxt_mask_cfg)
                pu, pt = xp.chunk(2, 0)
                return pu + guidance * (pt - pu)
            return predict(
                x_input=x, ctxt_input=ctxt, vtxt_input=vtxt,
                timesteps=t_val.expand(B), x_mask_temporal=x_mask,
                ctxt_mask_temporal=ctxt_mask)

        if initial_noise is None:
            y0 = torch.randn(B, Lp, motion_dim, device=device, dtype=dtype)
        else:
            y0 = initial_noise.to(device=device, dtype=dtype)
            if y0.shape[0] != B or y0.shape[2] != motion_dim:
                raise ValueError(
                    f"initial_noise shape {tuple(y0.shape)} incompatible with "
                    f"B={B}, motion_dim={motion_dim}")
            if y0.shape[1] < Lp:
                y0 = F.pad(y0, (0, 0, 0, Lp - y0.shape[1]))
            elif y0.shape[1] > Lp:
                y0 = y0[:, :Lp]
        try:
            from torchdiffeq import odeint
            t = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
            sampled = odeint(fn, y0, t, method='euler')[-1]
        except ImportError:
            x = y0
            dt = 1.0 / num_steps
            for i in range(num_steps):
                x = x + fn(torch.tensor(i * dt, device=device, dtype=dtype), x) * dt
            sampled = x
        sampled = sampled[:, :L, :]  # normalized (B, L, 38)
        if return_initial_noise:
            return sampled, y0
        return sampled

    @torch.no_grad()
    def sample_motion_grpo(
        self,
        text_vec: Tensor,
        text_ctxt: List[Tensor],
        ctxt_len: Tensor,
        lengths: Tensor,
        num_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        eta: float = 0.7,
        initial_noise: Optional[Tensor] = None,
        transition_noises: Optional[Tensor] = None,
        transformer=None,
        return_policy_artifacts: bool = True,
    ) -> Dict[str, Tensor]:
        """Sample stochastic rectified-flow trajectories for Flow-GRPO.

        The returned states/log-probabilities are detached behavior-policy
        artifacts. ``grpo_loss_g1`` later replays sampled transitions under the
        live and frozen-reference policies. Passing the same ``initial_noise``
        and ``transition_noises`` to an immutable G0 transformer yields a
        paired common-random-numbers counterfactual.
        """
        device = self._device()
        dtype = torch.float32
        num_steps = int(num_steps or self.sample_steps)
        guidance = float(guidance if guidance is not None else self.sample_guidance)
        if guidance != 1.0:
            raise ValueError("Flow-GRPO currently requires guidance=1.0 for exact policy log-probs")
        if num_steps < 2:
            raise ValueError("Flow-GRPO requires at least two sampling steps")

        vtxt = text_vec.to(device, dtype)
        ctxt, ctxt_mask = self._pack_ctxt(text_ctxt, ctxt_len, device, dtype)
        lengths = lengths.to(device)
        length_max = int(lengths.max().item())
        padded_length = max(length_max, self.TRAIN_FRAMES)
        batch = vtxt.shape[0]
        motion_dim = self._core_transformer.output_dim
        frame_mask = _len_to_mask(lengths, padded_length)
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

        expected_state_shape = (batch, padded_length, motion_dim)
        if initial_noise is None:
            state = torch.randn(expected_state_shape, device=device, dtype=dtype)
        else:
            state = initial_noise.to(device=device, dtype=dtype)
            if state.shape != expected_state_shape:
                raise ValueError(
                    f"initial_noise shape {tuple(state.shape)} does not match "
                    f"{expected_state_shape}"
                )
        initial_state = state
        if transition_noises is not None:
            transition_noises = transition_noises.to(device=device, dtype=dtype)
            expected_noise_shape = (batch, num_steps, padded_length, motion_dim)
            if transition_noises.shape != expected_noise_shape:
                raise ValueError(
                    f"transition_noises shape {tuple(transition_noises.shape)} "
                    f"does not match {expected_noise_shape}"
                )

        states = [state] if return_policy_artifacts else []
        old_log_probs = []
        used_transition_noises = []
        for index in range(num_steps):
            sigma = sigmas[index]
            t = (1.0 - sigma).expand(batch)
            if transformer is None:
                velocity = self.predict_flow(
                    x_input=state,
                    ctxt_input=ctxt,
                    vtxt_input=vtxt,
                    timesteps=t,
                    x_mask_temporal=frame_mask,
                    ctxt_mask_temporal=ctxt_mask,
                )
            else:
                velocity = transformer(
                    x=state,
                    ctxt_input=ctxt,
                    vtxt_input=vtxt,
                    timesteps=t,
                    x_mask_temporal=frame_mask,
                    ctxt_mask_temporal=ctxt_mask,
                    mask_density=None,
                    task_emb=None,
                )
            step_noise = (
                transition_noises[:, index]
                if transition_noises is not None
                else torch.randn_like(state)
            )
            state, log_prob = flow_grpo_transition(
                model_output=-velocity,
                latents=state,
                sigma=sigma,
                sigma_next=sigmas[index + 1],
                sigma_first_next=sigmas[1],
                eta=eta,
                frame_mask=frame_mask,
                transition_noise=step_noise,
            )
            used_transition_noises.append(step_noise)
            if return_policy_artifacts:
                states.append(state)
                old_log_probs.append(log_prob)

        result = {
            "sample": state[:, :length_max].detach(),
            "initial_noise": initial_state.detach(),
            "transition_noises": torch.stack(used_transition_noises, dim=1).detach(),
            "sigmas": sigmas.detach(),
        }
        if return_policy_artifacts:
            result["states"] = torch.stack(states, dim=1).detach()
            result["old_log_probs"] = torch.stack(old_log_probs, dim=1).detach()
        return result

    def grpo_loss_g1(
        self,
        text_vec: Tensor,
        text_ctxt: List[Tensor],
        ctxt_len: Tensor,
        lengths: Tensor,
        states: Tensor,
        old_log_probs: Tensor,
        sigmas: Tensor,
        advantages: Tensor,
        eta: float = 0.7,
        clip_range: float = 0.2,
        reference_kl_weight: float = 0.01,
        timesteps_per_update: int = 1,
        transition_microbatch_size: int = 0,
        timestep_indices: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Replay one or more sampled transitions per candidate under the live policy."""
        device = self._device()
        dtype = torch.float32
        vtxt = text_vec.to(device, dtype)
        ctxt, ctxt_mask = self._pack_ctxt(text_ctxt, ctxt_len, device, dtype)
        lengths = lengths.to(device)
        states = states.to(device, dtype)
        old_log_probs = old_log_probs.to(device, dtype)
        sigmas = sigmas.to(device, dtype)
        advantages = advantages.to(device, dtype)
        batch, num_transitions = old_log_probs.shape
        if states.shape[:2] != (batch, num_transitions + 1):
            raise ValueError(
                f"states {tuple(states.shape)} incompatible with log-probs {tuple(old_log_probs.shape)}"
            )
        if advantages.shape != (batch,):
            raise ValueError(f"advantages must be ({batch},), got {tuple(advantages.shape)}")

        if timestep_indices is None:
            timestep_indices = sample_unique_timestep_indices(
                batch,
                num_transitions,
                timesteps_per_update,
                device,
            )
        else:
            timestep_indices = timestep_indices.to(device=device, dtype=torch.long)
            if timestep_indices.ndim == 1:
                timestep_indices = timestep_indices.unsqueeze(1)
            if timestep_indices.ndim != 2 or timestep_indices.shape[0] != batch:
                raise ValueError(
                    "timestep_indices must have shape [batch] or [batch, K], "
                    f"got {tuple(timestep_indices.shape)}"
                )
            if (
                timestep_indices.numel()
                and (
                    int(timestep_indices.min()) < 0
                    or int(timestep_indices.max()) >= num_transitions
                )
            ):
                raise ValueError(
                    f"timestep_indices must be in [0, {num_transitions})"
                )

        timestep_count = int(timestep_indices.shape[1])
        row = torch.arange(batch, device=device).unsqueeze(1).expand(
            batch,
            timestep_count,
        )
        current_state = states[row, timestep_indices].flatten(0, 1)
        next_state = states[row, timestep_indices + 1].flatten(0, 1)
        sigma = sigmas[timestep_indices].reshape(-1)
        sigma_next = sigmas[timestep_indices + 1].reshape(-1)
        lengths = lengths.repeat_interleave(timestep_count)
        frame_mask = _len_to_mask(lengths, states.shape[2])
        vtxt = vtxt.repeat_interleave(timestep_count, dim=0)
        ctxt = ctxt.repeat_interleave(timestep_count, dim=0)
        ctxt_mask = ctxt_mask.repeat_interleave(timestep_count, dim=0)
        advantages = advantages.repeat_interleave(timestep_count)
        t = 1.0 - sigma

        transition_microbatch_size = int(transition_microbatch_size)
        if transition_microbatch_size < 0:
            raise ValueError("transition_microbatch_size must be non-negative")

        def predict_live(
            state_chunk: Tensor,
            ctxt_chunk: Tensor,
            vtxt_chunk: Tensor,
            timestep_chunk: Tensor,
            frame_mask_chunk: Tensor,
            ctxt_mask_chunk: Tensor,
        ) -> Tensor:
            return self.predict_flow(
                x_input=state_chunk,
                ctxt_input=ctxt_chunk,
                vtxt_input=vtxt_chunk,
                timesteps=timestep_chunk,
                x_mask_temporal=frame_mask_chunk,
                ctxt_mask_temporal=ctxt_mask_chunk,
            )

        if 0 < transition_microbatch_size < current_state.shape[0]:
            velocity_chunks = []
            for start in range(0, current_state.shape[0], transition_microbatch_size):
                stop = min(start + transition_microbatch_size, current_state.shape[0])
                # Checkpointing is required here: ordinary chunked forwards keep
                # every chunk's activations alive until the shared loss backward.
                velocity_chunks.append(
                    checkpoint(
                        predict_live,
                        current_state[start:stop],
                        ctxt[start:stop],
                        vtxt[start:stop],
                        t[start:stop],
                        frame_mask[start:stop],
                        ctxt_mask[start:stop],
                        use_reentrant=False,
                    )
                )
            velocity = torch.cat(velocity_chunks, dim=0)
        else:
            velocity = predict_live(
                current_state,
                ctxt,
                vtxt,
                t,
                frame_mask,
                ctxt_mask,
            )
        _, current_log_prob = flow_grpo_transition(
            model_output=-velocity,
            latents=current_state,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_first_next=sigmas[1],
            eta=eta,
            next_latents=next_state,
            frame_mask=frame_mask,
        )
        behavior_log_prob = old_log_probs[row, timestep_indices].reshape(-1)
        log_ratio = current_log_prob - behavior_log_prob.detach()
        policy_loss, ratio_mean, clip_fraction = clipped_grpo_loss(
            current_log_prob,
            behavior_log_prob,
            advantages,
            clip_range=clip_range,
        )
        detached_advantages = advantages.detach()
        detached_log_ratio = log_ratio.detach()
        advantage_log_ratio_cov = (
            (detached_advantages - detached_advantages.mean())
            * (detached_log_ratio - detached_log_ratio.mean())
        ).mean()
        advantage_log_ratio_alignment = (
            detached_advantages.sign() * detached_log_ratio
        ).mean()
        positive_mask = detached_advantages > 0
        negative_mask = detached_advantages < 0
        positive_log_ratio_mean = (
            detached_log_ratio[positive_mask].mean()
            if bool(positive_mask.any())
            else detached_log_ratio.new_zeros(())
        )
        negative_log_ratio_mean = (
            detached_log_ratio[negative_mask].mean()
            if bool(negative_mask.any())
            else detached_log_ratio.new_zeros(())
        )

        reference_kl = current_log_prob.new_zeros(())
        if reference_kl_weight > 0:
            anchor = self._reference_transformer()
            if anchor is None:
                if self.require_immutable_anchor:
                    raise RuntimeError(
                        "reference_kl_weight > 0 but immutable G0 anchor is unavailable"
                    )
            else:
                with torch.no_grad():
                    if 0 < transition_microbatch_size < current_state.shape[0]:
                        reference_velocity = torch.cat(
                            [
                                anchor(
                                    x=current_state[start:stop],
                                    ctxt_input=ctxt[start:stop],
                                    vtxt_input=vtxt[start:stop],
                                    timesteps=t[start:stop],
                                    x_mask_temporal=frame_mask[start:stop],
                                    ctxt_mask_temporal=ctxt_mask[start:stop],
                                    mask_density=None,
                                    task_emb=None,
                                )
                                for start, stop in (
                                    (
                                        start,
                                        min(
                                            start + transition_microbatch_size,
                                            current_state.shape[0],
                                        ),
                                    )
                                    for start in range(
                                        0,
                                        current_state.shape[0],
                                        transition_microbatch_size,
                                    )
                                )
                            ],
                            dim=0,
                        )
                    else:
                        reference_velocity = anchor(
                            x=current_state,
                            ctxt_input=ctxt,
                            vtxt_input=vtxt,
                            timesteps=t,
                            x_mask_temporal=frame_mask,
                            ctxt_mask_temporal=ctxt_mask,
                            mask_density=None,
                            task_emb=None,
                        )
                    _, reference_log_prob = flow_grpo_transition(
                        model_output=-reference_velocity,
                        latents=current_state,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        sigma_first_next=sigmas[1],
                        eta=eta,
                        next_latents=next_state,
                        frame_mask=frame_mask,
                    )
                reference_kl = reverse_kl_from_log_probs(current_log_prob, reference_log_prob)

        loss = policy_loss + float(reference_kl_weight) * reference_kl
        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "reference_kl": reference_kl.detach(),
            "ratio_mean": ratio_mean,
            "ratio_abs_deviation_mean": (log_ratio.exp() - 1.0).abs().mean().detach(),
            "log_ratio_abs_mean": log_ratio.abs().mean().detach(),
            "advantage_log_ratio_cov": advantage_log_ratio_cov,
            "advantage_log_ratio_alignment": advantage_log_ratio_alignment,
            "positive_advantage_log_ratio_mean": positive_log_ratio_mean,
            "negative_advantage_log_ratio_mean": negative_log_ratio_mean,
            "clip_fraction": clip_fraction,
            "current_log_prob": current_log_prob.detach().mean(),
            "behavior_log_prob": behavior_log_prob.detach().mean(),
            "timestep_mean": timestep_indices.float().mean().detach(),
            "timesteps_per_update": current_log_prob.new_tensor(
                float(timestep_count)
            ),
            "transition_microbatch_size": current_log_prob.new_tensor(
                float(transition_microbatch_size)
            ),
        }

    def flow_dpo_loss_g1(
        self,
        text_vec: Tensor,
        text_ctxt: List[Tensor],
        ctxt_len: Tensor,
        winner_motion: Tensor,
        loser_motion: Tensor,
        lengths: Tensor,
        *,
        beta: float = 100.0,
        timesteps_per_pair: int = 1,
    ) -> Dict[str, Tensor]:
        """Apply Flow-DPO to same-prompt physical-preference motion pairs.

        Winner and loser receive the same flow noise and timestep. The frozen
        immutable G0 predicts the same noised pair, making the objective the
        established Diffusion-DPO error difference rather than an ad-hoc
        reward-weighted motion loss.
        """
        if self.pred_type != "velocity":
            raise ValueError("Flow-DPO currently requires velocity prediction")

        device = self._device()
        dtype = torch.float32
        winner = winner_motion.to(device=device, dtype=dtype).detach()
        loser = loser_motion.to(device=device, dtype=dtype).detach()
        if winner.shape != loser.shape or winner.ndim != 3:
            raise ValueError(
                "Flow-DPO winner and loser must have matching (B,T,D) shapes, "
                f"got {tuple(winner.shape)} and {tuple(loser.shape)}"
            )
        batch, frames, motion_dim = winner.shape
        lengths = lengths.to(device=device, dtype=torch.long)
        if lengths.shape != (batch,):
            raise ValueError(
                f"Flow-DPO lengths must have shape {(batch,)}, got {tuple(lengths.shape)}"
            )
        timesteps_per_pair = int(timesteps_per_pair)
        if timesteps_per_pair < 1:
            raise ValueError("timesteps_per_pair must be positive")

        vtxt = text_vec.to(device=device, dtype=dtype)
        ctxt, ctxt_mask = self._pack_ctxt(
            text_ctxt,
            ctxt_len,
            device,
            dtype,
        )
        if vtxt.shape[0] != batch or ctxt.shape[0] != batch:
            raise ValueError("Flow-DPO requires one text condition per motion pair")

        winner = winner.repeat_interleave(timesteps_per_pair, dim=0)
        loser = loser.repeat_interleave(timesteps_per_pair, dim=0)
        lengths = lengths.repeat_interleave(timesteps_per_pair, dim=0)
        vtxt = vtxt.repeat_interleave(timesteps_per_pair, dim=0)
        ctxt = ctxt.repeat_interleave(timesteps_per_pair, dim=0)
        ctxt_mask = ctxt_mask.repeat_interleave(timesteps_per_pair, dim=0)
        effective_batch = batch * timesteps_per_pair

        pair_noise = torch.randn_like(winner)
        pair_timestep = torch.rand(
            effective_batch,
            device=device,
            dtype=dtype,
        )
        x1 = torch.cat((winner, loser), dim=0)
        x0 = torch.cat((pair_noise, pair_noise), dim=0)
        timesteps = torch.cat((pair_timestep, pair_timestep), dim=0)
        t = timesteps.view(2 * effective_batch, 1, 1)
        x_t = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0

        frame_mask = _len_to_mask(lengths, frames)
        pair_frame_mask = torch.cat((frame_mask, frame_mask), dim=0)
        pair_ctxt = torch.cat((ctxt, ctxt), dim=0)
        pair_vtxt = torch.cat((vtxt, vtxt), dim=0)
        pair_ctxt_mask = torch.cat((ctxt_mask, ctxt_mask), dim=0)

        prediction = self.predict_flow(
            x_input=x_t,
            ctxt_input=pair_ctxt,
            vtxt_input=pair_vtxt,
            timesteps=timesteps,
            x_mask_temporal=pair_frame_mask,
            ctxt_mask_temporal=pair_ctxt_mask,
        )
        reference = self._reference_transformer()
        if reference is None:
            raise RuntimeError("Flow-DPO requires an immutable G0 reference")
        with torch.no_grad():
            reference_prediction = reference(
                x=x_t,
                ctxt_input=pair_ctxt,
                vtxt_input=pair_vtxt,
                timesteps=timesteps,
                x_mask_temporal=pair_frame_mask,
                ctxt_mask_temporal=pair_ctxt_mask,
                mask_density=None,
                task_emb=None,
            )

        mask = pair_frame_mask.unsqueeze(-1).to(dtype)
        denominator = (
            mask.sum(dim=(1, 2)) * float(motion_dim)
        ).clamp_min(1.0)
        model_mse = (
            ((prediction.float() - target_velocity.float()).square() * mask)
            .sum(dim=(1, 2))
            / denominator
        )
        reference_mse = (
            (
                (reference_prediction.float() - target_velocity.float()).square()
                * mask
            )
            .sum(dim=(1, 2))
            / denominator
        )
        model_winner_mse, model_loser_mse = model_mse.chunk(2)
        reference_winner_mse, reference_loser_mse = reference_mse.chunk(2)
        output = flow_dpo_pair_loss(
            model_winner_mse,
            model_loser_mse,
            reference_winner_mse,
            reference_loser_mse,
            beta=beta,
        )
        output.update(
            {
                "model_winner_mse": model_winner_mse.detach().mean(),
                "model_loser_mse": model_loser_mse.detach().mean(),
                "reference_winner_mse": reference_winner_mse.detach().mean(),
                "reference_loser_mse": reference_loser_mse.detach().mean(),
                "beta": model_mse.new_tensor(float(beta)),
                "timestep_mean": pair_timestep.detach().mean(),
                "timestep_std": pair_timestep.detach().std(unbiased=False),
                "timesteps_per_pair": model_mse.new_tensor(
                    float(timesteps_per_pair)
                ),
            }
        )
        return output

    @torch.no_grad()
    def latents_to_qpos(self, latent: Tensor) -> np.ndarray:
        """Normalized 38-d motion (B, L, 38) -> qpos numpy (B, L, 36)."""
        denorm = self.denormalize_motion(latent.to(self._device()).float())  # (B,L,38)
        qpos = []
        for b in range(denorm.shape[0]):
            qpos.append(decode_g1_to_qpos(denorm[b].cpu()).numpy())
        return np.stack(qpos, axis=0)

    @staticmethod
    def save_qpos_csv(qpos_sample: np.ndarray, csv_path: str) -> None:
        """qpos (T, 36) -> header-less, frame-column-less CSV for the converter."""
        arr = np.asarray(qpos_sample, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None]
        np.savetxt(csv_path, arr, delimiter=",")

    # -------------------------------------------------------------------- loss
    def sft_loss_g1(
        self,
        text_vec: Tensor,            # (B, 1, 768)
        text_ctxt: List[Tensor],     # list of (seq_i, 4096)
        ctxt_len: Tensor,            # (B,)
        target_motion: Tensor,       # (B, L, 38) NORMALIZED selected sample (detached)
        lengths: Tensor,             # (B,)
        good_mask: Optional[Tensor] = None,    # (B,) {0,1}
        sample_weights: Optional[Tensor] = None,  # (B,) reward weights
        anchor_weight: float = 0.0,
        gt_target: Optional[Tensor] = None,    # (B, Lg, 38) NORMALIZED ground-truth motion
        gt_lengths: Optional[Tensor] = None,   # (B,)
        gt_weight: float = 0.0,                # weight of the GT supervised term
    ) -> Dict[str, Tensor]:
        """Reward-filtered flow-matching velocity SFT toward the selected sample.

        x0 ~ N(0, I); t ~ U(0, 1); x_t = (1-t) x0 + t x1; the FM target velocity
        is (x1 - x0).  Loss = ||predict_flow(x_t, t) - (x1 - x0)||^2, masked to
        valid frames, reward-filtered by ``good_mask`` (rejected prompts give 0
        SFT gradient), plus optional anchor MSE to the frozen base generator.

        When ``gt_target`` / ``gt_weight`` are given a ground-truth supervised FM
        term is added.  Both the reward target and the GT target share the SAME
        text conditioning (the batch prompts), so they are concatenated along the
        batch dim and pushed through **one** ``predict_flow`` forward -- this is
        deliberate: under DDP a second grad-forward on the wrapped transformer
        would trip the reducer ("mark ready twice").  The total objective is
        ``reward_sft + gt_weight * gt_sft + anchor_weight * anchor`` (a sum of
        per-group means, not a blended average).
        """
        device = self._device()
        dtype = torch.float32
        vtxt = text_vec.to(device, dtype)
        ctxt, ctxt_mask = self._pack_ctxt(text_ctxt, ctxt_len, device, dtype)
        x1 = target_motion.to(device, dtype).detach()
        lengths = lengths.to(device)
        B, Lr, D = x1.shape

        use_gt = gt_target is not None and gt_weight and gt_weight > 0
        Lg = int(gt_target.shape[1]) if use_gt else 0
        L = max(Lr, Lg)
        if Lr < L:
            x1 = F.pad(x1, (0, 0, 0, L - Lr))
        x_mask = _len_to_mask(lengths, L)

        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device, dtype=dtype)
        x_t = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        v_target = x1 - x0

        if use_gt:
            x1g = gt_target.to(device, dtype).detach()
            gt_lengths = gt_lengths.to(device)
            if Lg < L:
                x1g = F.pad(x1g, (0, 0, 0, L - Lg))
            x_mask_g = _len_to_mask(gt_lengths, L)
            x0g = torch.randn_like(x1g)
            tg = torch.rand(B, device=device, dtype=dtype)
            x_tg = (1.0 - tg.view(B, 1, 1)) * x0g + tg.view(B, 1, 1) * x1g
            v_target_g = x1g - x0g

            # one combined forward over [reward; gt] (DDP-safe single grad-forward)
            pred_cat = self.predict_flow(
                x_input=torch.cat([x_t, x_tg], dim=0),
                ctxt_input=torch.cat([ctxt, ctxt], dim=0),
                vtxt_input=torch.cat([vtxt, vtxt], dim=0),
                timesteps=torch.cat([t, tg], dim=0),
                x_mask_temporal=torch.cat([x_mask, x_mask_g], dim=0),
                ctxt_mask_temporal=torch.cat([ctxt_mask, ctxt_mask], dim=0),
            )
            pred_v, pred_v_g = pred_cat[:B], pred_cat[B:]
        else:
            pred_v = self.predict_flow(
                x_input=x_t, ctxt_input=ctxt, vtxt_input=vtxt,
                timesteps=t, x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask,
            )

        frame_mask = x_mask.unsqueeze(-1).to(dtype)            # (B, L, 1)
        err = (pred_v - v_target) ** 2 * frame_mask
        denom = (frame_mask.sum(dim=(1, 2)) * D).clamp_min(1.0)
        per_sample = err.sum(dim=(1, 2)) / denom               # (B,)

        gm = (good_mask.to(device, dtype) if good_mask is not None
              else torch.ones_like(per_sample))
        w = (sample_weights.to(device, dtype) * gm if sample_weights is not None else gm)
        wsum = w.sum()
        sft = (per_sample * w).sum() / wsum.clamp_min(1e-8) if float(wsum) > 0 \
            else (per_sample * 0.0).sum()

        n_good = gm.sum()
        out: Dict[str, Tensor] = {
            "sft_mse": ((per_sample * gm).sum() / n_good.clamp_min(1.0)).detach(),
            "n_good": n_good.detach(),
        }
        loss = sft

        if use_gt:
            fmg = x_mask_g.unsqueeze(-1).to(dtype)
            errg = (pred_v_g - v_target_g) ** 2 * fmg
            denomg = (fmg.sum(dim=(1, 2)) * D).clamp_min(1.0)
            gt_sft = (errg.sum(dim=(1, 2)) / denomg).mean()
            loss = loss + float(gt_weight) * gt_sft
            out["gt_mse"] = gt_sft.detach()

        if anchor_weight and anchor_weight > 0:
            anc = self._reference_transformer()
            if anc is None:
                if self.require_immutable_anchor:
                    raise RuntimeError(
                        "anchor_weight > 0 but immutable G0 anchor is unavailable"
                    )
            else:
                with torch.no_grad():
                    base_v = anc(
                        x=x_t, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t,
                        x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask,
                        mask_density=None, task_emb=None,
                    )
                anchor_err = (pred_v - base_v) ** 2 * frame_mask
                anchor = (anchor_err.sum(dim=(1, 2)) / denom).mean()
                loss = loss + float(anchor_weight) * anchor
                out["anchor_mse"] = anchor.detach()

        out["loss"] = loss
        return out
