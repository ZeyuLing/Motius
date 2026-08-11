"""MotionCanvas Pipeline: ODE-based inference with clean imputation.

The trained conditioning protocol fixes known coordinates to clean evidence:
``x_t[known] = x1[known]``. Inference mirrors it by replacing known
coordinates with ``clean_motion`` at each ODE step.

Replacement guidance modes
--------------------------
- ``"none"``: Diagnostic ablation without per-step replacement.
- ``"all"``: At every ODE step, replace known regions with ``clean_motion``.
- ``"skip_last"``: Same as ``"all"`` but skip replacement on the final step.

When ``replacement_guidance != 'none'``, the batch **must** contain a
``clean_motion`` key: the full normalized motion ``(B, T, D)`` **without**
masked regions zeroed.  The initial noise ``y0`` is also set to
``clean_motion`` in known regions (matching training where ``x_t[known] = x1``).

Position constraint support
---------------------------
When ``position_constraints`` is provided in the batch, the pipeline applies
IK projection at each ODE step (after imputation) to enforce world-space
position constraints on specified joints. See
:class:`~motius.motion.pipeline_utils.position_constraint.PositionConstraint`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.motion.pipeline_utils.repair_utils import (
    compute_ada_keep_mask,
    compute_strict_adaptive_mask,
    joint_mask_to_dim_mask,
)


def normalize_repair_seed(seed: int) -> tuple[int, int]:
    """Return deterministic Torch/NumPy seeds for one repair sample.

    Torch accepts a wider integer range than NumPy's legacy global RNG. Keep
    the caller-visible seed intact for Torch while mapping it into NumPy's
    uint32 domain. Reject negative seeds so ``base + sample_index`` cannot
    silently acquire backend-specific wrap semantics.
    """
    torch_seed = int(seed)
    if torch_seed < 0:
        raise ValueError(f"repair_seed must be non-negative, got {torch_seed}")
    return torch_seed, torch_seed % (2**32)


def reset_repair_rng(seed: int) -> None:
    """Reset every RNG used by the repair pass at the Step-E boundary."""
    torch_seed, numpy_seed = normalize_repair_seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def _sdedit_integration_times(
    base_times: Tensor,
    t_init: Optional[float],
    *,
    exact_start: bool,
) -> Tensor:
    """Return the active ODE grid for a partial-noise SDEdit start."""
    if base_times.ndim != 1 or len(base_times) < 2:
        raise ValueError("base_times must be a one-dimensional ODE grid")
    if t_init is None:
        return base_times
    start = float(t_init)
    if not 0.0 <= start < 1.0:
        raise ValueError(f"SDEdit t_init must lie in [0,1), got {start}")
    if exact_start:
        start_t = torch.as_tensor(
            start, device=base_times.device, dtype=base_times.dtype
        ).reshape(1)
        future_t = base_times[base_times > start_t[0] + 1e-7]
        active = torch.cat([start_t, future_t], dim=0)
    else:
        values = base_times.detach().cpu().tolist()
        start_index = next(
            (index for index, value in enumerate(values) if value + 1e-6 >= start),
            0,
        )
        active = base_times[start_index:]
    if len(active) < 1 or (
        len(active) > 1 and not bool(torch.all(active[1:] > active[:-1]))
    ):
        raise RuntimeError("SDEdit integration schedule is invalid")
    if exact_start and len(active) < 2:
        raise RuntimeError("exact SDEdit schedule has no positive-time step")
    return active


def _base_grid_step_index(base_times: Tensor, active_time: Tensor) -> int:
    """Map an active SDEdit step back to the legacy base-grid phase.

    A snapped partial start is a suffix of ``base_times``. Its first active
    loop index is zero, but position-IK cadence historically used the original
    global base-grid index. Exact inserted detector substeps map to the interval
    immediately below them; detector probes currently carry no position IK.
    """
    if base_times.ndim != 1 or len(base_times) < 2:
        raise ValueError("base_times must be a one-dimensional ODE grid")
    value = torch.as_tensor(
        active_time, device=base_times.device, dtype=base_times.dtype
    ).reshape(())
    index = int(torch.searchsorted(base_times, value, right=True).item() - 1)
    if index < 0 or index >= len(base_times) - 1:
        raise ValueError(
            f"active ODE time must lie in [base_times[0], base_times[-1]): "
            f"got {float(value)}"
        )
    return index


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _batch_sources(batch: Dict[str, Any], batch_size: int) -> Optional[List[str]]:
    sources = batch.get('data_src', batch.get('source'))
    if sources is None:
        return None
    if isinstance(sources, str):
        return [sources] * batch_size
    if isinstance(sources, Tensor):
        return None
    if isinstance(sources, (list, tuple)):
        if len(sources) != batch_size:
            return None
        return [str(s) for s in sources]
    return None


def _length_list(value: Any, batch_size: int, fallback: int) -> List[int]:
    if value is None:
        return [fallback] * batch_size
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, int):
        return [value] * batch_size
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and batch_size > 1:
            return [int(value[0])] * batch_size
        return [int(v) for v in value]
    return [fallback] * batch_size


def _gaussian_temporal_smooth(
    x: Tensor,
    sigma: float,
    protect_mask: Optional[Tensor] = None,
) -> Tensor:
    """1-D Gaussian temporal smoothing along axis 1 of a (B, T, D) tensor.

    Ported from ``scripts/eval/eval_m2m_v2_all_tasks._gaussian_temporal_smooth``.
    Used to pre-smooth the LQ ``clean_motion`` (the *kept* region the model
    conditions on, and which is copied back into the output) before masked
    imputation. Because partial regeneration keeps the jittery LQ on unmasked
    cells, smoothing there both lowers the residual jitter in the output and
    gives the regenerated region a smooth boundary to blend against.

    Where ``protect_mask > 0.5`` (the *generate* region) values pass through
    unchanged -- smoothing there is pointless (overwritten by the model) and
    would bleed bad values across defect boundaries.
    """
    if sigma <= 0.0:
        return x
    T = x.shape[1]
    radius = max(1, int(round(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    B, _, D = x.shape
    x_flat = x.permute(0, 2, 1).reshape(B * D, 1, T)
    w = kernel.view(1, 1, -1)
    x_pad = F.pad(x_flat, (radius, radius), mode='replicate')
    y_flat = F.conv1d(x_pad, w)
    y = y_flat.reshape(B, D, T).permute(0, 2, 1).contiguous()
    if protect_mask is not None:
        y = torch.where(protect_mask > 0.5, x, y)
    return y


@PIPELINES.register_module(force=True)
class MotionCanvasPipeline(BasePipeline):
    """Inference pipeline for MotionCanvas.

    Uses ODE integration to solve the flow matching ODE from noise to clean
    motion, conditioned on imputed motion evidence, edit context, and optional text.

    Parameters
    ----------
    bundle : MotionCanvasBundle
        The model bundle.
    num_steps : int
        Number of ODE integration steps.
    text_guidance_scale : float
        Classifier-free guidance scale for text conditioning.
    replacement_guidance : str
        Controls per-step replacement of unmasked (known) regions during
        ODE integration. This implements train-consistent imputation: during training,
        ``x_t[known] = x1`` (clean), so replacing known regions with clean
        motion at each step matches the training distribution.

        - ``"none"``: Diagnostic ablation without replacement.
        - ``"all"``: At every ODE step, replace known regions with
          ``clean_motion`` from the batch.
        - ``"skip_last"``: Same as ``"all"`` but skip replacement on the
          final step.

        When not ``"none"``, the batch must contain ``clean_motion``:
        the full normalized motion (B, T, D) without masked-region zeroing.
    """

    VALID_REPLACEMENT_MODES = ('none', 'all', 'skip_last', 'flow_interp')
    BUNDLE_CLS = 'motius.models.motioncanvas.MotionCanvasBundle'

    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 1.0,
        replacement_guidance: str = 'all',
        position_constraint_interval: int = 5,
        max_text_len: int = 128,
        sdedit_tau: float = 0.0,
        official_t2m_frames: int = 360,
    ):
        super().__init__(bundle)
        if replacement_guidance not in self.VALID_REPLACEMENT_MODES:
            raise ValueError(
                f'replacement_guidance must be one of '
                f'{self.VALID_REPLACEMENT_MODES}, got {replacement_guidance!r}'
            )
        self.bundle = bundle
        self.num_steps = num_steps
        self.text_guidance_scale = text_guidance_scale
        self.replacement_guidance = replacement_guidance
        self.position_constraint_interval = position_constraint_interval
        # IMPORTANT: must match the trainer's max_text_len (default 128) so
        # the context attention mask and positional statistics at inference
        # match what the model saw during training. Using the raw per-sample
        # token length (12-20) instead of padding to 128 was a bug that made
        # captioned inference produce distorted outputs (2026-04-20).
        self.max_text_len = max_text_len
        self.official_t2m_frames = int(official_t2m_frames)
        # SDEdit-style partial-noise start for E9 motion repair. In
        # flow-matching convention (t=0 -> pure noise, t=1 -> clean data),
        # the default inpainting path starts from t=0 on the masked region
        # (full regeneration). SDEdit τ lets us start from t = 1 - τ instead
        # — i.e. the masked region is initialized as `(1-τ)*x_clean + τ*z`
        # and the ODE runs from t=1-τ up to t=1. Smaller τ → more LQ retained,
        # closer to a "cleanup" of defects; larger τ (→1) → full regeneration.
        # Only applied when replacement_guidance != 'none' (requires
        # `clean_motion` in the batch to know what the masked region's LQ is).
        if not (0.0 <= sdedit_tau <= 1.0):
            raise ValueError(
                f'sdedit_tau must be in [0, 1], got {sdedit_tau!r}'
            )
        self.sdedit_tau = float(sdedit_tau)

    @staticmethod
    def _as_motion_batch(motion: Tensor, name: str) -> Tensor:
        motion = torch.as_tensor(motion)
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if motion.ndim != 3 or motion.shape[-1] != 198:
            raise ValueError(
                f'{name} must have shape (T, 198) or (B, T, 198), '
                f'got {tuple(motion.shape)}'
            )
        return motion.float()

    @staticmethod
    def _as_generation_mask(mask: Tensor, motion: Tensor) -> Tensor:
        mask = torch.as_tensor(mask, dtype=motion.dtype, device=motion.device)
        if mask.ndim == 1:
            mask = mask[None, :, None]
        elif mask.ndim == 2:
            if mask.shape == motion.shape[:2]:
                mask = mask[..., None]
            elif motion.shape[0] == 1 and mask.shape[0] == motion.shape[1]:
                mask = mask[None]
        if mask.ndim != 3:
            raise ValueError(
                'generation_mask must be frame-level (T)/(B,T) or '
                'channel-level (T,198)/(B,T,198)'
            )
        if mask.shape[0] == 1 and motion.shape[0] > 1:
            mask = mask.expand(motion.shape[0], -1, -1)
        if mask.shape[:2] != motion.shape[:2]:
            raise ValueError(
                f'generation_mask leading shape {tuple(mask.shape[:2])} '
                f'does not match motion {tuple(motion.shape[:2])}'
            )
        if mask.shape[-1] == 1:
            mask = mask.expand_as(motion)
        if mask.shape[-1] != motion.shape[-1]:
            raise ValueError(
                f'generation_mask must end in 1 or 198, got {mask.shape[-1]}'
            )
        return mask.clamp(0.0, 1.0)

    @torch.no_grad()
    def infer_m2m(
        self,
        source_motion: Optional[Tensor] = None,
        generation_mask: Optional[Tensor] = None,
        *,
        captions: Optional[List[str] | str] = None,
        num_frames: Optional[List[int] | int] = None,
        edit_mode: bool = False,
        edit_source_motion: Optional[Tensor] = None,
        input_is_normalized: bool = False,
        seed: Optional[int] = None,
        **batch_overrides,
    ) -> Dict[str, Any]:
        """Run the unified MotionCanvas generation/editing contract.

        ``generation_mask`` follows the training convention: 1 regenerates a
        coordinate and 0 preserves clean source evidence. The native motion is
        the 198-D tensor ``[root translation, local rotation-6D, FK joints]``.
        """
        device = next(self.bundle.motion_transformer.parameters()).device
        dtype = next(self.bundle.motion_transformer.parameters()).dtype

        if source_motion is None:
            caption_count = (
                1
                if captions is None or isinstance(captions, str)
                else len(captions)
            )
            if num_frames is None:
                num_frames = 180
            lengths = (
                [int(num_frames)] * caption_count
                if isinstance(num_frames, int)
                else [int(value) for value in num_frames]
            )
            source = torch.zeros(
                len(lengths),
                max(lengths),
                198,
                device=device,
                dtype=dtype,
            )
            clean_motion = source.clone()
            mask = torch.ones_like(source)
        else:
            source_raw = self._as_motion_batch(source_motion, 'source_motion').to(
                device=device,
                dtype=dtype,
            )
            lengths = (
                [source_raw.shape[1]] * source_raw.shape[0]
                if num_frames is None
                else (
                    [int(num_frames)] * source_raw.shape[0]
                    if isinstance(num_frames, int)
                    else [int(value) for value in num_frames]
                )
            )
            if len(lengths) != source_raw.shape[0]:
                raise ValueError('num_frames must contain one value per motion')
            clean_motion = (
                source_raw
                if input_is_normalized
                else self.bundle.normalize_motion(source_raw)
            )
            mask = (
                torch.ones_like(clean_motion)
                if generation_mask is None
                else self._as_generation_mask(generation_mask, clean_motion)
            )
            if edit_source_motion is not None:
                edit_source = self._as_motion_batch(
                    edit_source_motion,
                    'edit_source_motion',
                ).to(device=device, dtype=dtype)
                if edit_source.shape != source_raw.shape:
                    raise ValueError('edit_source_motion must match source_motion')
                edit_source = (
                    edit_source
                    if input_is_normalized
                    else self.bundle.normalize_motion(edit_source)
                )
            else:
                edit_source = clean_motion
            source = (
                clean_motion * (1.0 - mask)
                if not edit_mode
                else clean_motion * (1.0 - mask) + edit_source * mask
            )

        batch = {
            'src_motion': source,
            'src_mask': mask,
            'clean_motion': clean_motion,
            'src_length': lengths,
            'tgt_length': lengths,
            **batch_overrides,
        }
        if captions is not None:
            caption_list = [captions] if isinstance(captions, str) else list(captions)
            if len(caption_list) == 1 and len(lengths) > 1:
                caption_list *= len(lengths)
            if len(caption_list) != len(lengths):
                raise ValueError('captions must contain one string per motion')
            batch.update(self.bundle.encode_text(caption_list))
            batch['caption'] = caption_list

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        result = self(batch)
        max_length = max(lengths)
        for key in (
            'latent_denorm',
            'keypoints3d',
            'rot6d',
            'transl',
            'root_rotations_mat',
            'latent',
        ):
            value = result.get(key)
            if isinstance(value, Tensor) and value.ndim >= 2:
                result[key] = value[:, :max_length]
        result['motion_198'] = result['latent_denorm']
        result['lengths'] = lengths
        return result

    def infer_text_to_motion(
        self,
        captions: List[str] | str,
        num_frames: List[int] | int = 180,
        **kwargs,
    ) -> Dict[str, Any]:
        return self.infer_m2m(
            captions=captions,
            num_frames=num_frames,
            **kwargs,
        )

    infer_t2m = infer_text_to_motion

    def infer_temporal_motion_completion(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_motion_inbetweening(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_keyframe_motion_control(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_kinematic_motion_control(
        self,
        source_motion,
        generation_mask,
        *,
        position_constraints=None,
        **kwargs,
    ):
        if position_constraints is not None:
            kwargs['position_constraints'] = position_constraints
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_motion_editing(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=True, **kwargs)

    def infer_motion_repair(self, motion: Tensor, **kwargs) -> Dict[str, Any]:
        return self.infer_repair(motion, **kwargs)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a batch.

        Uses midpoint ODE solver for numerical stability (euler diverges).
        """
        return self._inference(batch)

    def _inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Actual inference logic.

        Returns:
            Dict with keys: rot6d, transl, keypoints3d (optional), latent.
        """
        device = next(self.bundle.motion_transformer.parameters()).device

        src_motion = batch['src_motion'].to(device)
        B, T, D = src_motion.shape

        src_mask = batch.get('src_mask')
        if src_mask is not None:
            src_mask = src_mask.to(device)

        src_length = batch.get('src_length')
        src_length = _length_list(src_length, B, T)

        tgt_length = batch.get('tgt_length', src_length)
        tgt_length = _length_list(tgt_length, B, max(src_length) if src_length else T)

        # Every official single-segment MotionCanvas call runs the ODE on the
        # same 360-frame canvas used by training, then lets the caller crop the
        # decoded result to ``tgt_length``.  This is not specific to text-only
        # full regeneration: conditional completion/editing on a short raw
        # canvas changes the transformer's temporal statistics and invalidates
        # cross-setting comparisons, including checkpoint-matched ablations.
        #
        # Keep the valid lengths unchanged so ``tgt_padding_mask`` marks only
        # the requested frames.  Pad normalized motion, condition mask, and
        # clean imputation target with training-parity zeros.  Padding mask
        # entries must be zero (not "generate"); valid-frame filtering below
        # prevents them from activating replacement guidance.
        use_official_canvas = (
            T < self.official_t2m_frames
            and max(tgt_length) <= self.official_t2m_frames
        )
        if use_official_canvas:
            pad_t = self.official_t2m_frames - T
            src_motion = F.pad(src_motion, (0, 0, 0, pad_t))
            if src_mask is not None:
                src_mask = F.pad(src_mask, (0, 0, 0, pad_t), value=0.0)
            if 'clean_motion' in batch and isinstance(batch['clean_motion'], Tensor):
                batch = dict(batch)
                batch['clean_motion'] = F.pad(
                    batch['clean_motion'].to(device),
                    (0, 0, 0, pad_t),
                )
            if (
                'completion_unanchored_mask' in batch
                and isinstance(batch['completion_unanchored_mask'], Tensor)
            ):
                batch = dict(batch)
                batch['completion_unanchored_mask'] = F.pad(
                    batch['completion_unanchored_mask'].to(device),
                    (0, 0, 0, pad_t),
                    value=False,
                )
            T = self.official_t2m_frames

        ref_pose = batch.get('ref_pose')
        if ref_pose is not None and not isinstance(ref_pose, Tensor):
            ref_pose = None
        if ref_pose is not None:
            ref_pose = ref_pose.to(device)

        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), T
        )

        # Prepare text
        # CRITICAL: must match training-time convention (see
        # MotionCanvasTrainer._prepare_and_forward):
        #   1. ctxt_input is always padded to max_text_len=128, regardless of
        #      the per-sample caption length.
        #   2. ctxt_mask_temporal marks valid tokens (True = valid, False = pad).
        #   3. Null-caption samples get the *learned* null_ctxt_input broadcast
        #      to the full (1, 128, 4096) shape — NOT a zero tensor with one
        #      active token. Attention masks are all-False for null samples
        #      (training convention in `_length_to_mask(ctxt_length=0, 128)`).
        #   4. All tensors must match the transformer's parameter dtype so
        #      attention math happens in the same precision as training.
        # Earlier code used raw caption seq_len (12-20) for ctxt and
        # zeros-with-first-token-null for CFG, which led to distorted
        # captioned outputs because the model never saw those distributions
        # during training. (2026-04-20)
        pad_len = self.max_text_len
        model_dtype = next(self.bundle.motion_transformer.parameters()).dtype

        def _pad_ctxt(ctxt: Tensor, length_is_valid: bool) -> Tensor:
            """Pad / truncate ctxt to (B, pad_len, D)."""
            if ctxt.shape[1] == pad_len:
                return ctxt
            if ctxt.shape[1] < pad_len:
                return F.pad(ctxt, (0, 0, 0, pad_len - ctxt.shape[1]))
            return ctxt[:, :pad_len]

        if 'text_vec_raw' in batch:
            vtxt_input = batch['text_vec_raw'].to(device=device, dtype=model_dtype)
            ctxt_raw = batch['text_ctxt_raw'].to(device=device, dtype=model_dtype)
            ctxt_input = _pad_ctxt(ctxt_raw, True)
            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
        else:
            # Unconditioned inference: MUST match training convention in
            # MotionCanvasTrainer._prepare_and_forward lines 212-215:
            #   ctxt_input = null_ctxt_input.expand(B, 1, -1)   ← 1 token
            #   ctxt_length = 1
            #   ctxt_mask = all-True over length 1
            # Earlier code used pad_len (128) here for symmetry with the
            # captioned branch, but the uncond-trained model never saw a
            # 128-token context during training — it always saw a single
            # null token. Feeding 128 repeated null tokens + all-False
            # attention mask is a severe OOD shift that produces catastrophic
            # jitter in output (found 2026-04-21).
            vtxt_input = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(B, 1, -1).contiguous()
            ctxt_length = torch.ones(B, dtype=torch.long, device=device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1)

        # Prepare edit context and target mask.
        condition_context = self.bundle.prepare_condition_context(
            src_motion=src_motion,
            ref_pose=ref_pose,
            src_mask=src_mask,
        )

        do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode

        # CFG null-branch construction.  The "silent" CFG branch nulls BOTH
        # sentence-level vtxt AND token-level ctxt to match training-time
        # mask_text_cond behavior (which masks both vtxt and ctxt).
        # Previously only vtxt was nulled while ctxt was kept intact, making
        # CFG guidance depend solely on the 768-dim vtxt difference — far too
        # weak for effective caption guidance.  Fixed 2026-05-15.
        if do_cfg:
            null_vtxt = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand_as(vtxt_input)
            # Expand null_ctxt to match ctxt_input's sequence length so
            # torch.cat along batch dim works correctly.
            null_ctxt = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(
                ctxt_input.shape[0], ctxt_input.shape[1], -1
            ).contiguous()
            # Match HYMotion T2M: the null branch uses learned null embeddings
            # expanded to the caption length and reuses the caption attention
            # mask.  Collapsing to a single token changes the CFG distribution
            # for HYMotion-Lite warm starts.
            null_ctxt_mask = ctxt_mask_temporal

        # ODE function
        ode_cfg = dict(self.bundle._noise_scheduler_cfg)
        ode_cfg.pop('method', None)  # odeint uses it positionally
        sources = _batch_sources(batch, B)
        sources_cfg = ([''] * B + sources) if (do_cfg and sources is not None) else sources

        def fn(t: Tensor, x: Tensor) -> Tensor:
            x_input = torch.cat([x, condition_context], dim=-1)
            if do_cfg:
                x_input = torch.cat([x_input, x_input], dim=0)
            x_pred = self.bundle.predict_flow(
                x_input=x_input,
                ctxt_input=(
                    ctxt_input if not do_cfg
                    else torch.cat([null_ctxt, ctxt_input], dim=0)
                ),
                vtxt_input=(
                    vtxt_input if not do_cfg
                    else torch.cat([null_vtxt, vtxt_input], dim=0)
                ),
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=(
                    tgt_padding_mask if not do_cfg
                    else tgt_padding_mask.repeat(2, 1)
                ),
                ctxt_mask_temporal=(
                    ctxt_mask_temporal if not do_cfg
                    else torch.cat([null_ctxt_mask, ctxt_mask_temporal], dim=0)
                ),
                sources=sources_cfg,
                trigger_sources={'Taobao', 'Game'},
                special_game_prob=1.0,
            )

            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                if do_cfg:
                    x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t).clamp_min(t_eps)
                else:
                    x_pred = (x_pred - x) / (1.0 - t).clamp_min(t_eps)

            if do_cfg:
                pred_basic, pred_text = x_pred.chunk(2, dim=0)
                x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
            return x_pred

        # -----------------------------------------------------------------
        # Initial y0 and replacement guidance setup
        # -----------------------------------------------------------------
        z = torch.randn(B, T, D, device=device, dtype=src_motion.dtype)
        t = torch.linspace(0, 1, self.num_steps + 1, device=device, dtype=src_motion.dtype)

        rep_mode = self.replacement_guidance
        valid_frame_mask = tgt_padding_mask.unsqueeze(-1)  # (B, T, 1)
        valid_generate = (
            (src_mask > 0.5) & valid_frame_mask
            if src_mask is not None
            else None
        )
        valid_known = (
            (src_mask < 0.5) & valid_frame_mask
            if src_mask is not None
            else None
        )
        use_replacement = (
            rep_mode != 'none'
            and src_mask is not None
            and bool(valid_generate.any().item())
            and bool(valid_known.any().item())
        )

        if use_replacement:
            # keep_mask: (B, T, D), True = known region (mask=0).
            #
            # ⚠️ 2026-04-24 bug fix ("E13 每段尾帧静止"): exclude PAD frames
            # from the known region even if src_mask=0 there. Rationale: under
            # training distribution, pad frames (idx >= tgt_length) carry
            # src_mask=0 AND src_motion=0 AND attention is masked out by
            # tgt_padding_mask AND loss is masked out. The model never
            # "sees" pad frames during training. If we leave keep_mask=True
            # on pad frames at inference time, the replacement loop below
            # pins x[pad] ← x_clean[pad] = replicate(normalize(synthetic-zero
            # last frame)) every ODE step — i.e. it anchors the entire pad
            # region to the training-set MEAN pose. For cases where the
            # "synthetic zero" is just the training mean (E13 where src_raw
            # is zeros outside the prefix, or any short-clip inference where
            # we pad by replicating a valid end frame), this mean-pose
            # anchor leaks into the valid region via shared LayerNorms /
            # residual paths (pad is key-masked but still flows as a
            # query/value through per-token feedforwards) and pulls the
            # TAIL of the valid region visibly toward static "mean pose".
            # Users reported this as "每段尾帧几乎不动、静止" on E13.
            #
            # Fix: combine src_mask (per-frame-per-dim "is this a known
            # sample?") with tgt_padding_mask ("is this a valid frame in
            # the model's view?"). Pad frames get keep_mask=False → neither
            # y0 init (below) nor per-step replacement touches them. They
            # become ordinary ODE free-evolve tokens — consistent with
            # training where the model is simply not asked about them.
            keep_mask = (src_mask < 0.5) & valid_frame_mask
            # clean_motion: full normalized motion WITHOUT masked-region
            # zeroing. Required for clean imputation.
            assert 'clean_motion' in batch, (
                'replacement_guidance requires "clean_motion" in batch '
                '(full normalized motion without zeroing masked regions)'
            )
            x_clean = batch['clean_motion'].to(device)

            if self.sdedit_tau > 0.0:
                # SDEdit-style partial-noise start on masked region.
                # Flow-matching convention (in this pipeline): t=0 → noise,
                # t=1 → clean, so x_t = (1-t)*z + t*clean. To start from τ
                # noise fraction we set t_init = 1 - τ. The loop below will
                # honor this by skipping ODE steps with t_curr < t_init.
                tau = self.sdedit_tau
                t_init = 1.0 - tau
                completion_unanchored = batch.get(
                    'completion_unanchored_mask'
                )
                promote_completion_to_full_regeneration = False
                if completion_unanchored is not None:
                    completion_unanchored = completion_unanchored.to(
                        device=device, dtype=torch.bool
                    )
                    if completion_unanchored.shape != x_clean.shape:
                        raise ValueError(
                            'completion_unanchored_mask shape differs from '
                            f'latent: {tuple(completion_unanchored.shape)} != '
                            f'{tuple(x_clean.shape)}'
                        )
                    promote_completion_to_full_regeneration = bool(
                        completion_unanchored.any().item()
                    )
                if promote_completion_to_full_regeneration:
                    # Full-span generated channels have no clean-blind SDEdit
                    # anchor. Starting them at t=0 noise while integrating
                    # only the final tau fraction left essentially raw noise
                    # in the output. Promote the sample to ordinary completion:
                    # all generated cells receive the complete ODE trajectory,
                    # while known cells remain hard-imputed throughout.
                    y0 = torch.where(keep_mask, x_clean, z)
                    sdedit_t_init = None
                else:
                    x_partial_noised = (
                        (1.0 - t_init) * z + t_init * x_clean
                    )
                    y0 = torch.where(keep_mask, x_clean, x_partial_noised)
                    sdedit_t_init = t_init
            else:
                # Training fixes x_t[known] = x1 (clean). Match it at t=0.
                y0 = torch.where(keep_mask, x_clean, z)
                sdedit_t_init = None
        else:
            y0 = z
            sdedit_t_init = None

        # -----------------------------------------------------------------
        # Position constraint setup
        # -----------------------------------------------------------------
        position_constraints = batch.get('position_constraints')
        use_pos_constraint = position_constraints is not None and len(position_constraints) > 0
        pos_solver = None
        pos_affected_dims = None

        if use_pos_constraint:
            from motius.motion.pipeline_utils.position_constraint import (
                PositionConstraintSolver,
                get_affected_dims,
            )
            bone_offsets = self.bundle.get_bone_offsets()
            rotation_space = getattr(self.bundle, 'rotation_space', 'local')
            pos_solver = PositionConstraintSolver(
                bone_offsets=bone_offsets,
                rotation_space=rotation_space,
            )
            pos_affected_dims = get_affected_dims(position_constraints)
            pc_interval = self.position_constraint_interval

        # -----------------------------------------------------------------
        # ODE integration
        # -----------------------------------------------------------------
        exact_sdedit_start = bool(batch.get("integration_t_start_exact", False))
        if use_replacement or use_pos_constraint:
            # Manual Euler with per-step imputation and/or position constraint.
            # ``y0`` is sampled at the exact SDEdit time. Snapping its first
            # vector-field evaluation to the next coarse grid point creates a
            # deterministic residual on every channel. Detector probes insert
            # the exact start; legacy Step-E repair retains its frozen grid.
            integration_t = _sdedit_integration_times(
                t,
                sdedit_t_init,
                exact_start=exact_sdedit_start,
            )
            # Store initial noise for flow_interp mode
            z0 = y0.clone() if rep_mode == 'flow_interp' else None
            x = y0
            active_steps = len(integration_t) - 1
            for active_i in range(active_steps):
                t_curr = integration_t[active_i]
                dt = integration_t[active_i + 1] - integration_t[active_i]
                is_last_step = active_i == active_steps - 1

                v = fn(t_curr, x)
                x = x + v * dt

                # Imputation: force known regions back to expected values.
                if use_replacement:
                    if rep_mode == 'flow_interp' and not is_last_step:
                        t_next = integration_t[active_i + 1]
                        x_interp = (1 - t_next) * z0 + t_next * x_clean
                        x = torch.where(keep_mask, x_interp, x)
                    elif rep_mode == 'all' or (rep_mode == 'skip_last' and not is_last_step):
                        x = torch.where(keep_mask, x_clean, x)

                # Position constraint projection
                if use_pos_constraint:
                    # Analytic IK (root/2-bone/1-bone): every step
                    # Gradient IK: every pc_interval steps + last step
                    # Preserve the legacy cadence on base-grid steps. The
                    # detector-only exact substep has no position constraints,
                    # while Step-E must keep its historical global-grid phase.
                    base_step_index = _base_grid_step_index(t, t_curr)
                    do_gradient_ik = is_last_step or (
                        base_step_index % pc_interval == 0
                    )

                    # Denormalize -> IK solve -> renormalize
                    x_denorm = self.bundle.denormalize_motion(x)
                    x_fixed = x_denorm.clone()

                    for b_idx in range(B):
                        frame_constraints = {}  # frame -> list of constraints
                        for c in position_constraints:
                            frame_constraints.setdefault(c.frame, []).append(c)

                        for frame, cs in frame_constraints.items():
                            if frame >= T:
                                continue
                            # Filter by IK type
                            from motius.motion.pipeline_utils.ik_solver import get_ik_strategy
                            analytic_cs = [
                                c for c in cs
                                if get_ik_strategy(c.joint) in ('root', 'two_bone', 'one_bone')
                            ]
                            gradient_cs = [
                                c for c in cs
                                if get_ik_strategy(c.joint) == 'gradient'
                            ]

                            active_cs = analytic_cs
                            if do_gradient_ik:
                                active_cs = active_cs + gradient_cs

                            if active_cs:
                                frame_motion = x_fixed[b_idx, frame]
                                for c in active_cs:
                                    frame_result, _ = pos_solver._solve_single(
                                        frame_motion.unsqueeze(0), [c]
                                    )
                                    frame_motion = frame_result.squeeze(0)
                                x_fixed[b_idx, frame] = frame_motion

                    # Renormalize and selectively replace affected dims
                    x_renorm = self.bundle.normalize_motion(x_fixed)
                    if pos_affected_dims is not None:
                        dim_idx = torch.tensor(pos_affected_dims, device=device)
                        # Only replace affected frames and dims
                        affected_frames = set(c.frame for c in position_constraints if c.frame < T)
                        for f in affected_frames:
                            x[:, f, dim_idx] = x_renorm[:, f, dim_idx]
                    else:
                        x = x_renorm

            # Final hard replacement guarantees exact evidence preservation for
            # skip_last and flow_interp. Training now targets zero velocity on
            # clean-imputed coordinates, but an exact projection still avoids
            # accumulating solver and finite-precision error on known values.
            if use_replacement and rep_mode in ('skip_last', 'flow_interp'):
                x = torch.where(keep_mask, x_clean, x)

            sampled = x
        else:
            # Standard path: use torchdiffeq if available, else manual Euler.
            try:
                from torchdiffeq import odeint
                method = self.bundle._noise_scheduler_cfg.get('method', 'euler')
                trajectory = odeint(fn, y0, t, method=method)
            except ImportError:
                trajectory = [y0]
                dt = 1.0 / self.num_steps
                x = y0
                for i in range(self.num_steps):
                    t_val = torch.tensor(i * dt, device=device, dtype=src_motion.dtype)
                    v = fn(t_val, x)
                    x = x + v * dt
                    trajectory.append(x)
                trajectory = torch.stack(trajectory, dim=0)

            sampled = trajectory[-1]

        # Decode to motion
        result = self.bundle.decode_motion_from_latent(sampled)
        result['latent'] = sampled
        result['rotation_space'] = getattr(self.bundle, 'rotation_space', 'local')
        return result

    # ------------------------------------------------------------------ #
    # Motion repair (E9): defect detection + masked regeneration.        #
    # ------------------------------------------------------------------ #
    T_PAD_REPAIR = 360  # the context length the model was trained with

    def _repair_forward(
        self,
        motion_norm: Tensor,     # (1, T_pad, D) normalized LQ, zero-padded
        mask135: Tensor,         # (1, T_pad, D) 1=generate, 0=keep
        clean_motion: Tensor,    # (1, T_pad, D) normalized full LQ (no zeroing)
        valid_len: int,
        replacement_guidance: str,
        sdedit_tau: float,
        *,
        edit_mode: bool,
        exact_sdedit_start: bool = False,
        text_fields: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """One masked-imputation forward pass. Returns normalized latent
        ``(1, T_pad, D)``. Temporarily overrides the instance replacement /
        SDEdit settings so a single pipeline object can serve every repair
        configuration without re-construction."""
        prev_repl, prev_tau = self.replacement_guidance, self.sdedit_tau
        self.replacement_guidance = replacement_guidance
        self.sdedit_tau = float(sdedit_tau)
        try:
            # Step-E repair is edit-mode: corrupted LQ inside target_mask is
            # evidence. Stage-A self-denoise remains completion-style. Known
            # cells are protected independently by replacement/projection.
            src_motion = (
                motion_norm
                if edit_mode
                else motion_norm * (1.0 - mask135)
            )
            # In completion mode, masked values are invalid observations.  The
            # reactive/source channel above already zeros them, but SDEdit also
            # reads ``clean_motion`` when constructing its partially noised
            # initial state.  Passing the raw LQ tensor there used to leak
            # ``(1 - tau) * corrupted`` into every masked cell (98% at the
            # Oracle-v6 tau=0.02 setting), even though the reactive channel was
            # correctly disabled.
            #
            # Build a clean-blind anchor from adjacent *known* cells instead.
            # Dimensions masked for the full valid clip have no observable
            # anchor and promote that sample to full-regeneration completion.
            completion_unanchored = None
            completion_anchor_policy = "editing_source_values"
            if not edit_mode:
                clean_motion, completion_unanchored = (
                    self._completion_safe_sdedit_anchor(
                        clean_motion,
                        mask135,
                        valid_len=valid_len,
                    )
                )
                completion_anchor_policy = (
                    "known_only_interpolation_fullspan_full_regeneration_v2"
                )
            batch = {
                'src_motion': src_motion,
                'src_mask': mask135,
                'src_length': [valid_len],
                'tgt_length': [valid_len],
                'clean_motion': clean_motion,
                'completion_unanchored_mask': completion_unanchored,
                'completion_anchor_policy': completion_anchor_policy,
                'integration_t_start_exact': bool(exact_sdedit_start),
            }
            if text_fields is not None:
                batch.update(text_fields)
            out = self._inference(batch)
        finally:
            self.replacement_guidance, self.sdedit_tau = prev_repl, prev_tau
        return out['latent']

    @staticmethod
    def _completion_safe_sdedit_anchor(
        clean_motion: Tensor,
        generate_mask: Tensor,
        *,
        valid_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Remove masked LQ values from a completion-mode SDEdit anchor.

        Masked runs are linearly interpolated from their nearest known values
        in the same channel.  One-sided runs use the nearest known value.
        Channels masked over the complete valid interval cannot be inferred
        from condition support; their returned ``unanchored`` mask instructs
        :meth:`_inference` to start those cells from pure noise.
        """
        clean = clean_motion.clone()
        mask = generate_mask >= 0.5
        if clean.ndim != 3 or mask.shape != clean.shape:
            raise ValueError(
                "completion SDEdit anchor expects equal [B,T,D] tensors, got "
                f"{tuple(clean.shape)} and {tuple(mask.shape)}"
            )
        length = int(valid_len)
        if length < 1 or length > clean.shape[1]:
            raise ValueError(
                f"invalid completion anchor valid_len={length} for T={clean.shape[1]}"
            )
        unanchored = torch.zeros_like(mask)
        frame_indices = torch.arange(
            length, device=clean.device, dtype=clean.dtype
        )
        for batch_index in range(clean.shape[0]):
            for channel in range(clean.shape[2]):
                channel_mask = mask[batch_index, :length, channel]
                if not bool(channel_mask.any().item()):
                    continue
                known_indices = torch.nonzero(
                    ~channel_mask, as_tuple=False
                ).flatten()
                if known_indices.numel() == 0:
                    clean[batch_index, :length, channel] = 0.0
                    unanchored[batch_index, :length, channel] = channel_mask
                    continue

                insertion = torch.searchsorted(known_indices, frame_indices)
                left_slot = torch.clamp(insertion - 1, min=0)
                right_slot = torch.clamp(
                    insertion, max=known_indices.numel() - 1
                )
                left_index = known_indices[left_slot]
                right_index = known_indices[right_slot]
                left_value = clean[batch_index, left_index, channel]
                right_value = clean[batch_index, right_index, channel]
                interval = (right_index - left_index).to(clean.dtype)
                alpha = torch.where(
                    interval > 0,
                    (frame_indices - left_index.to(clean.dtype))
                    / interval.clamp_min(1.0),
                    torch.zeros_like(frame_indices),
                )
                interpolated = left_value + alpha * (right_value - left_value)
                clean[batch_index, :length, channel] = torch.where(
                    channel_mask,
                    interpolated,
                    clean[batch_index, :length, channel],
                )
        return clean, unanchored

    @torch.no_grad()
    def _self_denoise_joint_change(
        self, mn: Tensor, stage1: Tensor, valid_len: Optional[int] = None
    ):
        """Physical per-joint change for self-denoise detection.

        Returns ``(jchg, tchg)`` where ``jchg`` is the (T, 22) geodesic angle
        (radians) between the LQ and stage-1 *local* joint rotations and
        ``tchg`` is the (T,) translation L2 change (meters) after removing the
        robust clip-level offset between the input and the model projection.
        Measuring in a physical space (angle/meters) -- instead of the z-scored channel |Δ|
        used by ``compute_ada_keep_mask`` -- gives the threshold a real meaning
        and avoids the below-noise-floor saturation; magnitude aggregation
        (one scalar per joint) replaces the coverage-amplifying 6-channel OR.
        """
        from motius.models.motioncanvas.network.geometry import (
            rot6d_to_rotation_matrix,
        )
        raw_lq = self.bundle.denormalize_motion(mn)        # (1, T, D)
        raw_s1 = self.bundle.denormalize_motion(stage1)
        T = raw_lq.shape[1]
        r6_lq = raw_lq[0, :, 3:135].reshape(T, 22, 6)
        r6_s1 = raw_s1[0, :, 3:135].reshape(T, 22, 6)
        if getattr(self.bundle, 'rotation_space', 'local') == 'global':
            # Compare *local* joint rotations (isolates each joint's own
            # defect; global angles would propagate parent error down the
            # kinematic chain and inflate coverage).
            from motius.motion.skeleton.fk import global_to_local_rot6d
            r6_lq = global_to_local_rot6d(r6_lq.unsqueeze(0))[0]
            r6_s1 = global_to_local_rot6d(r6_s1.unsqueeze(0))[0]
        R_lq = rot6d_to_rotation_matrix(r6_lq.reshape(-1, 6)).reshape(T, 22, 3, 3)
        R_s1 = rot6d_to_rotation_matrix(r6_s1.reshape(-1, 6)).reshape(T, 22, 3, 3)
        R_rel = torch.matmul(R_lq.transpose(-1, -2), R_s1)
        tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        jchg = torch.arccos(cos)                            # (T, 22) radians
        translation_delta = raw_lq[0, :, :3] - raw_s1[0, :, :3]
        # A self-denoising projection is free to choose a different global
        # origin.  Treating that clip-wide bias as a per-frame corruption made
        # the root mask saturate at 99.97% on BrokenAMASS.  Remove the median
        # bias first; the remaining residual measures trajectory-shape changes.
        valid = T if valid_len is None else max(1, min(int(valid_len), T))
        clip_offset = translation_delta[:valid].median(dim=0).values
        tchg = torch.linalg.norm(
            translation_delta - clip_offset[None], dim=-1)  # (T,) meters
        return jchg.cpu().numpy(), tchg.cpu().numpy()

    @staticmethod
    def _joint_change_to_raw_mask(jchg, tchg, model_dim, joint_thr, trans_thr):
        """Threshold the physical per-joint angle / translation change into a
        ``(T, model_dim)`` raw generate-mask (1=defect). Flags a joint's rot6d
        (and FK-position) channels when its angle change exceeds ``joint_thr``
        (rad); flags translation when ``tchg`` exceeds ``trans_thr`` (m)."""
        T = jchg.shape[0]
        out = np.zeros((T, model_dim), dtype=np.float32)
        jflag = jchg > float(joint_thr)        # (T, 22)
        tflag = tchg > float(trans_thr)        # (T,)
        out[tflag, :3] = 1.0
        for j in range(22):
            out[jflag[:, j], 3 + j * 6: 3 + (j + 1) * 6] = 1.0
            if model_dim >= 198 and j >= 1:
                out[jflag[:, j], 135 + (j - 1) * 3: 135 + j * 3] = 1.0
        return out

    @torch.no_grad()
    def infer_repair(
        self,
        motion: Tensor,
        lengths: Optional[List[int]] = None,
        *,
        # mask source (axis 2)
        mask_source: str = 'self_denoise',     # 'self_denoise' | 'provided'
        adaptive_mask: Optional[Tensor] = None,  # (B,T,22) or (B,T,D), 1=defect
        adaptive_translation_mask: Optional[Tensor] = None,  # (B,T), 1=defect
        adaptive_translation_axis_mask: Optional[Tensor] = None,  # (B,T,3), 1=defect
        adaptive_position_mask: Optional[Tensor] = None,  # (B,T,22|21), 1=defect
        detect_tau: float = 0.3,                # SDEdit tau for stage-1 projection
        detect_metric: str = 'angle',           # 'angle' (MoGenDIT-style) | 'abs'
        detect_joint_thr_rad: float = 0.15,      # per-joint geodesic angle (rad)
        detect_trans_thr_m: float = 0.05,        # translation change (meters)
        detect_threshold_mode: str = 'abs',     # 'abs' | 'topk_pct' (metric='abs')
        detect_threshold: float = 0.1,
        # mask tightening
        strict_tighten: bool = True,
        strict_dilate: int = 2,
        strict_min_blob: int = 3,
        # the 4 configurable axes
        translation_mode: str = 'lock',         # axis 1: 'lock'|'detected'|'all'
        mask_granularity: str = 'joint',        # axis 4: 'joint'|'frame'
        position_mask_mode: str = 'joint_coupled',
        sdedit_tau: float = 0.5,                 # axis 3: 0=from-scratch, >0=partial
        replacement_guidance: str = 'skip_last',
        presmooth_sigma: float = 0.0,            # Gaussian temporal pre-smooth of kept LQ
        repair_edit_mode: bool = False,
        text_fields: Optional[Dict[str, Any]] = None,
        return_mask: bool = True,
        return_debug_latent: bool = False,
        detector_only: bool = False,
        repair_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Repair defective motion with masked regeneration.

        This is the single canonical entry point for MotionCanvas motion
        repair -- call this instead of hand-assembling a batch, so the repair
        recipe lives in one discoverable place.

        The four configurable axes
        --------------------------
        translation_mode : axis 1 -- how the global root translation is treated.
            ``'lock'`` never regenerates translation (M7 convention; keeps the
            global trajectory locked to the input -- recommended, avoids root
            drift). ``'detected'`` regenerates translation only on frames where
            the independent root-trajectory detector fires. ``'all'``
            regenerates translation on every valid frame.
        mask_source : axis 2 -- where the defect mask comes from.
            ``'self_denoise'`` (ours): run a stage-1 SDEdit-from-LQ projection
            with this model and threshold ``|LQ - projection|`` (MoGenDIT-style
            ada_denoise but with *our* model). ``'provided'``: use the mask in
            ``adaptive_mask`` (e.g. a MoGenDIT-computed or QC mask). Keeping the
            external mask as a passed-in argument avoids coupling this pipeline
            to other methods' models.
        sdedit_tau : axis 3 -- regeneration strength for masked cells.
            ``0`` starts the masked region from pure noise (full regeneration
            from scratch). ``>0`` starts from ``tau*noise + (1-tau)*LQ`` and only
            runs the last ``tau`` of the ODE (partial re-noise; stays close to
            the input -- gentle cleanup).
        mask_granularity : axis 4 -- spatial extent of regeneration.
            ``'joint'`` regenerates only the flagged joints' channels.
            ``'frame'`` regenerates every joint of any frame that has at least
            one flagged joint (whole-frame regeneration). ``'channel'``
            regenerates only the individual flagged channels (no per-joint OR,
            no strict tightening) -- the MoGenDIT-faithful per-element scheme;
            requires ``mask_source='self_denoise'``.

        Parameters
        ----------
        motion : (B, T, D) or (T, D) tensor
            Raw (un-normalized) LQ motion in the model's representation
            (135-dim, local rotation). Must already be at the model fps.
        lengths : optional list[int]
            Valid frame count per sample (defaults to full T).

        Returns
        -------
        dict with ``motion`` (B,T,135 repaired: transl + rot6d), and when
        ``return_mask``: ``joint_mask`` (B,T,22 bool), ``translation_mask``
        (B,T bool), ``translation_axis_mask`` (B,T,3 bool), and ``mask``
        (B,T,model_dim generate-mask). For the
        physical-angle self-denoise detector, the raw pre-threshold scores are
        also returned as ``detector_joint_change_rad`` and
        ``detector_translation_change_m``. For ``detect_metric='abs'``, the
        pre-threshold normalized per-channel residual is returned as
        ``detector_channel_change_normalized``. ``detector_only=True`` skips
        the second (repair) diffusion pass and returns the input motion
        unchanged; it is intended for threshold sweeps that reuse one stage-1
        projection.
        When ``repair_seed`` is set, every sample resets Torch CPU/CUDA and
        NumPy RNGs immediately before its Step-E repair pass, using
        ``repair_seed + sample_index``. This makes the final repair noise
        independent of whether a self-denoise detector pass ran first.
        """
        if mask_source not in ('self_denoise', 'provided'):
            raise ValueError(f'mask_source must be self_denoise|provided, got {mask_source!r}')
        if mask_source == 'provided' and adaptive_mask is None:
            raise ValueError("mask_source='provided' requires adaptive_mask")
        if position_mask_mode not in ('fk_descendant', 'joint_coupled'):
            raise ValueError(
                'position_mask_mode must be fk_descendant|joint_coupled, '
                f'got {position_mask_mode!r}'
            )
        if detector_only and mask_source != 'self_denoise':
            raise ValueError(
                "detector_only requires mask_source='self_denoise'"
            )

        device = next(self.bundle.motion_transformer.parameters()).device
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        motion = motion.float().to(device)
        B, T, D = motion.shape
        T_PAD = self.T_PAD_REPAIR
        if T > T_PAD:
            raise NotImplementedError(
                f'infer_repair supports motions up to {T_PAD} frames (got {T}). '
                'For longer sequences, chunk into <=360-frame windows and stitch, '
                'or use the windowed eval path in eval_m2m_v2_all_tasks.py.'
            )
        if lengths is None:
            lengths = [T] * B

        # The model operates on its native motion_dim (198 = 3 transl + 132
        # rot6d + 63 FK joint positions). The caller passes 135-dim (transl +
        # rot6d). We FK-expand 135->198 with the bundle's bone offsets, mirroring
        # the trainer/eval (eval_m2m_v2_all_tasks.motion_135_to_198) so mean/std
        # and the per-channel layout match training. Output is decoded back to
        # 135 (transl + rot6d) which is the canonical caller representation.
        from motius.motion.pipeline_utils.repair_utils import motion_135_to_198
        rot_space = getattr(self.bundle, 'rotation_space', 'local')
        # The model dim is the normalizer width (mean/std), not necessarily a
        # `motion_dim` attribute (which may be unset on some bundles).
        model_dim = int(self.bundle.mean.shape[-1])
        if model_dim not in (135, 198):
            raise RuntimeError(
                'infer_repair requires 135- or 198-d normalization stats; '
                f'got mean shape={tuple(self.bundle.mean.shape)} and '
                f'std shape={tuple(self.bundle.std.shape)}. This usually '
                'means the configured mean_std_dir or Mean.npy/Std.npy is '
                'missing or was resolved relative to the wrong cwd.'
            )
        if tuple(self.bundle.std.shape) != tuple(self.bundle.mean.shape):
            raise RuntimeError(
                'infer_repair normalization shape mismatch: '
                f'mean={tuple(self.bundle.mean.shape)} '
                f'std={tuple(self.bundle.std.shape)}'
            )
        bone_offsets_np = None
        if model_dim >= 198:
            bone_offsets_np = self.bundle.get_bone_offsets().cpu().numpy()  # (22,3)

        local_to_global = None
        if rot_space == 'global':
            from motius.motion.skeleton.fk import (
                local_to_global_rot6d as local_to_global,
            )

        repaired = motion.clone()
        joint_masks = np.zeros((B, T, 22), dtype=bool)
        translation_masks = np.zeros((B, T), dtype=bool)
        translation_axis_masks = np.zeros((B, T, 3), dtype=bool)
        position_masks = np.zeros((B, T, 21), dtype=bool)
        mask_dim_out = np.zeros((B, T, model_dim), dtype=np.float32)
        detector_joint_change = np.full((B, T, 22), np.nan, dtype=np.float32)
        detector_translation_change = np.full((B, T), np.nan, dtype=np.float32)
        detector_scores_available = False
        detector_channel_change = np.full(
            (B, T, model_dim), np.nan, dtype=np.float32
        )
        detector_channel_scores_available = False
        debug_latent_denorm = None
        debug_candidate_135 = None
        if return_debug_latent:
            debug_latent_denorm = np.full(
                (B, T, model_dim), np.nan, dtype=np.float32
            )
            debug_candidate_135 = np.full(
                (B, T, 135), np.nan, dtype=np.float32
            )

        for b in range(B):
            L = int(lengths[b])
            m135 = motion[b].detach().cpu().numpy()              # (T,135) local

            # 135 -> model_dim raw (append FK positions for 198-dim models).
            if model_dim >= 198:
                raw = motion_135_to_198(m135, bone_offsets_np)   # (T,198) local
            else:
                raw = m135.astype(np.float32).copy()
            # local -> global rot6d if the model trains in world frame.
            if local_to_global is not None:
                rl = torch.from_numpy(
                    raw[:, 3:135].reshape(T, 22, 6)).float()
                raw = raw.copy()
                raw[:, 3:135] = local_to_global(rl).reshape(T, 132).numpy()

            mn = self.bundle.normalize_motion(
                torch.from_numpy(raw).float().unsqueeze(0).to(device))  # (1,T,model_dim)
            clean = mn.clone()
            if T < T_PAD:
                # Replicate the last valid frame into the pad region (static
                # hold) rather than zero-padding. Zeros == the normalized MEAN
                # pose, and because pad frames free-evolve under the ODE the
                # last *valid* frame gets pulled toward that mean pose -> a
                # systematic last-frame teleport (jumpLast ~20x the normal
                # per-frame step on ~all clips). A replicated static hold is
                # in-distribution (many training clips end static) and keeps
                # the boundary continuous.
                n_pad = T_PAD - T
                mn = torch.cat([mn, mn[:, -1:].expand(-1, n_pad, -1)], dim=1)
                clean = torch.cat([clean, clean[:, -1:].expand(-1, n_pad, -1)], dim=1)

            # Step A: base defect mask (model_dim, 1=generate).
            if mask_source == 'self_denoise':
                ones = torch.ones_like(mn)
                if mask_granularity == 'channel' and detect_metric == 'abs':
                    # MoreDiff's adaptive-denoise probe observes the complete
                    # first frame. Keep the same anchor contract for the
                    # representation-native residual-mask path; preserving a
                    # single scalar (the legacy detector behavior below) is
                    # not an equivalent conditioning problem.
                    ones[:, :1, :] = 0.0
                else:
                    ones[:, 0, 0] = 0.0  # legacy detector anchor
                stage1 = self._repair_forward(
                    mn, ones, clean, valid_len=L,
                    replacement_guidance='skip_last', sdedit_tau=detect_tau,
                    edit_mode=False,
                    exact_sdedit_start=True,
                    text_fields=None,
                )
                if mask_granularity == 'channel':
                    # MoGenDIT-faithful: pure per-channel keep/regenerate in the
                    # normalized space (high_change = |LQ - projection| > thr),
                    # with NO per-joint OR aggregation. Each of the model_dim
                    # channels is decided independently, exactly like MoGenDIT's
                    # official ada_denoise (change_threshold on the 201-dim rep).
                    ch_change = np.abs(
                        mn[0].cpu().numpy() - stage1[0].cpu().numpy())
                    detector_channel_change[b, :T] = np.asarray(
                        ch_change[:T], dtype=np.float32
                    )
                    detector_channel_scores_available = True
                    raw_mask = (ch_change > float(detect_threshold)).astype(
                        np.float32)  # (T_PAD, model_dim)
                elif detect_metric == 'angle':
                    # MoGenDIT-style: compare in a *physical* space (per-joint
                    # geodesic angle in radians + translation in meters) so the
                    # threshold has meaning and is not buried under the z-scored
                    # reconstruction noise floor; aggregate by magnitude (one
                    # scalar/joint), not a 6-channel OR.
                    jchg, tchg = self._self_denoise_joint_change(
                        mn, stage1, valid_len=L
                    )
                    detector_joint_change[b, :T] = np.asarray(
                        jchg[:T], dtype=np.float32
                    )
                    detector_translation_change[b, :T] = np.asarray(
                        tchg[:T], dtype=np.float32
                    )
                    detector_scores_available = True
                    raw_mask = self._joint_change_to_raw_mask(
                        jchg, tchg, model_dim,
                        joint_thr=detect_joint_thr_rad,
                        trans_thr=detect_trans_thr_m,
                    )
                elif detect_metric == 'abs':
                    ch_change = np.abs(
                        mn[0].cpu().numpy() - stage1[0].cpu().numpy())
                    detector_channel_change[b, :T] = np.asarray(
                        ch_change[:T], dtype=np.float32
                    )
                    detector_channel_scores_available = True
                    raw_mask = compute_ada_keep_mask(
                        mn[0].cpu().numpy(), stage1[0].cpu().numpy(),
                        threshold_mode=detect_threshold_mode,
                        threshold=detect_threshold,
                    )  # (T_PAD, model_dim)
                else:
                    raise ValueError(
                        f'detect_metric must be angle|abs, got {detect_metric!r}')
            else:
                am = adaptive_mask[b]
                am = am.cpu().numpy() if isinstance(am, Tensor) else np.asarray(am)
                raw_mask = np.zeros((T_PAD, model_dim), dtype=np.float32)
                if am.ndim == 2 and am.shape[-1] == 22:
                    jm = am[:T].astype(np.float32)
                    raw_mask[:T, :3] = jm[:, 0:1]
                    for j in range(22):
                        raw_mask[:T, 3 + j * 6:3 + (j + 1) * 6] = jm[:, j:j + 1]
                    if model_dim >= 198:
                        for j in range(1, 22):
                            raw_mask[:T, 135 + (j - 1) * 3:135 + j * 3] = jm[:, j:j + 1]
                else:  # already (T,D-ish): copy what fits
                    cd = min(am.shape[-1], model_dim)
                    raw_mask[:min(am.shape[0], T_PAD), :cd] = \
                        am[:T_PAD, :cd].astype(np.float32)

            # Keep root translation detection separate from pelvis rotation.
            # Previously the joint aggregation below collapsed both into joint
            # 0, so a root-orientation flag also regenerated the global path.
            translation_axis_flag = raw_mask[:, :3] >= 0.5
            if adaptive_translation_axis_mask is not None:
                tm = adaptive_translation_axis_mask[b]
                tm = tm.cpu().numpy() if isinstance(tm, Tensor) else np.asarray(tm)
                tm = np.asarray(tm, dtype=bool)
                if tm.ndim != 2 or tm.shape[1] != 3:
                    raise ValueError(
                        'adaptive_translation_axis_mask must have shape '
                        f'(B,T,3), got sample shape {tm.shape}'
                    )
                translation_axis_flag[:] = False
                translation_axis_flag[:min(len(tm), T_PAD)] = tm[:T_PAD]
            elif adaptive_translation_mask is not None:
                tm = adaptive_translation_mask[b]
                tm = tm.cpu().numpy() if isinstance(tm, Tensor) else np.asarray(tm)
                tm = np.asarray(tm, dtype=bool).reshape(-1)
                translation_axis_flag[:] = False
                count = min(len(tm), T_PAD)
                translation_axis_flag[:count] = tm[:count, None]
            translation_axis_flag[L:] = False
            translation_flag = translation_axis_flag.any(axis=-1)
            translation_flag[L:] = False
            position_flag = None
            if adaptive_position_mask is not None:
                pm = adaptive_position_mask[b]
                pm = (
                    pm.cpu().numpy()
                    if isinstance(pm, Tensor)
                    else np.asarray(pm)
                )
                pm = np.asarray(pm, dtype=bool)
                if pm.ndim != 2 or pm.shape[1] not in (21, 22):
                    raise ValueError(
                        'adaptive_position_mask must have shape '
                        f'(B,T,21|22), got sample shape {pm.shape}'
                    )
                position_flag = np.zeros(
                    (T_PAD, pm.shape[1]), dtype=bool
                )
                position_flag[:min(len(pm), T_PAD)] = pm[:T_PAD]
                position_flag[L:] = False

            if mask_granularity == 'channel':
                # MoGenDIT-faithful path: no joint aggregation, no strict
                # tightening -- the per-channel mask IS the dim mask. Only apply
                # the translation policy and the valid-length guard.
                dim_mask = raw_mask.astype(np.float32).copy()
                dim_mask[L:] = 0.0
                if translation_mode == 'lock':
                    dim_mask[:, :3] = 0.0
                elif translation_mode == 'detected':
                    dim_mask[:, :3] = translation_axis_flag.astype(np.float32)
                elif translation_mode == 'all':
                    dim_mask[:L, :3] = 1.0
                if position_flag is not None and model_dim >= 198:
                    resolved_position = (
                        position_flag[:, 1:]
                        if position_flag.shape[1] == 22
                        else position_flag
                    )
                    dim_mask[:, 135:198] = np.repeat(
                        resolved_position[:, :, None], 3, axis=-1
                    ).reshape(T_PAD, 63).astype(np.float32)
                jflag = (dim_mask[:, 3:135].reshape(T_PAD, 22, 6) >= 0.5).any(-1)
                jflag[L:] = False
            else:
                # Step B: tighten (strict) -> per-joint flag.
                if strict_tighten:
                    tight = compute_strict_adaptive_mask(
                        raw_mask, dilate=strict_dilate, min_blob=strict_min_blob,
                        motion_dim=model_dim,
                        lock_trans=(translation_mode == 'lock'),
                    )
                else:
                    tight = raw_mask
                jflag = (tight[:, 3:135].reshape(T_PAD, 22, 6) >= 0.5).any(-1)
                jflag[L:] = False  # pad frames are always known

                # Step C: granularity (axis 4).
                if mask_granularity == 'frame':
                    frame_hit = jflag.any(axis=-1, keepdims=True)
                    jflag = np.broadcast_to(frame_hit, jflag.shape).copy()
                    jflag[L:] = False
                    if position_flag is not None:
                        position_width = position_flag.shape[1]
                        position_flag = np.broadcast_to(
                            frame_hit,
                            (T_PAD, position_width),
                        ).copy()
                        position_flag[L:] = False
                elif mask_granularity != 'joint':
                    raise ValueError(
                        f'mask_granularity must be joint|frame|channel, '
                        f'got {mask_granularity!r}')

                # Step D: expand to dim mask with translation policy (axis 1).
                dim_mask = joint_mask_to_dim_mask(
                    jflag, motion_dim=model_dim,
                    translation_mode=translation_mode,
                    translation_flag=translation_flag,
                    position_flag=position_flag,
                    position_mask_mode=position_mask_mode,
                    valid_len=L,
                )
                if translation_mode == 'detected':
                    # The legacy helper expands one frame flag to XYZ.  Oracle
                    # v7 keeps these axes independent so a horizontal defect
                    # can never regenerate the clean vertical root channel.
                    dim_mask[:, :3] = translation_axis_flag.astype(np.float32)

            if detector_only:
                # ``repaired`` was initialized from ``motion``. Keep it
                # untouched and expose only the detector support/scores, so a
                # grid search pays for one stage-1 projection per detect_tau
                # instead of one projection plus one repair for every
                # threshold combination.
                joint_masks[b] = jflag[:T]
                translation_masks[b] = (
                    dim_mask[:T, :3] >= 0.5
                ).any(axis=-1)
                translation_axis_masks[b] = dim_mask[:T, :3] >= 0.5
                if model_dim >= 198:
                    position_masks[b] = (
                        dim_mask[:T, 135:198].reshape(T, 21, 3) >= 0.5
                    ).any(axis=-1)
                mask_dim_out[b] = dim_mask[:T]
                continue
            mask_t = torch.from_numpy(dim_mask).float().unsqueeze(0).to(device)

            # Step D.5: pre-smooth the kept (unmasked) LQ. Partial regeneration
            # keeps jittery LQ on unmasked cells -- both as the conditioning the
            # model sees and as the values copied back into the output -- so the
            # residual corruption jitter survives and seams appear at mask
            # boundaries. A light Gaussian temporal smooth of the kept region
            # lowers that jitter (protect_mask = generate region, left intact).
            mn_e, clean_e = mn, clean
            if presmooth_sigma > 0.0:
                mn_e = _gaussian_temporal_smooth(mn, presmooth_sigma, protect_mask=mask_t)
                clean_e = _gaussian_temporal_smooth(clean, presmooth_sigma, protect_mask=mask_t)

            # Step E: masked regeneration (axis 3 = sdedit_tau). Motion repair
            # treats masked values as invalid by default, so they must not be
            # exposed through the edit-context branch. ``repair_edit_mode`` is
            # retained only as an explicit semantic-edit ablation.
            if repair_seed is not None:
                reset_repair_rng(int(repair_seed) + b)
            latent = self._repair_forward(
                mn_e, mask_t, clean_e, valid_len=L,
                replacement_guidance=replacement_guidance, sdedit_tau=sdedit_tau,
                edit_mode=bool(repair_edit_mode),
                text_fields=text_fields,
            )
            dec = self.bundle.decode_motion_from_latent(latent)   # local rot6d
            transl = dec['transl'][0, :T]                          # (T,3)
            rot6d = dec['rot6d'][0, :T].reshape(T, 132)            # (T,132) local
            repaired_candidate = torch.cat([transl, rot6d], dim=-1)
            if return_debug_latent:
                debug_latent_denorm[b] = (
                    dec['latent_denorm'][0, :T].detach().cpu().float().numpy()
                )
                debug_candidate_135[b] = (
                    repaired_candidate.detach().cpu().float().numpy()
                )

            # ``skip_last`` deliberately omits imputation after the final ODE
            # update so generated cells can meet the condition boundary more
            # smoothly.  The generic E9 evaluator subsequently projects every
            # condition cell back to the input, but this canonical repair entry
            # used to return the unprojected decode directly.  Consequently the
            # final vector field update modified even fully unmasked frames
            # (6.7 cm mean joint drift on the BrokenAMASS oracle run).
            #
            # Restore the API invariant here: mask=0 means an observed value
            # and must be bitwise inherited from ``motion``.  ``dim_mask`` may
            # use the model's 198-D layout, but its first 135 dimensions have
            # the same translation + per-joint-rotation support as ``m135``.
            known_135 = torch.from_numpy(
                dim_mask[:T, :135] < 0.5
            ).to(device=device)
            repaired[b] = torch.where(
                known_135,
                motion[b, :T, :135],
                repaired_candidate,
            )

            joint_masks[b] = jflag[:T]
            translation_masks[b] = (dim_mask[:T, :3] >= 0.5).any(axis=-1)
            translation_axis_masks[b] = dim_mask[:T, :3] >= 0.5
            if model_dim >= 198:
                position_masks[b] = (
                    dim_mask[:T, 135:198].reshape(T, 21, 3) >= 0.5
                ).any(axis=-1)
            mask_dim_out[b] = dim_mask[:T]

        result: Dict[str, Any] = {'motion': repaired}
        if return_mask:
            result['joint_mask'] = torch.from_numpy(joint_masks)
            result['translation_mask'] = torch.from_numpy(translation_masks)
            result['translation_axis_mask'] = torch.from_numpy(
                translation_axis_masks
            )
            result['position_mask'] = torch.from_numpy(position_masks)
            result['mask'] = torch.from_numpy(mask_dim_out)
        if detector_scores_available:
            result['detector_joint_change_rad'] = torch.from_numpy(
                detector_joint_change
            )
            result['detector_translation_change_m'] = torch.from_numpy(
                detector_translation_change
            )
        if detector_channel_scores_available:
            result['detector_channel_change_normalized'] = torch.from_numpy(
                detector_channel_change
            )
        if return_debug_latent:
            result['debug_model_latent_denorm'] = torch.from_numpy(
                debug_latent_denorm
            )
            result['debug_model_candidate_motion135'] = torch.from_numpy(
                debug_candidate_135
            )
        return result
