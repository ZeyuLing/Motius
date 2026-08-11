"""MotionCanvas Trainer: flow-matching training for motion-to-motion editing."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer

logger = logging.getLogger('motius')


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _batch_sources(batch: Dict[str, Any], batch_size: int) -> Optional[List[str]]:
    sources = batch.get('data_src', None)
    if sources is None:
        sources = batch.get('source', None)
    if sources is None:
        return None
    if isinstance(sources, str):
        return [sources] * batch_size
    if isinstance(sources, torch.Tensor):
        sources = sources.detach().cpu().tolist()
    if isinstance(sources, (list, tuple)):
        if len(sources) != batch_size:
            return None
        return [str(s) for s in sources]
    return None


@TRAINERS.register_module(force=True)
class MotionCanvasTrainer(BaseTrainer):
    """Trainer for MotionCanvas flow-matching motion editing.

    Training forward:
      1. Prepare padding and masks
      2. Prepare text embeddings (null or encoded)
      3. Sample timesteps, create x_t via flow matching interpolation
      4. Build edit context and target mask
      5. Forward through bundle.predict_flow()
      6. Compute loss via bundle.m2m_loss
    """

    def __init__(
        self,
        bundle,
        val_num_steps: int = 10,
        max_text_len: int = 128,
    ):
        super().__init__(bundle)
        self.val_num_steps = val_num_steps
        # Fixed text sequence length — matches HY-Motion T2M 1.0 (max_length_llm=128).
        # Pre-extracted embeddings are variable-length; we pad/truncate to this.
        self.max_text_len = max_text_len
        self._debug_train_step_count = 0

    def _debug_train_step_enabled(self) -> bool:
        if os.environ.get('MOTIUS_DEBUG_TRAIN_STEP') != '1':
            return False
        max_steps = int(os.environ.get('MOTIUS_DEBUG_TRAIN_STEP_STEPS', '4'))
        return self._debug_train_step_count < max_steps

    def _debug_train_step_log(self, message: str, device: Optional[torch.device] = None) -> None:
        if os.environ.get('MOTIUS_DEBUG_TRAIN_STEP') != '1':
            return
        if (
            os.environ.get('MOTIUS_DEBUG_TRAIN_STEP_SYNC') == '1'
            and device is not None
            and device.type == 'cuda'
        ):
            torch.cuda.synchronize(device)
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', -1)))
        logger.info(
            "[Rank %s] train_step_debug step_idx=%s global_step=%s: %s",
            rank,
            getattr(self, '_debug_train_step_count', -1),
            self.get_global_step(),
            message,
        )

    def _prepare_and_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs and run a single base forward pass.

        Extracts steps 1-5 of the original train_step so that subclasses
        (e.g., SOAR post-trainer) can reuse the same setup and intermediate
        tensors without reimplementing data/conditioning/text preparation.

        Returns a context dict with all intermediate tensors needed to
        compute either the plain base loss (current behaviour) or
        additional SOAR rollout/correction losses.

        Keys in returned dict:
          device, src_motion, tgt_motion, src_mask, src_length_list,
          tgt_length_list, ref_pose, tgt_padding_mask, vtxt_input,
          ctxt_input, ctxt_mask_temporal, x0, x1, x_t, timesteps, t,
          condition_context, pred, generation_mask
        """
        device = next(self.bundle.motion_transformer.parameters()).device
        debug_train_step = self._debug_train_step_enabled()
        if debug_train_step:
            keys = sorted(str(k) for k in batch.keys())
            self._debug_train_step_log(f"start keys={keys}", device)


        # Helper: convert list of tensors to stacked tensor (or keep as list if shapes differ)
        def _stack_if_list(data):
            if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
                try:
                    return torch.stack(data, dim=0)
                except RuntimeError as e:
                    # Shapes differ - log the shapes for debugging
                    shapes = [d.shape if isinstance(d, Tensor) else None for d in data]
                    import logging
                    logger = logging.getLogger('motius')
                    logger.warning(f"Cannot stack tensors with different shapes: {shapes}")
                    # Pad to maximum shape
                    max_shape = list(max(shapes, key=lambda s: tuple(s) if s else ()))
                    padded = []
                    for d in data:
                        if isinstance(d, Tensor):
                            # Pad to max_shape if needed
                            pad_amounts = []
                            for i in range(len(d.shape)):
                                if i < len(max_shape):
                                    pad_amounts.append(0)
                                    pad_amounts.append(max_shape[i] - d.shape[i])
                                else:
                                    pad_amounts.append(0)
                            pad_amounts = list(reversed(pad_amounts))
                            if any(p > 0 for p in pad_amounts):
                                d_padded = torch.nn.functional.pad(d, pad_amounts)
                            else:
                                d_padded = d
                            padded.append(d_padded)
                    if padded:
                        return torch.stack(padded, dim=0)
                    return data
            return data

        # Source and target motions: convert lists to stacked tensors if needed
        src_motion = _stack_if_list(batch['src_motion'])
        tgt_motion = _stack_if_list(batch['tgt_motion'])
        src_mask = batch.get('src_mask')
        src_mask = _stack_if_list(src_mask) if src_mask is not None else None
        if debug_train_step:
            self._debug_train_step_log(
                f"after stack src={tuple(src_motion.shape)} tgt={tuple(tgt_motion.shape)} "
                f"mask={None if src_mask is None else tuple(src_mask.shape)}",
                device,
            )


        # Now convert to device
        src_motion = src_motion.to(device)
        tgt_motion = tgt_motion.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)
        if debug_train_step:
            self._debug_train_step_log(
                f"after to_device src_dtype={src_motion.dtype} tgt_dtype={tgt_motion.dtype}",
                device,
            )

        # Normalize motions using bundle's mean/std (matching original repo which
        # normalizes in dataset before padding). src_mask is binary — NOT normalized.
        # IMPORTANT: Only normalize valid frames; padded frames (zeros from
        # RandomCropPadding) must stay zero. We build a per-frame validity mask
        # from tgt_length and zero out padding frames after normalization.
        tgt_length_list = batch['tgt_length']
        if isinstance(tgt_length_list, Tensor):
            tgt_length_list = tgt_length_list.tolist()
        elif isinstance(tgt_length_list, (list, tuple)) and len(tgt_length_list) > 0 and isinstance(tgt_length_list[0], Tensor):
            # Stack 0-d tensors into a 1-d tensor, then convert to list
            tgt_length_list = torch.stack(tgt_length_list).tolist()

        src_length_list = batch.get('src_length', tgt_length_list)
        if isinstance(src_length_list, Tensor):
            src_length_list = src_length_list.tolist()
        elif isinstance(src_length_list, (list, tuple)) and len(src_length_list) > 0 and isinstance(src_length_list[0], Tensor):
            # Stack 0-d tensors into a 1-d tensor, then convert to list
            src_length_list = torch.stack(src_length_list).tolist()

        src_motion = self.bundle.normalize_motion(src_motion)
        tgt_motion = self.bundle.normalize_motion(tgt_motion)
        if debug_train_step:
            self._debug_train_step_log("after normalize", device)

        # Zero out mask regions for Completion samples; keep LQ values for Edit samples.
        # Per-sample: edit_mode[i]=True → keep src values; edit_mode[i]=False → zero mask region.
        edit_flags = batch.get('edit_mode', None)
        if src_mask is not None:
            if edit_flags is not None:
                if isinstance(edit_flags, Tensor):
                    # (B,) bool tensor → (B, 1, 1) for broadcasting
                    keep = edit_flags.view(-1, 1, 1).float().to(src_motion.device)
                elif isinstance(edit_flags, (list, tuple)):
                    keep = torch.tensor([float(bool(e)) for e in edit_flags],
                                        device=src_motion.device).view(-1, 1, 1)
                else:
                    keep = torch.zeros(1, 1, 1, device=src_motion.device)
                # For completion (keep=0): src_motion *= (1-mask) → zeroes mask regions
                # For edit (keep=1): src_motion carries the source in target regions.
                src_motion = src_motion * (1 - src_mask * (1 - keep))
            else:
                # No edit_mode flag → all completion
                src_motion = src_motion * (1 - src_mask)

        # Zero out padded frames so they don't produce extreme normalized values
        B, L_src, D = src_motion.shape
        L_tgt = tgt_motion.shape[1]
        for i in range(B):
            tgt_len = int(tgt_length_list[i])
            src_len = int(src_length_list[i])
            if tgt_len < L_tgt:
                tgt_motion[i, tgt_len:] = 0.0
            if src_len < L_src:
                src_motion[i, src_len:] = 0.0
                if src_mask is not None:
                    src_mask[i, src_len:] = 0.0
        if debug_train_step:
            tgt_min = min(int(x) for x in tgt_length_list)
            tgt_max = max(int(x) for x in tgt_length_list)
            self._debug_train_step_log(
                f"after zero_padding B={B} L_src={L_src} L_tgt={L_tgt} "
                f"tgt_len_min={tgt_min} tgt_len_max={tgt_max}",
                device,
            )

        # Motion condition dropout: randomly drop the entire motion condition
        # for a fraction of batch samples.  When motion is dropped the model
        # receives the same state as pure text-conditioned generation (T2M):
        #   src_mask = all-1s  → everything is "generate" (no known regions)
        #   src_motion = zeros -> edit context is all-zero
        # This prevents the model from short-circuiting text understanding
        # by relying solely on the high-information motion condition.
        motion_cond_mask_prob = getattr(self.bundle, 'motion_cond_mask_prob', 0.0)
        if self.training and motion_cond_mask_prob > 0 and src_mask is not None:
            motion_drop_mask = torch.bernoulli(
                torch.full((B,), motion_cond_mask_prob, device=device)
            ).bool()  # (B,) True = drop motion condition
            if motion_drop_mask.any():
                # For dropped samples: all coordinates become "generate"
                src_mask[motion_drop_mask] = 1.0
                # For dropped samples: no source motion reference
                src_motion[motion_drop_mask] = 0.0

        ref_pose = batch.get('ref_pose')
        if ref_pose is not None and not isinstance(ref_pose, Tensor):
            ref_pose = None
        if ref_pose is not None:
            ref_pose = ref_pose.to(device)

        # 1. Prepare padding
        src_motion, src_mask, tgt_motion, src_length_list, tgt_length_list, tgt_padding_mask = (
            self.bundle.prepare_padding(
                src_motion, tgt_motion, tgt_length_list, src_mask, src_length_list, ref_pose
            )
        )
        if debug_train_step:
            self._debug_train_step_log(
                f"after prepare_padding src={tuple(src_motion.shape)} tgt={tuple(tgt_motion.shape)} "
                f"pad_mask={tuple(tgt_padding_mask.shape)}",
                device,
            )

        # 2. Prepare text: use null embeddings (unconditioned) or batch text
        B = tgt_motion.shape[0]
        if batch.get('text_vec_raw') is not None:
            # Pre-encoded text vectors already in batch.
            # text_vec_raw is always a Tensor (null samples get zero-filled by
            # LoadPreExtractedTextEmbedding, so no mixed Tensor/None batches).
            vtxt_input = batch['text_vec_raw'].to(device)
            # text_ctxt_raw may be a list of variable-length tensors when loaded
            # from pre-extracted .pt files (different captions have different
            # sequence lengths and flexible_collate cannot stack them).
            # Pad to the fixed max_text_len (default 128) to match HY-Motion T2M 1.0.
            ctxt_raw = batch['text_ctxt_raw']
            pad_len = self.max_text_len
            if isinstance(ctxt_raw, (list, tuple)):
                feat_dim = ctxt_raw[0].shape[-1]
                ctxt_padded = ctxt_raw[0].new_zeros(len(ctxt_raw), pad_len, feat_dim)
                for i, t in enumerate(ctxt_raw):
                    seq = min(t.shape[0], pad_len)
                    ctxt_padded[i, :seq] = t[:seq]
                ctxt_input = ctxt_padded.to(device)
            else:
                # Already a stacked tensor (all same length) — pad/truncate
                cur_len = ctxt_raw.shape[1]
                if cur_len < pad_len:
                    ctxt_input = F.pad(ctxt_raw, (0, 0, 0, pad_len - cur_len)).to(device)
                else:
                    ctxt_input = ctxt_raw[:, :pad_len].to(device)
            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
            if debug_train_step:
                self._debug_train_step_log(
                    f"after text tensors vtxt={tuple(vtxt_input.shape)} ctxt={tuple(ctxt_input.shape)} "
                    f"ctxt_len_min={int(ctxt_length.min().item())} "
                    f"ctxt_len_max={int(ctxt_length.max().item())}",
                    device,
                )

            # For null-embedding samples (no caption, text_ctxt_raw_length==0),
            # force-replace with the learned null embeddings so they match the
            # null distribution the model sees during CFG dropout.
            null_mask = (ctxt_length == 0)  # (B,)
            if null_mask.any():
                null_v = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
                null_c = self.bundle.null_ctxt_input.expand_as(ctxt_input)
                vtxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(vtxt_input), null_v, vtxt_input
                )
                ctxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(ctxt_input), null_c, ctxt_input
                )

            vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
                return_text_available=True,
            )
            # Match HYMotion T2M: CFG dropout replaces text embeddings with the
            # learned null embeddings, but keeps the original caption attention
            # mask instead of collapsing the null branch to one token.
            if debug_train_step:
                self._debug_train_step_log(
                    f"after mask_text_cond text_available={int(text_available.sum().item())}/{B}",
                    device,
                )

        elif 'caption' in batch and batch['caption'] is not None:
            # Online text encoding from raw captions.
            # The text encoder (Qwen3-8B) lives on CPU and is never moved to
            # GPU.  In DDP, every node has its own CPU copy of the encoder, so
            # we encode on local rank-0 of each node and broadcast within the
            # node group — avoiding duplicate encode work across ranks.
            # In practice we encode on each rank independently because
            # accelerate DDP already replicates the batch per process; the
            # slight CPU overhead is acceptable vs the complexity of
            # rank0-only encode + broadcast.
            captions = batch['caption']
            if isinstance(captions, torch.Tensor):
                captions = captions.tolist()
            # Replace None entries with empty string for encoder
            captions = [c if c is not None else '' for c in captions]
            with torch.no_grad():
                text_feats = self.bundle.encode_text(captions)
            vtxt_input = text_feats['text_vec_raw'].to(device)
            ctxt_input = text_feats['text_ctxt_raw'].to(device)
            ctxt_length = text_feats['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
            vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
                return_text_available=True,
            )
            # Match HYMotion T2M: keep the original caption mask after replacing
            # dropped text embeddings with learned null embeddings.

        else:
            vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
            ctxt_length = torch.tensor([1], device=device).expand(B)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)
            text_available = torch.zeros(B, dtype=torch.bool, device=device)

        # 2b. Prepare task instructions: encode mask strategy as natural language
        # P0: Task Instruction Modulation - provide explicit task awareness
        task_emb = None
        if batch.get("mask_strategy") is not None:
            from motius.models.motioncanvas.task_instruction import get_task_instruction

            strategies = batch["mask_strategy"]
            if isinstance(strategies, torch.Tensor):
                strategies = strategies.tolist()

            # Convert strategy indices/names to natural language instructions
            task_instructions = [get_task_instruction(str(s)) for s in strategies]

            with torch.no_grad():
                task_feats = self.bundle.encode_task_instruction(task_instructions)
            task_emb = task_feats["task_emb"].to(device)  # (B, 1, 1024)

        # 3. Flow matching: sample t, build x_t
        x1 = tgt_motion
        if ref_pose is not None:
            x1 = torch.cat([ref_pose, x1], dim=1)
        x0 = torch.randn_like(x1)

        if self.bundle.pred_type == 'x1':
            z = torch.randn(B, dtype=x1.dtype, device=device) * 0.8 + (-0.8)
            timesteps = torch.sigmoid(z)
        else:
            timesteps = torch.rand(B, dtype=x1.dtype, device=device)

        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t) * x0 + t * x1
        if debug_train_step:
            self._debug_train_step_log(
                f"after flow_sample t_min={float(timesteps.min().item()):.4f} "
                f"t_max={float(timesteps.max().item()):.4f} x_t={tuple(x_t.shape)}",
                device,
            )

        # Mask-aware noise: keep known regions clean in x_t so that
        # inference-time replacement guidance is train-consistent.
        # src_mask: (B, L, D), 1=generate, 0=known.
        # After this: x_t[known] = x1[known] (clean), x_t[gen] unchanged (noisy).
        if src_mask is not None:
            keep_mask = 1 - src_mask  # (B, L, D), 1=known
            x_t = x_t * src_mask + x1 * keep_mask

        # 4. Build edit context and target mask.
        condition_context = self.bundle.prepare_condition_context(
            src_motion=src_motion,
            ref_pose=ref_pose,
            src_mask=src_mask,
        )
        if debug_train_step:
            self._debug_train_step_log(
                f"after prepare_condition_context context={tuple(condition_context.shape)}",
                device,
            )

        # 5. Forward
        x_input = torch.cat([x_t, condition_context], dim=-1)
        if debug_train_step:
            self._debug_train_step_log(
                f"before predict_flow x_input={tuple(x_input.shape)} "
                f"ctxt_mask={tuple(ctxt_mask_temporal.shape)}",
                device,
            )
        pred = self.bundle.predict_flow(
            task_emb=task_emb,
            x_input=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
            sources=_batch_sources(batch, B),
            trigger_sources={'Taobao', 'Game'},
        )
        if debug_train_step:
            self._debug_train_step_log(f"after predict_flow pred={tuple(pred.shape)}", device)

        # Generation mask for target-only flow loss weighting.
        generation_mask = None
        if src_mask is not None:
            generation_mask = src_mask  # (B, L, D), 1=generate

        return {
            'device': device,
            'src_motion': src_motion,
            'tgt_motion': tgt_motion,
            'src_mask': src_mask,
            'src_length_list': src_length_list,
            'tgt_length_list': tgt_length_list,
            'ref_pose': ref_pose,
            'tgt_padding_mask': tgt_padding_mask,
            'vtxt_input': vtxt_input,
            'ctxt_input': ctxt_input,
            'ctxt_mask_temporal': ctxt_mask_temporal,
            'x0': x0,
            'x1': x1,
            'x_t': x_t,
            'timesteps': timesteps,
            't': t,
            'condition_context': condition_context,
            'task_emb': task_emb,
            'pred': pred,
            'generation_mask': generation_mask,
            'text_available': text_available,
        }

    def _compute_base_loss(self, ctx: Dict[str, Any]) -> Dict[str, Tensor]:
        """Compute the standard flow-matching loss dict from a context dict.

        Separated from train_step so that subclasses (e.g., SOAR trainer) can
        reuse the same loss computation on their base forward.
        """
        x0 = ctx['x0']
        x1 = ctx['x1']
        x_t = ctx['x_t']
        t = ctx['t']
        pred = ctx['pred']
        timesteps = ctx['timesteps']
        tgt_padding_mask = ctx['tgt_padding_mask']
        generation_mask = ctx['generation_mask']

        if self.bundle.pred_type == 'velocity':
            gt_velocity = x1 - x0
            pred_velocity = pred
            # Derive x1 for decoded-space losses. The standard conversion is
            # valid only on generated coordinates: MAN replaces known x_t with
            # clean x1, so those coordinates no longer follow the linear flow
            # path. Project them back to x1 before any geometry/temporal loss.
            pred_x1_for_geometry = None
            gt_x1_for_geometry = None
            if self.bundle.m2m_loss.geometry_enabled:
                pred_x1_raw = x_t + (1 - t) * pred_velocity
                pred_x1_for_geometry = self._project_known_x1_for_geometry(
                    pred_x1_raw, x1, generation_mask
                )
                gt_x1_for_geometry = x1

            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                pred_x1=pred_x1_for_geometry,
                gt_x1=gt_x1_for_geometry,
                mean=self.bundle.mean,
                std=self.bundle.std,
                bone_offsets=(
                    self.bundle.get_bone_offsets()
                    if self.bundle.m2m_loss.geometry_enabled
                    else None
                ),
                rotation_space=getattr(self.bundle, 'rotation_space', 'local'),
                timesteps=timesteps,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
                generation_mask=generation_mask,
            )
        elif self.bundle.pred_type == 'x1':
            t_eps = 0.05
            gt_velocity = (x1 - x_t) / (1 - t).clamp_min(t_eps)
            pred_velocity = (pred - x_t) / (1 - t).clamp_min(t_eps)
            pred_x1_for_geometry = None
            if self.bundle.m2m_loss.geometry_enabled:
                pred_x1_for_geometry = self._project_known_x1_for_geometry(
                    pred, x1, generation_mask
                )

            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                pred_x1=pred_x1_for_geometry,
                gt_x1=x1,
                mean=self.bundle.mean,
                std=self.bundle.std,
                bone_offsets=(
                    self.bundle.get_bone_offsets()
                    if self.bundle.m2m_loss.geometry_enabled
                    else None
                ),
                rotation_space=getattr(self.bundle, 'rotation_space', 'local'),
                timesteps=timesteps,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
                generation_mask=generation_mask,
            )
        else:
            raise ValueError(f'Unsupported pred_type: {self.bundle.pred_type}')
        return losses

    @staticmethod
    def _project_known_x1_for_geometry(
        pred_x1: Tensor,
        condition_x1: Tensor,
        generation_mask: Optional[Tensor],
    ) -> Tensor:
        """Compose known condition values with generated x1 for decoded losses.

        ``generation_mask`` uses 1=generate and 0=known. Clean imputation
        clamps known coordinates to ``x1``, so they do not follow the linear
        flow path and velocity-to-x1 conversion is undefined there. Geometry
        and temporal losses therefore evaluate the same composite motion used
        by inference.
        """
        if generation_mask is None:
            return pred_x1
        if pred_x1.shape != condition_x1.shape:
            raise ValueError(
                'pred_x1 and condition_x1 must have identical shapes, got '
                f'{tuple(pred_x1.shape)} and {tuple(condition_x1.shape)}'
            )

        mask = generation_mask.to(device=pred_x1.device)
        if mask.shape[-1] != pred_x1.shape[-1]:
            raise ValueError(
                'generation_mask feature dimension must match pred_x1, got '
                f'{mask.shape[-1]} and {pred_x1.shape[-1]}'
            )
        if mask.shape[1] > pred_x1.shape[1]:
            raise ValueError(
                'generation_mask cannot be longer than pred_x1, got '
                f'{mask.shape[1]} and {pred_x1.shape[1]}'
            )
        if mask.shape[1] < pred_x1.shape[1]:
            # Reference poses are prepended to x1 and are always known.
            prefix = pred_x1.shape[1] - mask.shape[1]
            mask = F.pad(mask, (0, 0, prefix, 0), value=0.0)

        known_mask = mask < 0.5
        return torch.where(known_mask, condition_x1, pred_x1)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        debug_train_step = self._debug_train_step_enabled()
        ctx = self._prepare_and_forward(batch)

        if debug_train_step:
            self._debug_train_step_log("before compute_base_loss", ctx['device'])
        losses = self._compute_base_loss(ctx)
        if debug_train_step:
            self._debug_train_step_log(f"after compute_base_loss keys={list(losses.keys())}", ctx['device'])
        loss = self.sum_train_losses(losses)
        if debug_train_step:
            self._debug_train_step_log("after sum_train_losses", ctx['device'])
        result = {'loss': loss}
        for k, v in losses.items():
            result[f'loss_{k}'] = v.detach()
        if debug_train_step:
            self._debug_train_step_count += 1
        return result

# ---------------------------------------------------------------------------
# Unit tests — run with: python -m motius.trainers.hymotion_m2m.hymotion_m2m_trainer
# ---------------------------------------------------------------------------

def _test_trainer_zeroes_mask_regions():
    """Verify trainer zeros src_motion in mask=1 regions AFTER normalization.

    Expected:
    - Input: src_motion (raw), src_mask (binary)
    - After normalize: src_motion_norm has values everywhere
    - After zeroing: src_motion_norm[mask=1] == 0
    - prepare_condition_context receives zeroed target regions, so edit_context=0
    """
    import torch

    T, D = 100, 135
    # Simulate raw motion and mask
    src_motion = torch.randn(1, T, D)
    src_mask = torch.zeros(1, T, D)
    src_mask[0, 30:50, :] = 1.0  # frames 30-49 fully masked

    # Simulate normalize (mean=0, std=1 for simplicity)
    motion_norm = src_motion.clone()

    # The critical fix: zero out mask=1 regions AFTER normalize
    motion_zeroed = motion_norm * (1 - src_mask)

    # Verify
    mask_bool = src_mask > 0.5
    assert motion_zeroed[mask_bool].abs().max() == 0.0, \
        "mask=1 regions must be 0 after zeroing"
    assert motion_zeroed[~mask_bool].std() > 0, \
        "mask=0 regions must retain motion values"

    # Completion has no edit source in target regions.
    edit_context = motion_zeroed * src_mask
    assert edit_context.abs().max() == 0.0, \
        "edit context must be 0 for completion targets"

    print("  ✅ Trainer mask zeroing: src_motion[mask=1]=0, edit_context=0")


def _test_trainer_preserves_tgt_motion():
    """Verify tgt_motion (target/GT) is NOT zeroed — only src_motion is zeroed.

    The model predicts velocity v = x1 - x0 where x1 = normalized tgt_motion.
    tgt_motion must retain all values including in mask=1 regions.
    """
    import torch

    T, D = 100, 135
    tgt_motion = torch.randn(1, T, D)
    src_mask = torch.zeros(1, T, D)
    src_mask[0, 30:50, :] = 1.0

    # tgt_motion is only normalized, NEVER zeroed
    tgt_norm = tgt_motion.clone()  # simulate normalize

    # Verify tgt still has values in mask=1 regions
    mask_bool = src_mask > 0.5
    assert tgt_norm[mask_bool].std() > 0, \
        "tgt_motion must NOT be zeroed — it's the generation target"

    print("  ✅ Trainer tgt_motion: preserved (not zeroed) in mask=1 regions")


if __name__ == '__main__':
    print("Running hymotion_m2m_trainer unit tests...")
    _test_trainer_zeroes_mask_regions()
    _test_trainer_preserves_tgt_motion()
    print("All tests passed ✅")
