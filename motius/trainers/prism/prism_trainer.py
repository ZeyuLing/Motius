"""PRISM trainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class PrismTrainer(BaseTrainer):
    """Trainer for PRISM flow-matching motion generation."""

    def __init__(
        self,
        bundle,
        condition_num_frames: Union[int, List[int]] = 1,
        frame_condition_rate: float = 0.1,
        prompt_drop_rate: float = 0.1,
        max_text_length: int = 128,
        val_prompts: Optional[List[str]] = None,
        num_val_inference_steps: int = 10,
        guidance_scale: float = 5.0,
        translation_loss_weight: float = 0.5,
        null_embedding_path: Optional[str] = None,
        use_fp16_autocast: bool = False,
        log_channel_loss: bool = False,
        **kwargs,
    ):
        super().__init__(bundle)
        self.condition_num_frames = condition_num_frames
        self.frame_condition_rate = frame_condition_rate
        self.prompt_drop_rate = prompt_drop_rate
        self.max_text_length = max_text_length
        self.val_prompts = val_prompts or ['a person walking forward']
        self.num_val_inference_steps = num_val_inference_steps
        self.guidance_scale = guidance_scale
        self.translation_loss_weight = translation_loss_weight
        self._null_embedding_path = null_embedding_path
        self.use_fp16_autocast = use_fp16_autocast
        self.log_channel_loss = log_channel_loss

    def _load_null_t5_embedding(self):
        """Load the pre-extracted null embedding (empty string '') for prompt dropout.

        This is loaded lazily on first use and cached. The null embedding is what
        T5 produces for '' input: a one-token non-zero EOS embedding.
        """
        if hasattr(self, '_null_text_embed'):
            return
        null_path = getattr(self, '_null_embedding_path', None)
        if null_path is None:
            raise ValueError(
                'Cached-feature prompt dropout requires null_embedding_path. '
                'Set it to the empty-prompt feature generated with your text encoder.'
            )
        try:
            data = torch.load(null_path, map_location='cpu', weights_only=True)
        except TypeError:
            data = torch.load(null_path, map_location='cpu')
        emb = data.get('embedding', data.get('t5_text_embeds'))
        if emb is None:
            raise KeyError(
                'Null text feature must contain embedding or t5_text_embeds'
            )
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]
        feature_mask = data.get('t5_text_mask')
        if feature_mask is not None:
            if feature_mask.ndim == 2 and feature_mask.shape[0] == 1:
                feature_mask = feature_mask[0]
            seq_len = int(feature_mask.sum().item())
        else:
            seq_len = int(data.get('seq_len', emb.shape[0]))
        emb = emb[: self.max_text_length]
        seq_len = min(seq_len, self.max_text_length)
        # Pad to max_text_length
        if emb.size(0) < self.max_text_length:
            pad = torch.zeros(
                self.max_text_length - emb.size(0), emb.size(1), dtype=emb.dtype
            )
            emb = torch.cat([emb, pad], dim=0)
        # Build mask
        mask = torch.zeros(self.max_text_length, dtype=torch.long)
        mask[:seq_len] = 1
        self._null_text_embed = emb   # [max_text_length, 4096] bf16
        self._null_text_mask = mask   # [max_text_length] int64

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        motion = batch['motion']
        captions = batch.get('caption')
        num_frames = batch.get('num_frames')

        # ---- VAE encoding + data prep (fp32, no autocast) ----
        latents = self.bundle.encode_motion(motion)
        batch_size, _, latent_frames, latent_joints = latents.shape

        padding_mask = self.bundle.create_padding_mask(
            num_frames=num_frames,
            batch_size=batch_size,
            latent_frames=latent_frames,
            latent_joints=latent_joints,
            device=latents.device,
        )

        # Use pre-extracted T5 features if available, otherwise encode online
        if 't5_text_embeds' in batch:
            transformer_dtype = next(self.bundle.transformer.parameters()).dtype
            text_states = batch['t5_text_embeds'].to(
                device=latents.device, dtype=transformer_dtype
            )
            text_mask = batch['t5_text_mask'].to(device=latents.device)
            # Apply prompt dropout: replace with null (empty-string) embedding
            if self.prompt_drop_rate > 0:
                self._load_null_t5_embedding()
                drop_mask = torch.rand(batch_size, device=latents.device) < self.prompt_drop_rate
                if drop_mask.any():
                    null_emb = self._null_text_embed.to(
                        device=latents.device, dtype=transformer_dtype
                    )
                    null_mask = self._null_text_mask.to(device=latents.device)
                    text_states[drop_mask] = null_emb
                    text_mask[drop_mask] = null_mask
        else:
            text_states, text_mask = self.bundle.encode_prompt_with_mask(
                captions,
                max_sequence_length=self.max_text_length,
                prompt_drop_rate=self.prompt_drop_rate,
                dtype=next(self.bundle.transformer.parameters()).dtype,
            )
        condition_frame_mask_vae = self.bundle.create_condition_mask(
            latents,
            frame_condition_rate=self.frame_condition_rate,
            condition_num_frames=self.condition_num_frames,
            num_frames=num_frames,
        )

        step_indices = torch.randint(
            0,
            len(self.bundle.scheduler.timesteps),
            (batch_size,),
            device=latents.device,
        )
        scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
        timesteps = scheduler_timesteps[step_indices]

        noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
        noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
        # Access patch_size from transformer config (unwrap FSDP/DDP wrapper)
        transformer_module = getattr(
            self.bundle.transformer, 'module', self.bundle.transformer
        )
        timesteps = self.bundle.create_sequence_ts(
            timesteps,
            condition_frame_mask_vae,
            transformer_module.config.patch_size,
        )
        transformer_dtype = next(self.bundle.transformer.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=transformer_dtype)

        # ---- Transformer forward (optional fp16 autocast) ----
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.use_fp16_autocast
            else torch.cuda.amp.autocast(enabled=False)
        )
        with autocast_ctx:
            model_pred = self.bundle.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=text_states,
                timestep=timesteps,
                hidden_states_mask=padding_mask if num_frames is not None else None,
                # No text cross-attention mask: matches the official Wan
                # implementation (text padded with zeros, context_lens=None).
                encoder_hidden_states_mask=None,
            )

        # Loss computation in fp32 (outside autocast for numerical stability)
        model_pred = model_pred.float()
        mse = F.mse_loss(model_pred, targets.float(), reduction='none')
        # mse shape: [B, C, T', J] where J=23 (token 0=translation, 1-22=rotation)
        condition_mask = condition_frame_mask_vae.expand_as(mse).float()
        padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
        full_mask = condition_mask * padding_mask

        # Separate translation (J=0) and rotation (J=1:) to prevent
        # translation loss dilution (1/23 ~= 4.3% vs 22/23 ~= 95.7%).
        mse_transl = mse[:, :, :, :1]           # [B, C, T', 1]
        mask_transl = full_mask[:, :, :, :1]
        loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

        mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
        mask_rot = full_mask[:, :, :, 1:]
        loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

        w_t = self.translation_loss_weight
        loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
        if not torch.isfinite(loss):
            raise FloatingPointError(
                'Non-finite PRISM loss: '
                f'loss={loss.detach().item()}, '
                f'loss_transl={loss_transl.detach().item()}, '
                f'loss_rot={loss_rot.detach().item()}'
            )
        output = {
            'loss': loss,
            'loss_flow': loss.detach(),
            'loss_transl': loss_transl.detach(),
            'loss_rot': loss_rot.detach(),
        }
        if self.log_channel_loss:
            channel_denom = full_mask.sum(dim=(0, 2, 3)).clamp_min(1e-6)
            token_denom = full_mask.sum(dim=(0, 1, 2)).clamp_min(1e-6)
            loss_by_channel = (mse * full_mask).sum(dim=(0, 2, 3)) / channel_denom
            loss_by_token = (mse * full_mask).sum(dim=(0, 1, 2)) / token_denom
            transl_by_channel = (
                (mse_transl * mask_transl).sum(dim=(0, 2, 3))
                / mask_transl.sum(dim=(0, 2, 3)).clamp_min(1e-6)
            )
            rot_by_channel = (
                (mse_rot * mask_rot).sum(dim=(0, 2, 3))
                / mask_rot.sum(dim=(0, 2, 3)).clamp_min(1e-6)
            )

            for idx, value in enumerate(loss_by_channel):
                output[f'loss_ch{idx:02d}'] = value.detach()
            for idx, value in enumerate(transl_by_channel):
                output[f'loss_transl_ch{idx:02d}'] = value.detach()
            for idx, value in enumerate(rot_by_channel):
                output[f'loss_rot_ch{idx:02d}'] = value.detach()
            for idx, value in enumerate(loss_by_token):
                output[f'loss_tok{idx:02d}'] = value.detach()

            topk = min(4, loss_by_channel.numel())
            top_values, top_indices = torch.topk(loss_by_channel, k=topk)
            for rank, (idx, value) in enumerate(zip(top_indices, top_values)):
                output[f'loss_ch_top{rank}_idx'] = idx.float().detach()
                output[f'loss_ch_top{rank}'] = value.detach()
        latent_stats = getattr(self.bundle, '_last_latent_norm_stats', None)
        if latent_stats:
            output.update({
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in latent_stats.items()
            })
        return output

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from motius.pipelines.prism import PRISMPipeline

        pipeline = PRISMPipeline(bundle=self.bundle)
        preds = pipeline.text_to_motion(
            caption=self.val_prompts[0],
            num_frames=33,
            num_inference_steps=self.num_val_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        return {'preds': preds, 'prompts': self.val_prompts}
