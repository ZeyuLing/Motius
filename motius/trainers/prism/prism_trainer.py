"""PRISM trainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer

def _get_sigmas(scheduler, timesteps: torch.Tensor, n_dim: int) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=torch.float32)
    schedule = scheduler.timesteps.to(device=timesteps.device)
    indices = [(schedule == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma



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
        self.bundle.load_model()
        from diffusers import FlowMatchEulerDiscreteScheduler
        from motius.motion.representation.prism import PRISMMotionProcessor
        config = self.bundle.artifact_config
        root = self.bundle.artifact_dir
        components = config.get("components", {})
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / components.get("scheduler", "scheduler"))
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.processor = PRISMMotionProcessor(root / config.get("stats_file", "motion_stats.json"))
        vae = self.bundle.vae
        parameter = next(vae.parameters())
        self.register_buffer("latents_mean", torch.tensor(vae.config.latents_mean, dtype=parameter.dtype,
            device=parameter.device).view(1, vae.config.z_dim, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(vae.config.latents_std, dtype=parameter.dtype,
            device=parameter.device).view(1, vae.config.z_dim, 1, 1), persistent=False)
        self.tokenizer = None

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
        latents = self.encode_motion(motion)
        batch_size, _, latent_frames, latent_joints = latents.shape

        padding_mask = self.create_padding_mask(
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
            text_states, text_mask = self.encode_prompt_with_mask(
                captions,
                max_sequence_length=self.max_text_length,
                prompt_drop_rate=self.prompt_drop_rate,
                dtype=next(self.bundle.transformer.parameters()).dtype,
            )
        condition_frame_mask_vae = self.create_condition_mask(
            latents,
            frame_condition_rate=self.frame_condition_rate,
            condition_num_frames=self.condition_num_frames,
            num_frames=num_frames,
        )

        step_indices = torch.randint(
            0,
            len(self.scheduler.timesteps),
            (batch_size,),
            device=latents.device,
        )
        scheduler_timesteps = self.scheduler.timesteps.to(device=latents.device)
        timesteps = scheduler_timesteps[step_indices]

        noisy_latents, targets = self.add_flow_noise(latents, timesteps)
        noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
        # Access patch_size from transformer config (unwrap FSDP/DDP wrapper)
        transformer_module = getattr(
            self.bundle.transformer, 'module', self.bundle.transformer
        )
        timesteps = self.create_sequence_ts(
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
            model_pred = self.bundle.predict_flow(
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
        latent_stats = getattr(self, '_last_latent_norm_stats', None)
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


    @torch.no_grad()
    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize a 138D motion and encode it with the frozen fp32 VAE."""
        from motius.models.prism.gaussian_distribution import (
            DiagonalGaussianDistributionNd,
        )

        self.bundle.load_model()
        motion = motion.float()
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        motion = self.processor.normalize(motion)
        motion = motion.reshape(*motion.shape[:2], 23, 6)
        with torch.autocast(motion.device.type, enabled=False):
            parameters = self.bundle.encode_motion_parameters(motion.float())
        posterior = DiagonalGaussianDistributionNd(parameters)
        latents = (
            posterior.sample()
            if self.bundle.latent_sample_method == "sample"
            else posterior.mode()
        )
        latents = (
            latents - self.latents_mean.to(latents)
        ) / self.latents_std.to(latents)
        stats = latents.detach().float()
        channel_mean = stats.mean(dim=(0, 2, 3))
        channel_std = stats.std(dim=(0, 2, 3), unbiased=False)
        self._last_latent_norm_stats = {
            "latent_norm_mean": stats.mean(),
            "latent_norm_std": stats.std(unbiased=False),
            "latent_norm_channel_mean_abs_max": channel_mean.abs().max(),
            "latent_norm_channel_std_min": channel_std.min(),
            "latent_norm_channel_std_max": channel_std.max(),
        }
        return latents

    @torch.no_grad()
    def encode_prompt_with_mask(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 128,
        prompt_drop_rate: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.bundle.load_model()
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            components = self.bundle.artifact_config.get("components", {})
            self.tokenizer = AutoTokenizer.from_pretrained(self.bundle.artifact_dir / components.get("tokenizer", "tokenizer"))
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if prompt_drop_rate > 0:
            prompts = [
                "" if torch.rand(()).item() < prompt_drop_rate else value
                for value in prompts
            ]
        device = next(self.bundle.text_encoder.parameters()).device
        dtype = dtype or next(self.bundle.text_encoder.parameters()).dtype
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        attention_mask = tokens.attention_mask.to(device)
        encoded = self.bundle.encode_text_tokens(tokens.input_ids.to(device), attention_mask)
        return encoded.to(dtype=dtype), attention_mask

    def create_padding_mask(
        self,
        num_frames: Optional[torch.Tensor],
        batch_size: int,
        latent_frames: int,
        latent_joints: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_frames is None:
            return torch.ones(batch_size, latent_frames, latent_joints, device=device)
        lengths = torch.as_tensor(num_frames, device=device).long()
        scale = int(self.bundle.vae.config.scale_factor_temporal)
        lengths = ((lengths + scale - 1) // scale).clamp(0, latent_frames)
        valid = torch.arange(latent_frames, device=device)[None] < lengths[:, None]
        return valid.unsqueeze(-1).expand(-1, -1, latent_joints).float()

    def create_condition_mask(
        self,
        latents: torch.Tensor,
        frame_condition_rate: float = 0.1,
        condition_num_frames: Union[int, List[int]] = 1,
        num_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, latent_frames, latent_joints = latents.shape
        device = latents.device
        mask = torch.ones(
            batch_size, 1, latent_frames, latent_joints,
            dtype=torch.bool,
            device=device,
        )
        if frame_condition_rate <= 0:
            return mask
        candidates = (
            [condition_num_frames]
            if isinstance(condition_num_frames, int)
            else list(condition_num_frames)
        )
        selected = torch.tensor(candidates, device=device)[
            torch.randint(len(candidates), (batch_size,), device=device)
        ]
        scale = int(self.bundle.vae.config.scale_factor_temporal)
        selected = ((selected + scale - 1) // scale).clamp(0, latent_frames)
        selected *= (torch.rand(batch_size, device=device) < frame_condition_rate)
        frames = torch.arange(latent_frames, device=device)[None]
        condition = frames < selected[:, None]
        mask = (~condition).unsqueeze(1).unsqueeze(-1).expand_as(mask).clone()
        if num_frames is not None:
            valid = torch.as_tensor(num_frames, device=device).long()
            valid = ((valid + scale - 1) // scale).clamp(0, latent_frames)
            padded = frames >= valid[:, None]
            mask |= padded.unsqueeze(1).unsqueeze(-1).expand_as(mask)
        return mask

    @staticmethod
    def create_sequence_ts(
        timesteps: torch.Tensor,
        condition_mask: torch.Tensor,
        patch_size=(1, 1),
    ) -> torch.Tensor:
        patch_t, patch_j = patch_size
        sampled_mask = condition_mask[:, 0, ::patch_t, ::patch_j]
        expanded = timesteps[:, None, None].expand_as(sampled_mask)
        return torch.where(sampled_mask, expanded, 0).flatten(1)

    def add_flow_noise(
        self, latents: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(latents)
        sigmas = _get_sigmas(self.scheduler, timesteps, latents.ndim).to(latents)
        return (1 - sigmas) * latents + sigmas * noise, noise - latents
