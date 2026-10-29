"""Native ViMoGen text conditioning, CFG and flow-matching inference."""
from __future__ import annotations
from typing import List,Optional,Sequence
import torch

class ViMoGenInference:
    @property
    def text_tokenizer(self):
        if getattr(self,'_text_tokenizer',None) is None:
            from .tokenization import HuggingfaceTokenizer
            self._text_tokenizer=HuggingfaceTokenizer(name=str(self.bundle.wan_dir/'google/umt5-xxl'),seq_len=self.bundle.text_len,clean='whitespace')
        return self._text_tokenizer

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion*self.bundle.std.to(device=motion.device)+self.bundle.mean.to(device=motion.device)

    def clamp_length(self, n_frames: int) -> int:
        n_frames = int(n_frames)
        return max(self.bundle.min_length, min(self.bundle.max_length, n_frames))

    def _collate_contexts(self, contexts: List[torch.Tensor], min_len: int) -> torch.Tensor:
        max_len = max([min_len] + [int(x.shape[0]) for x in contexts])
        hidden = int(contexts[0].shape[-1])
        out = torch.zeros(len(contexts), max_len, hidden, device=self.device, dtype=self.bundle.text_dtype)
        for i, ctx in enumerate(contexts):
            ctx = ctx.to(device=self.device, dtype=self.bundle.text_dtype)
            out[i, :ctx.shape[0]] = ctx
        return out

    def encode_texts(self, captions: Sequence[str]) -> torch.Tensor:
        self.bundle.load_text_encoder()
        ids, mask = self.text_tokenizer(list(captions), return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        lengths = mask.gt(0).sum(dim=1).long()
        context = self.bundle.encode_text_tokens(ids, mask)
        return self._collate_contexts([value[:length] for value, length in zip(context, lengths)], min_len=self.bundle.prompt_emb_null.shape[1])

    def null_context(self, batch_size: int, context_len: int) -> torch.Tensor:
        null = self.bundle.prompt_emb_null.to(device=self.device, dtype=self.bundle.text_dtype)
        if null.shape[1] < context_len:
            pad = torch.zeros(null.shape[0], context_len - null.shape[1], null.shape[2], device=self.device, dtype=self.bundle.text_dtype)
            null = torch.cat([null, pad], dim=1)
        return null[:, :context_len].repeat(batch_size, 1, 1)

    @torch.no_grad()
    def generate_motion276_from_embeddings(self, prompt_emb: torch.Tensor, lengths: Sequence[int], seed: Optional[int]=None, cfg_scale: Optional[float]=None, num_inference_steps: Optional[int]=None, denoising_strength: Optional[float]=None, show_progress: bool=False) -> List[torch.Tensor]:
        if prompt_emb.ndim != 3:
            raise ValueError(f'prompt_emb must be (B,L,C), got {tuple(prompt_emb.shape)}')
        if int(prompt_emb.shape[0]) != len(lengths):
            raise ValueError('prompt_emb batch and lengths must have equal length')
        if self.device.type != 'cuda':
            raise RuntimeError('ViMoGen generation requires CUDA.')
        from tqdm import tqdm
        from motius.diffusion.vimogen import FlowMatchScheduler
        from .postprocessing import smooth_motion_rep
        lengths = [self.clamp_length(x) for x in lengths]
        batch_size = len(lengths)
        max_len = max(lengths)
        cfg_scale = float(self.bundle.cfg_scale if cfg_scale is None else cfg_scale)
        num_inference_steps = int(self.bundle.num_inference_steps if num_inference_steps is None else num_inference_steps)
        denoising_strength = float(self.bundle.denoising_strength if denoising_strength is None else denoising_strength)
        prompt_emb = prompt_emb.to(device=self.device, dtype=self.bundle.text_dtype)
        prompt_emb_null = self.null_context(batch_size, prompt_emb.shape[1])
        latents_mask = torch.zeros(batch_size, max_len, device=self.device, dtype=self.bundle.dtype)
        for i, length in enumerate(lengths):
            latents_mask[i, :length] = 1
        ref_latents = torch.zeros(batch_size, max_len, self.bundle.ref_motion_dim, device=self.device, dtype=self.bundle.dtype)
        ref_latents_mask = latents_mask.clone()
        attend_to_text_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        generator = torch.Generator(device=self.device)
        if seed is None:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        generator.manual_seed(int(seed))
        xt = torch.randn(batch_size, max_len, self.bundle.motion_dim, generator=generator, device=self.device, dtype=self.bundle.dtype)
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps, training=False, denoising_strength=denoising_strength)
        timesteps = scheduler.timesteps.to(self.device)
        latents_mask_input = torch.cat([latents_mask] * 2, dim=0)
        ref_latents_input = torch.cat([ref_latents, torch.zeros_like(ref_latents)], dim=0)
        ref_latents_mask_input = torch.cat([ref_latents_mask] * 2, dim=0)
        attend_to_text_mask_input = torch.cat([attend_to_text_mask] * 2, dim=0)
        context_input = torch.cat([prompt_emb, prompt_emb_null], dim=0)
        self.bundle.model.eval()
        iterator = tqdm(timesteps, desc='ViMoGen', disable=not show_progress)
        for timestep in iterator:
            with torch.amp.autocast(dtype=self.bundle.dtype, device_type=self.device.type):
                latent_model_input = torch.cat([xt] * 2, dim=0)
                noise_pred = self.bundle.model(x=latent_model_input, timestep=timestep.unsqueeze(0), context=context_input, x_mask=latents_mask_input, ref_motion=ref_latents_input, ref_motion_mask=ref_latents_mask_input, use_gradient_checkpointing=False, attend_to_text_mask=attend_to_text_mask_input)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                xt = scheduler.step(noise_pred, timestep, xt)
        # The source smooths the complete padded batch, then restores the
        # inference dtype before physical-scale export and valid-frame slicing.
        for i in range(batch_size):
            xt[i]=smooth_motion_rep(xt[i],kernel_size=5,sigma=1.0)
        xt=xt.to(dtype=self.bundle.dtype)
        outputs=[self.denormalize(xt[i,:length]).float().detach().cpu() for i,length in enumerate(lengths)]
        return outputs

    @torch.no_grad()
    def generate_motion276(self, captions: Sequence[str], lengths: Sequence[int], seed: Optional[int]=None, cfg_scale: Optional[float]=None, num_inference_steps: Optional[int]=None, denoising_strength: Optional[float]=None, show_progress: bool=False) -> List[torch.Tensor]:
        if len(captions) != len(lengths):
            raise ValueError('captions and lengths must have equal length')
        prompt_emb = self.encode_texts(captions)
        return self.generate_motion276_from_embeddings(prompt_emb=prompt_emb, lengths=lengths, seed=seed, cfg_scale=cfg_scale, num_inference_steps=num_inference_steps, denoising_strength=denoising_strength, show_progress=show_progress)
