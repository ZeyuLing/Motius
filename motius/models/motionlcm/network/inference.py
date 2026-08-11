"""MotionLCM text-to-motion sampling (faithful to ``mld.MLD``).

Ports the inference math from ``mld/models/modeltype/mld.py`` (``_diffusion_reverse``
+ ``t2m_eval``) for the **text-to-motion** path, without the training / metric /
ControlNet / data-module machinery:

* the latent consistency model (``denoiser.time_cond_proj_dim is not None``)
  uses *distilled* classifier-free guidance — the guidance scale is folded into
  ``timestep_cond`` via :func:`get_guidance_scale_embedding` and there is **no**
  second unconditional forward pass (so ``do_classifier_free_guidance`` is
  ``False``);
* the diffusers ``LCMScheduler`` performs the few-step (default 1) consistency
  sampling in the MLD VAE latent space;
* ``MldVae.decode`` maps the sampled latent ``z`` to the 263-dim feature.

A standard (non-distilled) diffusion denoiser path with explicit CFG is also
supported for completeness. Self-contained;.
"""

from typing import List, Optional, Sequence

import torch

from .utils import get_guidance_scale_embedding


@torch.no_grad()
def lcm_reverse_diffusion(
    denoiser,
    scheduler,
    text_emb: torch.Tensor,
    latent_size: int,
    latent_code_dim: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> torch.Tensor:
    """Run the latent reverse process and return ``z`` of shape ``(L, B, C)``.

    Mirrors ``MLD._diffusion_reverse`` for the text-only path. ``text_emb`` is
    ``(B, num_tokens, text_encoded_dim)`` (already the conditional embedding;
    no unconditional rows). When the denoiser is an LCM
    (``time_cond_proj_dim is not None``) guidance is applied through
    ``timestep_cond``; otherwise classifier-free guidance duplicates the text
    embedding (the caller must then pass ``[uncond; cond]``).
    """
    device = text_emb.device
    is_lcm = denoiser.time_cond_proj_dim is not None
    do_cfg = (guidance_scale > 1) and not is_lcm

    bsz = text_emb.shape[0]
    if do_cfg:
        bsz = bsz // 2

    latents = torch.randn(
        (bsz, latent_size, latent_code_dim), device=device, dtype=torch.float)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps.to(device)

    timestep_cond = None
    if is_lcm:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(latents.shape[0])
        timestep_cond = get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=denoiser.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    for t in timesteps:
        latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = denoiser(
            sample=latent_model_input,
            timestep=t,
            timestep_cond=timestep_cond,
            encoder_hidden_states=text_emb,
        )

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # [B, L, C] -> [L, B, C]
    latents = latents.permute(1, 0, 2)
    return latents


@torch.no_grad()
def generate_motion(
    text_encoder,
    vae,
    denoiser,
    scheduler,
    captions: Sequence[str],
    lengths: Sequence[int],
    guidance_scale: float = 7.5,
    num_inference_steps: int = 1,
) -> List[torch.Tensor]:
    """Text -> latent consistency sampling -> VAE decode -> 263-dim features.

    Returns a list of ``B`` tensors, each ``(T_i, 263)`` in **normalized**
    HumanML3D space (the caller de-normalises with ``Mean`` / ``Std``).
    """
    captions = list(captions)
    lengths = [int(x) for x in lengths]
    is_lcm = denoiser.time_cond_proj_dim is not None
    do_cfg = (guidance_scale > 1) and not is_lcm

    texts = (["" ] * len(captions) + captions) if do_cfg else captions
    text_emb = text_encoder(texts)

    latent_size = vae.latent_size
    latent_code_dim = vae.latent_code_dim

    z = lcm_reverse_diffusion(
        denoiser, scheduler, text_emb,
        latent_size=latent_size,
        latent_code_dim=latent_code_dim,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    feats = vae.decode(z, lengths)  # (B, T, 263)
    return [feats[i, : lengths[i]] for i in range(len(captions))]
