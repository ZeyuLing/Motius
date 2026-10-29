"""UniMuMo delayed-token sampling and classifier-free guidance."""
from __future__ import annotations
import torch
from motius.models.unimumo.generator import UniMuMoGenerator
from motius.motion.conditioning.unimumo_pattern import DelayedPattern

def _sample_logits(logits: torch.Tensor, *, temperature: float, top_k: int, generator: torch.Generator | None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits.float() / float(temperature), dim=-1)
    if top_k > 0:
        top_k = min(int(top_k), probabilities.shape[-1])
        top_values = torch.topk(probabilities, top_k, dim=-1).values
        threshold = top_values[..., -1, None]
        probabilities = probabilities * (probabilities >= threshold)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator).reshape(*probabilities.shape[:-1])

@torch.inference_mode()
def generate_parallel(model: UniMuMoGenerator, *, condition: torch.Tensor, condition_mask: torch.Tensor, unconditional_condition: torch.Tensor, unconditional_mask: torch.Tensor, timesteps: int, music_codes: torch.Tensor | None=None, motion_codes: torch.Tensor | None=None, guidance_scale: float=4.0, temperature: float=1.0, top_k: int=250, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    if music_codes is not None and motion_codes is not None:
        raise ValueError('At most one modality may be provided')
    batch = condition.shape[0]
    shape = (batch, model.num_codebooks, int(timesteps))
    device = condition.device
    unknown_token = -1
    music_dense = music_codes.to(device=device, dtype=torch.long) if music_codes is not None else torch.full(shape, unknown_token, device=device, dtype=torch.long)
    motion_dense = motion_codes.to(device=device, dtype=torch.long) if motion_codes is not None else torch.full(shape, unknown_token, device=device, dtype=torch.long)
    if music_dense.shape != shape or motion_dense.shape != shape:
        raise ValueError(f'provided codes must have shape {shape}')
    pattern = DelayedPattern(timesteps, tuple(range(model.num_codebooks)))
    music_sequence, music_valid = pattern.build(music_dense, special_token=model.special_token_id)
    motion_sequence, motion_valid = pattern.build(motion_dense, special_token=model.special_token_id)
    cfg_condition = torch.cat((condition, unconditional_condition), dim=0)
    cfg_mask = torch.cat((condition_mask, unconditional_mask), dim=0)
    for offset in range(1, pattern.sequence_length):
        current_music = music_sequence[..., :offset]
        current_motion = motion_sequence[..., :offset]
        if (current_music < 0).any() or (current_motion < 0).any():
            raise RuntimeError('generation reached an unfilled prefix token')
        duplicated_music = torch.cat((current_music, current_music), dim=0)
        duplicated_motion = torch.cat((current_motion, current_motion), dim=0)
        music_logits, motion_logits = model(duplicated_music, duplicated_motion, condition=cfg_condition, condition_mask=cfg_mask)
        music_cond, music_uncond = music_logits[:, :, -1].chunk(2, dim=0)
        motion_cond, motion_uncond = motion_logits[:, :, -1].chunk(2, dim=0)
        music_logits = music_uncond + guidance_scale * (music_cond - music_uncond)
        motion_logits = motion_uncond + guidance_scale * (motion_cond - motion_uncond)
        next_music = _sample_logits(music_logits, temperature=temperature, top_k=top_k, generator=generator)
        next_motion = _sample_logits(motion_logits, temperature=temperature, top_k=top_k, generator=generator)
        music_mask = music_valid[:, offset][None].expand(batch, -1)
        motion_mask = motion_valid[:, offset][None].expand(batch, -1)
        next_music = torch.where(music_mask, next_music, torch.full_like(next_music, model.special_token_id))
        next_motion = torch.where(motion_mask, next_motion, torch.full_like(next_motion, model.special_token_id))
        if music_codes is None:
            music_sequence[..., offset] = torch.where(music_sequence[..., offset] == unknown_token, next_music, music_sequence[..., offset])
        if motion_codes is None:
            motion_sequence[..., offset] = torch.where(motion_sequence[..., offset] == unknown_token, next_motion, motion_sequence[..., offset])
    music_output = pattern.revert(music_sequence)
    motion_output = pattern.revert(motion_sequence)
    if (music_output < 0).any() or (motion_output < 0).any():
        raise RuntimeError('generation returned unfilled tokens')
    return (music_output, motion_output)
