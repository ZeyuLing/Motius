"""Original two-forward AR and streaming objectives, outside the neural network."""

import torch
from functools import lru_cache
from motius.diffusion.motionstreamer import create_diffusion


@lru_cache(maxsize=1)
def training_schedule():
    return create_diffusion(timestep_respacing="", noise_schedule="cosine")


def diffusion_loss(head, target, z, mask=None, *, schedule=None):
    schedule = training_schedule() if schedule is None else schedule
    timesteps = torch.randint(
        0, schedule.num_timesteps, (target.shape[0],), device=target.device
    )
    values = schedule.training_losses(head.net, target, timesteps, dict(c=z))
    loss = values["loss"]
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum()
    return loss.mean(), values["pred_xstart"]


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):

    step = torch.tensor(step, dtype=torch.float32)
    total_steps = torch.tensor(total_steps, dtype=torch.float32)

    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor


def replace_with_pred(latents, pred_xstart, step, total_steps):

    decay_factor = cosine_decay(step, total_steps).to(latents.device)

    b, l, d = latents.shape
    num_replace = int(l * decay_factor)

    replace_indices = torch.randperm(l)[:num_replace]

    replace_mask = torch.zeros(b, l, dtype=torch.bool).to(latents.device)
    replace_mask[:, replace_indices] = 1

    updated_latents = latents.clone()
    updated_latents[replace_mask] = pred_xstart[replace_mask]

    return updated_latents


def forward_loss_withmask_2_forward(
    latents,
    trans,
    m_lens,
    feat_text,
    step,
    total_steps,
    diffmlps_batch_mul=4,
    schedule=None,
):
    """z: condition; latents: gt"""
    conditions = trans(latents, feat_text)
    conditions = conditions.contiguous()
    z = conditions[:, :-1, :]
    b, l, d = latents.shape
    mask = lengths_to_mask(m_lens, l)
    mask = mask.reshape(b * l).repeat(diffmlps_batch_mul)
    target = latents.clone().detach()
    target = target.reshape(b * l, -1)
    z = z.reshape(b * l, -1)
    with torch.no_grad():
        loss, pred_xstart = diffusion_loss(
            getattr(trans, "module", trans).diff_loss,
            target=target,
            z=z,
            schedule=schedule,
        )
    pred_xstart = pred_xstart.clone().detach()
    pred_xstart = pred_xstart.reshape(b, l, -1)
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)
    updated_conditions = trans(updated_latents, feat_text)
    updated_conditions = updated_conditions.contiguous()
    updated_z = updated_conditions[:, :-1, :]
    updated_target = latents.clone().detach()
    updated_target = updated_target.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)
    updated_z = updated_z.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)
    updated_target = updated_target[mask]
    updated_z = updated_z[mask]
    updated_loss, _ = diffusion_loss(
        getattr(trans, "module", trans).diff_loss,
        target=updated_target,
        z=updated_z,
        schedule=schedule,
    )
    return updated_loss


def forward_loss_withmask_2_forward_streaming(
    latents,
    trans,
    m_lens,
    feat_text,
    step,
    total_steps,
    A_token_length,
    diffmlps_batch_mul=4,
    schedule=None,
):
    latents = latents.to(latents.device)
    feat_text = feat_text.to(latents.device)
    A_token_length = A_token_length.to(latents.device)
    conditions = trans(latents, feat_text)
    conditions = conditions.contiguous()
    z = conditions[:, :-1, :]
    b, l, d = latents.shape
    mask = lengths_to_mask(m_lens, l)
    for j in range(b):
        mask[j, : A_token_length[j].item()] = False
    mask = mask.reshape(b * l).repeat(diffmlps_batch_mul)
    target = latents.clone().detach()
    target = target.reshape(b * l, -1)
    z = z.reshape(b * l, -1)
    with torch.no_grad():
        loss, pred_xstart = diffusion_loss(
            getattr(trans, "module", trans).diff_loss,
            target=target,
            z=z,
            schedule=schedule,
        )
    pred_xstart = pred_xstart.clone().detach()
    pred_xstart = pred_xstart.reshape(b, l, -1)
    for k in range(b):
        pred_xstart[k, : A_token_length[k].item(), :] = latents[
            k, : A_token_length[k].item(), :
        ]
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)
    updated_conditions = trans(updated_latents, feat_text)
    updated_conditions = updated_conditions.contiguous()
    updated_z = updated_conditions[:, :-1, :]
    updated_target = latents.clone().detach()
    updated_target = updated_target.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)
    updated_z = updated_z.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)
    updated_target = updated_target[mask]
    updated_z = updated_z[mask]
    updated_loss, updated_pred_xstart = diffusion_loss(
        getattr(trans, "module", trans).diff_loss,
        target=updated_target,
        z=updated_z,
        schedule=schedule,
    )
    return updated_loss
