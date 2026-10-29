from motius.motion.representation.hymotion_runtime.codecs import HyMotionCodec
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from motius.motion.conditioning.gentrack import _len_to_mask, pack_context
from motius.motion.representation.g1 import decode_g1_to_qpos
from motius.diffusion.gentrack import flow_grpo_transition

@torch.no_grad()
def sample_motion(self, text_vec: Tensor, text_ctxt: List[Tensor], ctxt_len: Tensor, lengths: Tensor, num_steps: Optional[int]=None, guidance: Optional[float]=None, initial_noise: Optional[Tensor]=None, transformer=None, return_initial_noise: bool=False) -> Tensor:
    """Flow-matching ODE -> NORMALIZED 38-d motion (B, Lmax, 38)."""
    device = self._device()
    dtype = torch.float32
    num_steps = int(num_steps or self.sample_steps)
    guidance = float(guidance if guidance is not None else self.sample_guidance)
    vtxt = text_vec.to(device, dtype)
    ctxt, ctxt_mask = pack_context(self, text_ctxt, ctxt_len, device, dtype)
    lengths = lengths.to(device)
    L = int(lengths.max().item())
    Lp = max(L, self.TRAIN_FRAMES)
    B = vtxt.shape[0]
    core = transformer
    motion_dim = getattr(core, 'output_dim', self._core_transformer.output_dim)
    x_mask = _len_to_mask(lengths, Lp)

    def predict(x_input, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal):
        if core is None:
            return self.predict_flow(x_input=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input, timesteps=timesteps, x_mask_temporal=x_mask_temporal, ctxt_mask_temporal=ctxt_mask_temporal)
        return core(x=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input, timesteps=timesteps, x_mask_temporal=x_mask_temporal, ctxt_mask_temporal=ctxt_mask_temporal, mask_density=None, task_emb=None)
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
            xp = predict(x_input=xd, ctxt_input=ctxt_cfg, vtxt_input=vtxt_cfg, timesteps=t_val.expand(2 * B), x_mask_temporal=x_mask_cfg, ctxt_mask_temporal=ctxt_mask_cfg)
            pu, pt = xp.chunk(2, 0)
            return pu + guidance * (pt - pu)
        return predict(x_input=x, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t_val.expand(B), x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)
    if initial_noise is None:
        y0 = torch.randn(B, Lp, motion_dim, device=device, dtype=dtype)
    else:
        y0 = initial_noise.to(device=device, dtype=dtype)
        if y0.shape[0] != B or y0.shape[2] != motion_dim:
            raise ValueError(f'initial_noise shape {tuple(y0.shape)} incompatible with B={B}, motion_dim={motion_dim}')
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
    sampled = sampled[:, :L, :]
    if return_initial_noise:
        return (sampled, y0)
    return sampled

@torch.no_grad()
def sample_motion_grpo(self, text_vec: Tensor, text_ctxt: List[Tensor], ctxt_len: Tensor, lengths: Tensor, num_steps: Optional[int]=None, guidance: Optional[float]=None, eta: float=0.7, initial_noise: Optional[Tensor]=None, transition_noises: Optional[Tensor]=None, transformer=None, return_policy_artifacts: bool=True) -> Dict[str, Tensor]:
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
        raise ValueError('Flow-GRPO currently requires guidance=1.0 for exact policy log-probs')
    if num_steps < 2:
        raise ValueError('Flow-GRPO requires at least two sampling steps')
    vtxt = text_vec.to(device, dtype)
    ctxt, ctxt_mask = pack_context(self, text_ctxt, ctxt_len, device, dtype)
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
            raise ValueError(f'initial_noise shape {tuple(state.shape)} does not match {expected_state_shape}')
    initial_state = state
    if transition_noises is not None:
        transition_noises = transition_noises.to(device=device, dtype=dtype)
        expected_noise_shape = (batch, num_steps, padded_length, motion_dim)
        if transition_noises.shape != expected_noise_shape:
            raise ValueError(f'transition_noises shape {tuple(transition_noises.shape)} does not match {expected_noise_shape}')
    states = [state] if return_policy_artifacts else []
    old_log_probs = []
    used_transition_noises = []
    for index in range(num_steps):
        sigma = sigmas[index]
        t = (1.0 - sigma).expand(batch)
        if transformer is None:
            velocity = self.predict_flow(x_input=state, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t, x_mask_temporal=frame_mask, ctxt_mask_temporal=ctxt_mask)
        else:
            velocity = transformer(x=state, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t, x_mask_temporal=frame_mask, ctxt_mask_temporal=ctxt_mask, mask_density=None, task_emb=None)
        step_noise = transition_noises[:, index] if transition_noises is not None else torch.randn_like(state)
        state, log_prob = flow_grpo_transition(model_output=-velocity, latents=state, sigma=sigma, sigma_next=sigmas[index + 1], sigma_first_next=sigmas[1], eta=eta, frame_mask=frame_mask, transition_noise=step_noise)
        used_transition_noises.append(step_noise)
        if return_policy_artifacts:
            states.append(state)
            old_log_probs.append(log_prob)
    result = {'sample': state[:, :length_max].detach(), 'initial_noise': initial_state.detach(), 'transition_noises': torch.stack(used_transition_noises, dim=1).detach(), 'sigmas': sigmas.detach()}
    if return_policy_artifacts:
        result['states'] = torch.stack(states, dim=1).detach()
        result['old_log_probs'] = torch.stack(old_log_probs, dim=1).detach()
    return result

@torch.no_grad()
def latents_to_qpos(self, latent: Tensor) -> np.ndarray:
    """Normalized 38-d motion (B, L, 38) -> qpos numpy (B, L, 36)."""
    denorm = HyMotionCodec(self).denormalize_motion(latent.to(self._device()).float())
    qpos = []
    for b in range(denorm.shape[0]):
        qpos.append(decode_g1_to_qpos(denorm[b].cpu()).numpy())
    return np.stack(qpos, axis=0)

def save_qpos_csv(qpos_sample: np.ndarray, csv_path: str) -> None:
    """qpos (T, 36) -> header-less, frame-column-less CSV for the converter."""
    arr = np.asarray(qpos_sample, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    np.savetxt(csv_path, arr, delimiter=',')
