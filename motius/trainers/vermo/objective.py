from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

@dataclass
class VermoCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal LM output with cheap modal-boundary diagnostics."""
    vermo_boundary_correct: Optional[torch.Tensor] = None
    vermo_boundary_count: Optional[torch.Tensor] = None
    vermo_boundary_loss_sum: Optional[torch.Tensor] = None
    vermo_sample_loss_sums: Optional[torch.Tensor] = None
    vermo_sample_loss_weights: Optional[torch.Tensor] = None
    vermo_sample_boundary_correct: Optional[torch.Tensor] = None
    vermo_sample_boundary_counts: Optional[torch.Tensor] = None
    vermo_sample_boundary_loss_sums: Optional[torch.Tensor] = None

def chunked_logit_loss(hidden_states: torch.Tensor, labels: torch.Tensor, lm_head: nn.Module, vocab_size: int, chunk_size: int, token_weights: Optional[torch.Tensor]=None, fsq_aux_mask: Optional[torch.Tensor]=None, motion_token_range: Optional[tuple[int, int]]=None, fsq_code_levels: Optional[torch.Tensor]=None, fsq_canonical_to_code: Optional[torch.Tensor]=None, fsq_level_sizes: Optional[tuple[int, ...]]=None, fsq_aux_weight: float=0.0, return_per_sample: bool=False):
    """Compute exact causal CE without retaining full float32 logits.

    Cross entropy upcasts logits to float32.  With VerMo's 147k vocabulary,
    materializing every long-sequence logit can consume several GiB.  Each
    chunk is activation-checkpointed so its logits are recomputed and released
    during backward instead of all chunks remaining live at once.
    """
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(f'Expected hidden_states [B, L, H] and labels [B, L], got {tuple(hidden_states.shape)} and {tuple(labels.shape)}')
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError('hidden_states and labels must share batch/sequence axes')
    if token_weights is not None and token_weights.shape != labels.shape:
        raise ValueError('token_weights and labels must share batch/sequence axes')
    if fsq_aux_mask is not None and fsq_aux_mask.shape != labels.shape:
        raise ValueError('fsq_aux_mask and labels must share batch/sequence axes')
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')
    fsq_enabled = fsq_aux_weight > 0.0
    if fsq_enabled:
        if fsq_aux_mask is None or motion_token_range is None or fsq_code_levels is None or (fsq_canonical_to_code is None) or (fsq_level_sizes is None):
            raise ValueError('FSQ auxiliary loss requires its mask and code metadata')
        motion_start, motion_end = motion_token_range
        if motion_end - motion_start != fsq_code_levels.shape[0]:
            raise ValueError('Motion token range does not match FSQ codebook size')
        if int(torch.tensor(fsq_level_sizes).prod().item()) != motion_end - motion_start:
            raise ValueError('FSQ level sizes do not span the motion codebook')
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    valid_count = valid.sum()
    sample_loss_weights = valid.sum(dim=1).float()
    zero_by_sample = hidden_states.new_zeros((hidden_states.shape[0],), dtype=torch.float32)
    if int(valid_count.detach().cpu()) == 0:
        loss = hidden_states.sum() * 0.0
        if return_per_sample:
            return (loss, zero_by_sample, sample_loss_weights)
        return loss
    if token_weights is None:
        shift_weights = valid.to(dtype=torch.float32)
    else:
        shift_weights = token_weights[:, 1:].float() * valid
    weight_sum = shift_weights.sum()
    if float(weight_sum.detach().cpu()) <= 0:
        loss = hidden_states.sum() * 0.0
        if return_per_sample:
            return (loss, zero_by_sample, sample_loss_weights)
        return loss
    loss_sums = zero_by_sample

    def chunk_loss(hidden_chunk, label_chunk, weight_chunk, fsq_mask_chunk):
        logits = lm_head(hidden_chunk).float()
        losses = F.cross_entropy(logits.reshape(-1, vocab_size), label_chunk.reshape(-1), ignore_index=-100, reduction='none').reshape(label_chunk.shape)
        if fsq_enabled:
            selected = fsq_mask_chunk.bool() & (label_chunk >= motion_start) & (label_chunk < motion_end)
            if selected.any():
                local_targets = label_chunk[selected] - motion_start
                motion_logits = logits[selected][:, motion_start:motion_end]
                canonical_logits = motion_logits.index_select(-1, fsq_canonical_to_code).reshape(-1, *fsq_level_sizes)
                level_losses = []
                num_dims = len(fsq_level_sizes)
                for dim in range(num_dims):
                    reduce_dims = tuple((axis + 1 for axis in range(num_dims) if axis != dim))
                    marginal_logits = torch.logsumexp(canonical_logits, dim=reduce_dims)
                    target_levels = fsq_code_levels[local_targets, dim]
                    level_losses.append(F.cross_entropy(marginal_logits, target_levels, reduction='none'))
                aux_losses = torch.stack(level_losses, dim=-1).mean(dim=-1)
                losses = losses.clone()
                losses[selected] += float(fsq_aux_weight) * aux_losses
        weighted = losses * weight_chunk.float()
        return weighted.sum(dim=1)
    if fsq_aux_mask is None:
        shift_fsq_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    else:
        shift_fsq_mask = fsq_aux_mask[:, 1:].bool()
    if fsq_enabled:
        fsq_code_levels = fsq_code_levels.to(device=hidden_states.device, dtype=torch.long)
        fsq_canonical_to_code = fsq_canonical_to_code.to(device=hidden_states.device, dtype=torch.long)
    for start in range(0, shift_hidden.shape[1], chunk_size):
        end = min(start + chunk_size, shift_hidden.shape[1])
        loss_sums = loss_sums + checkpoint(chunk_loss, shift_hidden[:, start:end, :], shift_labels[:, start:end], shift_weights[:, start:end], shift_fsq_mask[:, start:end], use_reentrant=False)
    loss = loss_sums.sum() / valid_count.to(dtype=loss_sums.dtype)
    if return_per_sample:
        return (loss, loss_sums.detach(), sample_loss_weights)
    return loss

def modal_boundary_diagnostics(hidden_states: torch.Tensor, labels: torch.Tensor, boundary_mask: Optional[torch.Tensor], lm_head: nn.Module, return_per_sample: bool=False):
    """Evaluate top-1 and CE only at sparse supervised boundary positions."""
    zero = hidden_states.new_zeros((), dtype=torch.float32)
    zero_by_sample = hidden_states.new_zeros((hidden_states.shape[0],), dtype=torch.float32)
    if boundary_mask is None:
        if return_per_sample:
            return (zero, zero, zero, zero_by_sample, zero_by_sample.clone(), zero_by_sample.clone())
        return (zero, zero, zero)
    if boundary_mask.shape != labels.shape:
        raise ValueError('boundary_mask and labels must share batch/sequence axes')
    targets = labels[:, 1:]
    mask = boundary_mask[:, 1:].bool() & (targets != -100)
    count_by_sample = mask.sum(dim=1).float()
    count = count_by_sample.sum()
    if int(count.detach().cpu()) == 0:
        if return_per_sample:
            return (zero, count, zero, zero_by_sample, count_by_sample, zero_by_sample.clone())
        return (zero, count, zero)
    selected_hidden = hidden_states[:, :-1, :][mask]
    selected_targets = targets[mask]
    sample_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device).unsqueeze(1).expand_as(mask)[mask]
    with torch.no_grad():
        logits = lm_head(selected_hidden.detach()).float()
        correct_values = (logits.argmax(dim=-1) == selected_targets).float()
        loss_values = F.cross_entropy(logits, selected_targets, reduction='none').float()
        correct_by_sample = zero_by_sample.scatter_add(0, sample_ids, correct_values)
        loss_by_sample = zero_by_sample.scatter_add(0, sample_ids, loss_values)
        correct = correct_by_sample.sum()
        loss_sum = loss_by_sample.sum()
    if return_per_sample:
        return (correct, count, loss_sum, correct_by_sample, count_by_sample, loss_by_sample)
    return (correct, count, loss_sum)
