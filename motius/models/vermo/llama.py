from dataclasses import dataclass
from typing import Optional
import os
import torch
from transformers import LlamaConfig
try:
    from transformers import dynamic_rope_update
except ImportError:
    try:
        from transformers.modeling_rope_utils import dynamic_rope_update
    except ImportError:
        def dynamic_rope_update(rope_forward):
            return rope_forward
from transformers.models.llama import (
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from mmengine import print_log
from motius.registry import HF_MODELS, MODELS


def _disable_flash_attention_packed_sequence_check() -> None:
    """Avoid a GPU-side packed-sequence probe for non-packed VerMo batches."""
    if os.environ.get("VERMO_DISABLE_FA2_PACKED_SEQUENCE_CHECK", "1") == "0":
        return
    try:
        import transformers.modeling_flash_attention_utils as fa_utils
        import transformers.models.llama.modeling_llama as llama_modeling
    except Exception:
        return

    def _not_packed_sequence(position_ids, batch_size):
        return False

    fa_utils._is_packed_sequence = _not_packed_sequence

    def _wrap_flash_attention_forward(fn):
        if getattr(fn, "_vermo_no_position_ids_probe", False):
            return fn

        def _flash_attention_forward_without_position_probe(*args, **kwargs):
            args = list(args)
            if len(args) >= 8:
                args[7] = None
                kwargs.pop("position_ids", None)
            else:
                kwargs["position_ids"] = None
            return fn(*args, **kwargs)

        _flash_attention_forward_without_position_probe._vermo_no_position_ids_probe = True
        return _flash_attention_forward_without_position_probe

    fa_utils._flash_attention_forward = _wrap_flash_attention_forward(
        fa_utils._flash_attention_forward
    )
    if hasattr(llama_modeling, "_flash_attention_forward"):
        llama_modeling._flash_attention_forward = fa_utils._flash_attention_forward


_disable_flash_attention_packed_sequence_check()


def _configure_sdp_backends() -> None:
    backend = os.environ.get("VERMO_SDP_BACKEND", "").strip().lower().replace("-", "_")
    if not backend or not hasattr(torch.backends, "cuda"):
        return

    cuda_backends = torch.backends.cuda
    if backend in {"no_flash", "mem_efficient", "mem_efficient_or_math"}:
        cuda_backends.enable_flash_sdp(False)
        cuda_backends.enable_mem_efficient_sdp(True)
        cuda_backends.enable_math_sdp(True)
        if hasattr(cuda_backends, "enable_cudnn_sdp"):
            cuda_backends.enable_cudnn_sdp(False)
    elif backend == "math":
        cuda_backends.enable_flash_sdp(False)
        cuda_backends.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backends, "enable_cudnn_sdp"):
            cuda_backends.enable_cudnn_sdp(False)
        cuda_backends.enable_math_sdp(True)


_configure_sdp_backends()


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


def chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    vocab_size: int,
    chunk_size: int,
    token_weights: Optional[torch.Tensor] = None,
    fsq_aux_mask: Optional[torch.Tensor] = None,
    motion_token_range: Optional[tuple[int, int]] = None,
    fsq_code_levels: Optional[torch.Tensor] = None,
    fsq_canonical_to_code: Optional[torch.Tensor] = None,
    fsq_level_sizes: Optional[tuple[int, ...]] = None,
    fsq_aux_weight: float = 0.0,
    return_per_sample: bool = False,
):
    """Compute exact causal CE without retaining full float32 logits.

    Cross entropy upcasts logits to float32.  With VerMo's 147k vocabulary,
    materializing every long-sequence logit can consume several GiB.  Each
    chunk is activation-checkpointed so its logits are recomputed and released
    during backward instead of all chunks remaining live at once.
    """
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "Expected hidden_states [B, L, H] and labels [B, L], got "
            f"{tuple(hidden_states.shape)} and {tuple(labels.shape)}"
        )
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must share batch/sequence axes")
    if token_weights is not None and token_weights.shape != labels.shape:
        raise ValueError("token_weights and labels must share batch/sequence axes")
    if fsq_aux_mask is not None and fsq_aux_mask.shape != labels.shape:
        raise ValueError("fsq_aux_mask and labels must share batch/sequence axes")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    fsq_enabled = fsq_aux_weight > 0.0
    if fsq_enabled:
        if (
            fsq_aux_mask is None
            or motion_token_range is None
            or fsq_code_levels is None
            or fsq_canonical_to_code is None
            or fsq_level_sizes is None
        ):
            raise ValueError("FSQ auxiliary loss requires its mask and code metadata")
        motion_start, motion_end = motion_token_range
        if motion_end - motion_start != fsq_code_levels.shape[0]:
            raise ValueError("Motion token range does not match FSQ codebook size")
        if int(torch.tensor(fsq_level_sizes).prod().item()) != motion_end - motion_start:
            raise ValueError("FSQ level sizes do not span the motion codebook")

    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    valid_count = valid.sum()
    sample_loss_weights = valid.sum(dim=1).float()
    zero_by_sample = hidden_states.new_zeros(
        (hidden_states.shape[0],), dtype=torch.float32
    )
    if int(valid_count.detach().cpu()) == 0:
        loss = hidden_states.sum() * 0.0
        if return_per_sample:
            return loss, zero_by_sample, sample_loss_weights
        return loss

    if token_weights is None:
        shift_weights = valid.to(dtype=torch.float32)
    else:
        shift_weights = token_weights[:, 1:].float() * valid
    weight_sum = shift_weights.sum()
    if float(weight_sum.detach().cpu()) <= 0:
        loss = hidden_states.sum() * 0.0
        if return_per_sample:
            return loss, zero_by_sample, sample_loss_weights
        return loss

    loss_sums = zero_by_sample

    def chunk_loss(hidden_chunk, label_chunk, weight_chunk, fsq_mask_chunk):
        logits = lm_head(hidden_chunk).float()
        losses = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            label_chunk.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(label_chunk.shape)
        if fsq_enabled:
            selected = (
                fsq_mask_chunk.bool()
                & (label_chunk >= motion_start)
                & (label_chunk < motion_end)
            )
            if selected.any():
                local_targets = label_chunk[selected] - motion_start
                motion_logits = logits[selected][:, motion_start:motion_end]
                canonical_logits = motion_logits.index_select(
                    -1, fsq_canonical_to_code
                ).reshape(-1, *fsq_level_sizes)
                level_losses = []
                num_dims = len(fsq_level_sizes)
                for dim in range(num_dims):
                    reduce_dims = tuple(
                        axis + 1 for axis in range(num_dims) if axis != dim
                    )
                    marginal_logits = torch.logsumexp(
                        canonical_logits, dim=reduce_dims
                    )
                    target_levels = fsq_code_levels[local_targets, dim]
                    level_losses.append(
                        F.cross_entropy(
                            marginal_logits,
                            target_levels,
                            reduction="none",
                        )
                    )
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
        fsq_code_levels = fsq_code_levels.to(
            device=hidden_states.device, dtype=torch.long
        )
        fsq_canonical_to_code = fsq_canonical_to_code.to(
            device=hidden_states.device, dtype=torch.long
        )

    for start in range(0, shift_hidden.shape[1], chunk_size):
        end = min(start + chunk_size, shift_hidden.shape[1])
        loss_sums = loss_sums + checkpoint(
            chunk_loss,
            shift_hidden[:, start:end, :],
            shift_labels[:, start:end],
            shift_weights[:, start:end],
            shift_fsq_mask[:, start:end],
            use_reentrant=False,
        )
    # Keep the original per-valid-token CE scale.  Boundary weights add
    # supervision without reducing the gradient on ordinary output tokens.
    loss = loss_sums.sum() / valid_count.to(dtype=loss_sums.dtype)
    if return_per_sample:
        return loss, loss_sums.detach(), sample_loss_weights
    return loss


def modal_boundary_diagnostics(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    boundary_mask: Optional[torch.Tensor],
    lm_head: nn.Module,
    return_per_sample: bool = False,
):
    """Evaluate top-1 and CE only at sparse supervised boundary positions."""
    zero = hidden_states.new_zeros((), dtype=torch.float32)
    zero_by_sample = hidden_states.new_zeros(
        (hidden_states.shape[0],), dtype=torch.float32
    )
    if boundary_mask is None:
        if return_per_sample:
            return (
                zero, zero, zero,
                zero_by_sample, zero_by_sample.clone(), zero_by_sample.clone(),
            )
        return zero, zero, zero
    if boundary_mask.shape != labels.shape:
        raise ValueError("boundary_mask and labels must share batch/sequence axes")

    targets = labels[:, 1:]
    mask = boundary_mask[:, 1:].bool() & (targets != -100)
    count_by_sample = mask.sum(dim=1).float()
    count = count_by_sample.sum()
    if int(count.detach().cpu()) == 0:
        if return_per_sample:
            return (
                zero, count, zero,
                zero_by_sample, count_by_sample, zero_by_sample.clone(),
            )
        return zero, count, zero

    selected_hidden = hidden_states[:, :-1, :][mask]
    selected_targets = targets[mask]
    sample_ids = torch.arange(
        hidden_states.shape[0], device=hidden_states.device
    ).unsqueeze(1).expand_as(mask)[mask]
    with torch.no_grad():
        logits = lm_head(selected_hidden.detach()).float()
        correct_values = (
            logits.argmax(dim=-1) == selected_targets
        ).float()
        loss_values = F.cross_entropy(
            logits,
            selected_targets,
            reduction="none",
        ).float()
        correct_by_sample = zero_by_sample.scatter_add(
            0, sample_ids, correct_values
        )
        loss_by_sample = zero_by_sample.scatter_add(
            0, sample_ids, loss_values
        )
        correct = correct_by_sample.sum()
        loss_sum = loss_by_sample.sum()
    if return_per_sample:
        return (
            correct,
            count,
            loss_sum,
            correct_by_sample,
            count_by_sample,
            loss_by_sample,
        )
    return correct, count, loss_sum


@HF_MODELS.register_module()
class VermoLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VermoLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        loss_token_weights: Optional[torch.Tensor] = None,
        loss_boundary_mask: Optional[torch.Tensor] = None,
        loss_fsq_aux_mask: Optional[torch.Tensor] = None,
        motion_token_range: Optional[tuple[int, int]] = None,
        motion_fsq_code_levels: Optional[torch.Tensor] = None,
        motion_fsq_canonical_to_code: Optional[torch.Tensor] = None,
        motion_fsq_level_sizes: Optional[tuple[int, ...]] = None,
        motion_fsq_aux_weight: float = 0.0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        chunk_size = int(os.environ.get("VERMO_LM_LOSS_CHUNK_SIZE", "0"))
        if (
            labels is not None
            and loss_token_weights is not None
            and torch.is_grad_enabled()
            and chunk_size <= 0
        ):
            raise ValueError(
                "Weighted VerMo loss requires VERMO_LM_LOSS_CHUNK_SIZE > 0"
            )
        if labels is None or chunk_size <= 0 or not torch.is_grad_enabled():
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            cache_position=cache_position,
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=True,
        )
        loss, sample_loss_sums, sample_loss_weights = chunked_causal_lm_loss(
            outputs.last_hidden_state,
            labels,
            self.lm_head,
            self.config.vocab_size,
            chunk_size,
            token_weights=loss_token_weights,
            fsq_aux_mask=loss_fsq_aux_mask,
            motion_token_range=motion_token_range,
            fsq_code_levels=motion_fsq_code_levels,
            fsq_canonical_to_code=motion_fsq_canonical_to_code,
            fsq_level_sizes=motion_fsq_level_sizes,
            fsq_aux_weight=motion_fsq_aux_weight,
            return_per_sample=True,
        )
        (
            boundary_correct,
            boundary_count,
            boundary_loss_sum,
            sample_boundary_correct,
            sample_boundary_counts,
            sample_boundary_loss_sums,
        ) = (
            modal_boundary_diagnostics(
                outputs.last_hidden_state,
                labels,
                loss_boundary_mask,
                self.lm_head,
                return_per_sample=True,
            )
        )
        return VermoCausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vermo_boundary_correct=boundary_correct,
            vermo_boundary_count=boundary_count,
            vermo_boundary_loss_sum=boundary_loss_sum,
            vermo_sample_loss_sums=sample_loss_sums,
            vermo_sample_loss_weights=sample_loss_weights,
            vermo_sample_boundary_correct=sample_boundary_correct,
            vermo_sample_boundary_counts=sample_boundary_counts,
            vermo_sample_boundary_loss_sums=sample_boundary_loss_sums,
        )

class VermoLlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = (
            os.environ.get("VERMO_LM_GRADIENT_CHECKPOINTING", "0").lower()
            in {"1", "true", "yes", "on"}
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if torch.is_grad_enabled():
            use_cache = False
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )


class VermoLlamaRotaryEmbedding(LlamaRotaryEmbedding):

    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.no_grad():
            with torch.autocast(
                device_type=device_type, enabled=False
            ):  # Force float32
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def create_custom_forward(module):
    def custom_forward(*args, **kwargs):
        return module(*args, **kwargs)

    return custom_forward
