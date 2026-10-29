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
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from mmengine import print_log
from motius.registry import HF_MODELS, MODELS

@HF_MODELS.register_module()
class VermoLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VermoLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, cache_position=None, logits_to_keep=0, projection_chunk_size=0, **kwargs):
        if labels is not None:
            raise ValueError('Compute VerMo language objectives through VermoTrainer')
        if projection_chunk_size <= 0:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=False, cache_position=cache_position, return_dict=True)
        hidden = outputs.last_hidden_state[:, :-1, :]
        chunks = [self.lm_head(hidden[:, start:start + projection_chunk_size, :]) for start in range(0, hidden.shape[1], projection_chunk_size)]
        logits = torch.cat(chunks, dim=1)
        logits = torch.cat((logits, torch.zeros_like(logits[:, :1])), dim=1)
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)

class VermoLlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = os.environ.get('VERMO_LM_GRADIENT_CHECKPOINTING', '0').lower() in {'1', 'true', 'yes', 'on'}
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, cache_position: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> BaseModelOutputWithPast:
        if torch.is_grad_enabled():
            use_cache = False
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, **kwargs)

class VermoLlamaRotaryEmbedding(LlamaRotaryEmbedding):

    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

def create_custom_forward(module):

    def custom_forward(*args, **kwargs):
        return module(*args, **kwargs)
    return custom_forward
