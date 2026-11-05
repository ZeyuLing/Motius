from __future__ import annotations
import torch as t


def log(x, eps=1e-20):
    return t.log(x.clamp(min=eps))


def gumbel_noise(x):
    noise = t.zeros_like(x).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(x, temperature=1., dim=-1):
    return ((x / max(temperature, 1e-10)) + gumbel_noise(x)).argmax(dim=dim)


def select_root_token(network,num_token_logits,num_tokens,allowed_pred_num_tokens=None,config=None):
    """The original root-length decision, outside the neural forward graph."""
    import torch
    config={} if config is None else config
    batch_size=len(num_tokens);device=num_tokens.device
    if not config.get('allow_pred_out_of_reach_num_tokens',True):
        num_token_logits=num_token_logits.clone()
        num_token_logits[:,network.OUT_OF_REACH_NUM_TOKENS-network.args['min_tokens']]=-t.inf
    if allowed_pred_num_tokens is not None:
        num_time_tokens=network.args['max_tokens']-network.args['min_tokens']+1
        modified=t.where(allowed_pred_num_tokens==1,num_token_logits[:,:num_time_tokens],t.full([batch_size,num_time_tokens],-t.inf).to(device))
        num_token_logits=t.cat([modified,num_token_logits[:,num_time_tokens:]],dim=-1)
    chosen=torch.argmax(num_token_logits,dim=-1).int()
    chosen=t.where(num_tokens.view([batch_size])==network.MASKED_NUM_TOKENS,chosen,num_tokens.view([batch_size])-network.args['min_tokens'])
    return chosen,num_token_logits
