from types import SimpleNamespace
import torch
from torch import nn
from transformers import LlamaConfig
from motius.models.vermo import VermoBundle
from motius.models.vermo.llama import VermoLlamaForCausalLM
from motius.models.base_model_bundle import ModelBundle
from motius.trainers.vermo import VermoTrainer


def bundle():
    result=VermoBundle.__new__(VermoBundle);ModelBundle.__init__(result)
    result.lm=VermoLlamaForCausalLM(LlamaConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,tie_word_embeddings=True,attention_dropout=0.))
    result.motion_tokenizer=nn.Linear(1,1);result.audio_tokenizer=None
    return result


def batch():
    tokens=torch.randint(0,64,(2,9));labels=tokens.clone();labels[:,:3]=-100
    weights=torch.ones_like(tokens,dtype=torch.float32);weights[:,3]=2.
    boundary=torch.zeros_like(tokens,dtype=torch.bool);boundary[:,3]=True
    return dict(input_ids=tokens,labels=labels,loss_token_weights=weights,loss_boundary_mask=boundary)


def test_weighted_causal_loss_uses_actual_tied_language_network():
    torch.set_num_threads(1)
    model=bundle();trainer=VermoTrainer(model,processor=SimpleNamespace(),loss_chunk_size=3)
    result=trainer.train_step(batch());result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert model.lm.model.embed_tokens.weight is model.lm.lm_head.weight
    assert model.lm.lm_head.weight.grad.abs().sum()>0
    assert result['boundary_count']==2


def _worker(rank,path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(91);model=bundle();trainer=VermoTrainer(model,processor=SimpleNamespace(),loss_chunk_size=3)
    model.lm=DDP(model.lm)
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)
    for i in range(2):
        torch.manual_seed(i+rank+17);optimizer.zero_grad(set_to_none=True);trainer.train_step(batch())['loss'].backward();optimizer.step()
    weight=model.lm.module.lm_head.weight.detach()
    values=[torch.empty_like(weight) for _ in range(2)];dist.all_gather(values,weight)
    assert torch.equal(values[0],values[1]);dist.destroy_process_group()


def test_two_process_tied_head_and_backbone(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(str(tmp_path/'group'),),nprocs=2,join=True)
