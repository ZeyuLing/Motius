"""The native flow objective must train through the prepared neural root."""
import torch
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.vimogen import ViMoGenBundle
from motius.trainers.vimogen import ViMoGenTrainer


class VelocityNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(276,276)
    def forward(self,x,**kwargs):return self.projection(x)


def bundle():
    value=ViMoGenBundle.__new__(ViMoGenBundle);ModelBundle.__init__(value)
    value._device=torch.device('cpu');value.dtype=torch.float32
    value.model=VelocityNetwork();value.text_encoder=None
    return value


def batch():
    mask=torch.ones(2,8);mask[1,6:]=0
    return dict(motion=torch.randn(2,8,276),motion_mask=mask,prompt_emb=torch.randn(2,5,24),
        ref_motion=torch.zeros(2,8,138),ref_motion_mask=mask.clone(),motion_dim_mask=torch.ones(2,276),
        attend_to_text_mask=torch.tensor([True,False]))


def test_prepared_neural_root_and_both_condition_partitions():
    torch.manual_seed(17);value=bundle();trainer=ViMoGenTrainer(value)
    calls=[];handle=value.model.register_forward_hook(lambda *args:calls.append(True))
    try:
        result=trainer.train_step(batch());result['loss'].backward()
        assert calls==[True] and torch.isfinite(result['loss'])
        assert set(result)=={'loss','loss_text','loss_reference'}
        assert value.model.projection.weight.grad.abs().sum()>0
    finally:handle.remove()


def _worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(17);value=bundle();trainer=ViMoGenTrainer(value)
        value.model=DistributedDataParallel(value.model)
        optimizer=torch.optim.AdamW(value.model.parameters(),lr=1e-3)
        for step in range(2):
            torch.manual_seed(19+rank+step);optimizer.zero_grad()
            trainer.train_step(batch())['loss'].backward();optimizer.step()
        parameter=value.unwrap_network(value.model).projection.weight.detach()
        values=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(values,parameter)
        torch.testing.assert_close(*values,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_flow_training(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)


def _gpu_worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from motius.models.vimogen.motion_network import WanModelTM2M
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(17);value=ViMoGenBundle.__new__(ViMoGenBundle);ModelBundle.__init__(value)
        value._device=torch.device('cuda',rank);value.dtype=torch.bfloat16;value.text_encoder=None
        value.model=WanModelTM2M(in_dim=276,out_dim=276,dim=32,ffn_dim=64,freq_dim=16,text_dim=24,
            num_heads=4,num_layers=2,ref_motion_dim=138).to(value.device,value.dtype)
        trainer=ViMoGenTrainer(value);value.model=DistributedDataParallel(value.model,device_ids=[rank])
        optimizer=torch.optim.AdamW(value.model.parameters(),lr=1e-3)
        for step in range(2):
            torch.manual_seed(19+rank+step);optimizer.zero_grad();values=batch()
            values['ref_motion'][1,:6]=torch.randn(6,138)
            trainer.train_step(values)['loss'].backward();optimizer.step()
        parameter=value.unwrap_network(value.model).motion_embedding.weight.detach()
        copies=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(copies,parameter)
        torch.testing.assert_close(*copies,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_gpu_actual_wan_training(tmp_path):
    import pytest
    if torch.cuda.device_count()<2:pytest.skip('Two CUDA devices required for actual Wan attention kernels')
    torch.multiprocessing.spawn(_gpu_worker,args=(f'file://{tmp_path / "gpu_rendezvous"}',),nprocs=2,join=True)
