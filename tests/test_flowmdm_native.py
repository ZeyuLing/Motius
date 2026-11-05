"""FlowMDM native training uses the prepared neural module across successive steps."""
from argparse import Namespace
from unittest.mock import patch
import torch
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.flowmdm import FlowMDMBundle
from motius.models.flowmdm.network import FlowMDM
from motius.trainers.flowmdm import FlowMDMTrainer
from test_momask_native import TinyTextEncoder


def bundle():
    model=FlowMDMBundle.__new__(FlowMDMBundle)
    ModelBundle.__init__(model)
    with patch('motius.models.flowmdm.network.model.base_models.TextConditionalModel.load_and_freeze_clip',
               side_effect=lambda *args:TinyTextEncoder().requires_grad_(False)):
        model.model=FlowMDM(njoints=263,nfeats=1,translation=True,pose_rep='hml_vec',glob=True,glob_rot=True,
            latent_dim=16,ff_size=32,num_layers=1,num_heads=2,dropout=0.,data_rep='hml_vec',dataset='humanml',
            clip_dim=8,cond_mode='text',cond_mask_prob=.1,diffusion_steps=100,bpe_training_ratio=.5,bpe_denoising_step=20)
    model.flow_args=Namespace(diffusion_steps=100,noise_schedule='cosine',sigma_small=True,
        lambda_vel=0.,lambda_rcxyz=0.,lambda_vel_rcxyz=0.,lambda_fc=0.)
    model.register_buffer('mean',torch.zeros(263));model.register_buffer('std',torch.ones(263))
    return model


def batch():
    return dict(motion=torch.randn(2,8,263),text=['walk','jump'],motion_lens=torch.tensor([8,6]))


def test_training_successive_steps():
    torch.manual_seed(7)
    model=bundle();trainer=FlowMDMTrainer(model);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    calls=[];handle=model.model.register_forward_hook(lambda *args:calls.append(True))
    try:
        for _ in range(2):
            optimizer.zero_grad();loss=trainer.train_step(batch())['loss'];loss.backward();optimizer.step()
            assert torch.isfinite(loss)
        assert calls==[True,True]
        assert model.network.input_process.poseEmbedding.weight.grad.abs().sum()>0
    finally:handle.remove()


def _ddp_worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(7);model=bundle();trainer=FlowMDMTrainer(model)
        model.model=DistributedDataParallel(model.model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for step in range(2):
            torch.manual_seed(19+rank+step)
            optimizer.zero_grad();trainer.train_step(batch())['loss'].backward();optimizer.step()
        parameter=model.network.input_process.poseEmbedding.weight.detach()
        values=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(values,parameter)
        torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_training(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
