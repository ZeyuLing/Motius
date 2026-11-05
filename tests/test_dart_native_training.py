"""Exercise actual DART neural graphs and distributed optimizer boundaries."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from motius.models.dart import DARTBundle
from motius.models.dart.denoiser import DenoiserTransformer
from motius.models.dart.vae import AutoMldVae
from motius.models.base_model_bundle import ModelBundle
from motius.motion.representation.dart import PrimitiveUtility
from motius.trainers.dart import DARTTrainer


def tiny_bundle():
    bundle=DARTBundle.__new__(DARTBundle);ModelBundle.__init__(bundle)
    bundle.denoiser_model=DenoiserTransformer(h_dim=32,ff_size=64,num_layers=2,num_heads=4,dropout=0.,clip_dim=12,noise_shape=(1,8),history_shape=(2,276),cond_mask_prob=.1)
    bundle.vae_model=AutoMldVae(nfeats=276,latent_dim=(1,8),h_dim=16,ff_size=32,num_layers=3,num_heads=4,dropout=0.)
    bundle.clip_model=None
    bundle.denoiser_args=SimpleNamespace(diffusion_args=SimpleNamespace(diffusion_steps=10,respacing='',noise_schedule='cosine',sigma_small=True),rescale_latent=True,train_rollout_history='rollout')
    return bundle


def processor():
    representation=PrimitiveUtility.__new__(PrimitiveUtility)
    representation.motion_repr=dict(transl=3,poses_6d=132,transl_delta=3,global_orient_delta_6d=6,joints=66,joints_delta=66)
    return SimpleNamespace(history_length=2,future_length=8,num_primitive=1,primitive_utility=representation,denormalize=lambda x:x,tensor_to_dict=representation.tensor_to_dict)


def batch():
    return dict(motion=torch.randn(2,10,276),text_embedding=torch.randn(2,12),condition={'gender':['male']*2,'betas':torch.zeros(2,10,10)})


@pytest.mark.parametrize('stage',['denoiser','vae'])
def test_actual_network_objective(stage):
    torch.set_num_threads(1);bundle=tiny_bundle()
    trainer=DARTTrainer(bundle,processor(),stage=stage,loss_weights={'weight_smpl_joints_rec':0.,'weight_joints_consistency':0.})
    result=trainer.train_step(batch());result['loss'].backward()
    model=bundle.denoiser_model if stage=='denoiser' else bundle.vae_model
    assert torch.isfinite(result['loss']).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(p.grad is not None for p in model.parameters())>20


def _worker(rank,path,stage):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(41);bundle=tiny_bundle()
    trainer=DARTTrainer(bundle,processor(),stage=stage,loss_weights={'weight_smpl_joints_rec':0.,'weight_joints_consistency':0.})
    name='denoiser_model' if stage=='denoiser' else 'vae_model'
    network=getattr(bundle,name)
    setattr(bundle,name,DDP(network,find_unused_parameters=True))
    optimizer=torch.optim.SGD(network.parameters(),lr=1e-5)
    for _ in range(2):
        torch.manual_seed(100+rank);optimizer.zero_grad(set_to_none=True)
        trainer.train_step(batch())['loss'].backward();optimizer.step()
    weight=next(p for p in network.parameters() if p.grad is not None).detach()
    gathered=[torch.empty_like(weight) for _ in range(2)];dist.all_gather(gathered,weight)
    assert torch.equal(gathered[0],gathered[1])
    dist.destroy_process_group()


@pytest.mark.parametrize('stage',['denoiser','vae'])
def test_two_process_actual_networks(tmp_path,stage):
    torch.multiprocessing.spawn(_worker,args=(str(tmp_path/'rendezvous'),stage),nprocs=2,join=True)
