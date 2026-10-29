"""Native MoGenTS training owns four token objectives outside its neural graphs."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.mogents import MoGenTSBundle
from motius.models.mogents.network import MaskTransformer,ResidualTransformer,MaskTransformer2D,ResidualTransformer2D
from motius.trainers.mogents import MoGenTSTrainer
from motius.pipelines.mogents.decoding import aux_base_generate,aux_residual_generate,ts_base_generate,ts_residual_generate
from test_momask_native import TinyTextEncoder


def bundle():
    model=MoGenTSBundle.__new__(MoGenTSBundle);ModelBundle.__init__(model)
    opt=SimpleNamespace(num_tokens=8,num_tokens1d=8,num_tokens2d=8,num_quantizers=3,
                        attnj=True,attnt=True,device='cpu')
    config=dict(code_dim=16,cond_mode='text',latent_dim=16,ff_size=32,num_layers=1,num_heads=2,
                dropout=0.,clip_dim=8,opt=opt)
    with patch('motius.models.momask.network.mask_transformer.transformer._load_openai_clip_model',side_effect=lambda *args:TinyTextEncoder()), \
         patch('motius.models.mogents.network.transformer.transformer_ts._load_openai_clip_model',side_effect=lambda *args:TinyTextEncoder()):
        model.mask_transformer_aux=MaskTransformer(**config)
        model.mask_transformer_ts=MaskTransformer2D(**config)
        model.res_transformer_aux=ResidualTransformer(**config,shared_codebook=False,share_weight=True)
        model.res_transformer_ts=ResidualTransformer2D(**config,shared_codebook=False,share_weight=True)
    model.vq_model=nn.Identity();model.length_estimator=None
    model.register_buffer('mean',torch.zeros(263));model.register_buffer('std',torch.ones(263))
    return model


def batch(stage):
    shape=(2,5) if stage.startswith('aux') else (2,5,6)
    if stage.endswith('residual'):shape=(*shape,3)
    return dict(tokens=torch.randint(0,8,shape),text=['walk','jump'],token_lens=torch.tensor([5,3]))


def test_all_objectives_and_samplers():
    for stage in MoGenTSTrainer.MODULES:
        torch.manual_seed(17);model=bundle();trainer=MoGenTSTrainer(model,stage=stage)
        calls=[];handle=getattr(model,trainer.MODULES[stage]).register_forward_hook(lambda *args:calls.append(True))
        try:
            loss=trainer.train_step(batch(stage))['loss'];loss.backward()
            assert torch.isfinite(loss) and calls==[True]
            network=model.unwrap_network(getattr(model,trainer.MODULES[stage]))
            assert network.input_process.poseEmbedding.weight.grad.abs().sum()>0
        finally:handle.remove()
    model=bundle();lens=torch.tensor([5,3]);text=['walk','jump']
    aux=aux_base_generate(model.mask_transformer_aux,text,lens,3,2.)
    spatial=ts_base_generate(model.mask_transformer_ts,text,lens,3,2.,n_j=6)
    aux=aux_residual_generate(model.res_transformer_aux,aux,text,lens,cond_scale=2.)
    spatial=ts_residual_generate(model.res_transformer_ts,spatial,text,lens,cond_scale=2.)
    assert aux.shape==(2,5,3) and spatial.shape==(2,5,6,3)
    assert (aux[1,3:]==-1).all() and (spatial[1,3:]==-1).all()


def _worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        for stage in MoGenTSTrainer.MODULES:
            torch.manual_seed(17);model=bundle();trainer=MoGenTSTrainer(model,stage=stage)
            name=trainer.MODULES[stage];setattr(model,name,DistributedDataParallel(getattr(model,name)))
            optimizer=torch.optim.AdamW(getattr(model,name).parameters(),lr=1e-3)
            for step in range(2):
                torch.manual_seed(19+rank+step);optimizer.zero_grad()
                trainer.train_step(batch(stage))['loss'].backward();optimizer.step()
            parameter=model.unwrap_network(getattr(model,name)).input_process.poseEmbedding.weight.detach()
            values=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(values,parameter)
            torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_four_stage_training(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
