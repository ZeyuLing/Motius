"""MoMask objectives must execute the wrapped neural forward in both stages."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.momask.bundle import MoMaskBundle
from motius.models.momask.network.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from motius.trainers.momask import MoMaskTrainer
from motius.pipelines.momask.decoding import base_generate, residual_generate, base_edit


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(49408, 8)

    def encode_text(self, tokens):
        return self.embedding(tokens).mean(1)


def bundle():
    model = MoMaskBundle.__new__(MoMaskBundle)
    ModelBundle.__init__(model)
    opt = SimpleNamespace(num_tokens=8, num_quantizers=3, code_dim=8, device='cpu')
    with patch('motius.models.momask.network.mask_transformer.transformer._load_openai_clip_model',
               side_effect=lambda *args: TinyTextEncoder()):
        config = dict(code_dim=8, cond_mode='text', latent_dim=8, ff_size=16,
                      num_layers=1, num_heads=2, dropout=0., clip_dim=8, opt=opt)
        model.t2m_transformer = MaskTransformer(**config)
        model.res_transformer = ResidualTransformer(**config, shared_codebook=False, share_weight=True)
    model.vq_model = nn.Linear(1,1)
    model.length_estimator = None
    model.register_buffer('mean',torch.zeros(263))
    model.register_buffer('std',torch.ones(263))
    return model


def batch(stage):
    shape=(2,5) if stage=='base' else (2,5,3)
    return dict(tokens=torch.randint(0,8,shape),text=['walk','jump'],token_lens=torch.tensor([5,3]))


def train(model, stage):
    trainer=MoMaskTrainer(model,stage=stage)
    calls=[]
    handle=getattr(model,trainer.module_name).register_forward_hook(lambda *args:calls.append(True))
    try:
        loss=trainer.train_step(batch(stage))['loss']
        loss.backward()
        assert torch.isfinite(loss) and calls==[True]
        network=model.unwrap_network(getattr(model,trainer.module_name))
        assert network.input_process.poseEmbedding.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in network.clip_model.parameters())
        return trainer
    finally:handle.remove()


def test_native_base_and_residual_training_and_sampling():
    torch.manual_seed(17)
    for stage in ('base','residual'):
        train(bundle(),stage)
    model=bundle()
    lengths=torch.tensor([5,3])
    ids=base_generate(model.t2m_transformer,['walk','jump'],lengths,3,2.)
    edited=base_edit(model.t2m_transformer,['walk','jump'],ids.clamp_min(0),lengths,3,2.,edit_mask=torch.ones_like(ids,dtype=torch.bool))
    codes=residual_generate(model.res_transformer,edited,['walk','jump'],lengths,cond_scale=2.)
    assert codes.shape==(2,5,3)
    assert (codes[1,3:]==-1).all()
    assert ((codes>=-1)&(codes<8)).all()


def _ddp_worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        for stage in ('base','residual'):
            torch.manual_seed(17)
            model=bundle()
            trainer=MoMaskTrainer(model,stage=stage)
            name=trainer.module_name
            setattr(model,name,DistributedDataParallel(getattr(model,name)))
            optimizer=torch.optim.AdamW(getattr(model,name).parameters(),lr=1e-3)
            for step in range(2):
                torch.manual_seed(19+rank+step)
                optimizer.zero_grad()
                trainer.train_step(batch(stage))['loss'].backward()
                optimizer.step()
            parameter=model.unwrap_network(getattr(model,name)).input_process.poseEmbedding.weight.detach()
            values=[torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(values,parameter)
            torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_native_training(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
