"""Native InterMask owns token sampling and training outside the network."""
from types import SimpleNamespace

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.models.intermask import InterMaskBundle
from motius.models.intermask.network.models.mask_transformer.transformer import MaskTransformer
from motius.pipelines.intermask.sampling import generate, generate_reaction
from motius.trainers.intermask import InterMaskTrainer
from motius.motion.representation.intermask import pack_motion, unpack_motion


def bundle():
    model = InterMaskBundle.__new__(InterMaskBundle)
    ModelBundle.__init__(model)
    model.vq_model = torch.nn.Linear(1,1)
    model.transformer = MaskTransformer(code_dim=16, cond_mode='uncond', latent_dim=16,
        ff_size=32, num_layers=1, num_heads=2, dropout=0.,
        opt=SimpleNamespace(num_tokens=8, interaction_mask_prob=0., step_unroll=1.))
    return model


def batch():
    return dict(tokens=torch.randint(0,8,(2,30)), text=['a','b'], token_lens=torch.tensor([3,2]))


def test_train_sampling_and_codec():
    torch.manual_seed(7)
    model = bundle()
    calls=[]
    handle=model.transformer.register_forward_hook(lambda *args: calls.append(True))
    try:
        out = InterMaskTrainer(model).train_step(batch())
        assert len(calls)==2  # initial masked prediction and the step-unroll refinement
        assert torch.isfinite(out['loss'])
        out['loss'].backward()
        assert model.transformer.token_emb.weight.grad.abs().sum()>0
        lengths=torch.tensor([3,2])
        for fn,extra in [(generate,()),(generate_reaction,(torch.randint(0,8,(2,15)),))]:
            torch.manual_seed(9)
            ids=fn(model.transformer,['a','b'],*extra,lengths,4,2.)
            assert ids.shape==(2,30)
            assert ((ids>=-1)&(ids<8)).all()
        motion=torch.randn(2,8,262)
        unpacked=unpack_motion(pack_motion(motion,'interhuman',22),'interhuman')
        torch.testing.assert_close(unpacked[...,:258],motion[...,:258],rtol=0,atol=0)
        assert torch.count_nonzero(unpacked[...,-4:])==0
    finally:handle.remove()


def _ddp_worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(7)
        model=bundle()
        model.transformer=DistributedDataParallel(model.transformer)
        trainer=InterMaskTrainer(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for step in range(2):
            torch.manual_seed(33+rank+step)
            optimizer.zero_grad()
            trainer.train_step(batch())['loss'].backward()
            optimizer.step()
        parameter=model.token_network.token_emb.weight.detach()
        values=[torch.empty_like(parameter) for _ in range(2)]
        dist.all_gather(values,parameter)
        torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_step_unroll_training(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
