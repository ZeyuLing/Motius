"""Actual T5/VQ forward graphs and wrapped language-model training."""
import torch
from transformers import T5Config
from motius.models.base_model_bundle import ModelBundle
from motius.models.motiongpt import MotionGPTBundle
from motius.models.motiongpt.language import MotionGPTLanguageNetwork
from motius.models.motiongpt.vq import VQVae
from motius.trainers.motiongpt import MotionGPTTrainer


def bundle():
    value=MotionGPTBundle.__new__(MotionGPTBundle);ModelBundle.__init__(value)
    value.lm=MotionGPTLanguageNetwork(T5Config(vocab_size=64,d_model=16,d_kv=8,d_ff=32,
        num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,pad_token_id=0,tie_word_embeddings=False))
    value.vae=VQVae(nfeats=263,code_num=8,code_dim=16,output_emb_width=16,down_t=2,stride_t=2,width=32,depth=1)
    value.register_buffer('mean',torch.zeros(263));value.register_buffer('std',torch.ones(263))
    return value


def batch():
    return dict(input_ids=torch.randint(1,64,(2,8)),attention_mask=torch.ones(2,8,dtype=torch.long),
                labels=torch.randint(1,64,(2,6)))


def test_language_and_vq_stages_call_neural_roots():
    for stage in ['lm_instruct','vae']:
        torch.manual_seed(17);value=bundle();trainer=MotionGPTTrainer(value,stage=stage)
        name='vae' if stage=='vae' else 'lm';calls=[]
        handle=getattr(value,name).register_forward_hook(lambda *args:calls.append(True))
        try:
            result=trainer.train_step(dict(motion=torch.randn(2,32,263)) if stage=='vae' else batch())
            result['loss'].backward()
            assert calls==[True] and torch.isfinite(result['loss'])
            assert all(torch.isfinite(p.grad).all() for p in getattr(value,name).parameters() if p.grad is not None)
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(value,name).parameters())
            assert all(p.grad is None for p in getattr(value,'lm' if name=='vae' else 'vae').parameters())
        finally:handle.remove()


def _worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(17);value=bundle();trainer=MotionGPTTrainer(value)
        value.lm=DistributedDataParallel(value.lm)
        optimizer=torch.optim.AdamW(value.lm.parameters(),lr=1e-3)
        for step in range(2):
            torch.manual_seed(19+rank+step);optimizer.zero_grad()
            trainer.train_step(batch())['loss'].backward();optimizer.step()
        parameter=value.unwrap_network(value.lm).language_model.lm_head.weight.detach()
        values=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(values,parameter)
        torch.testing.assert_close(*values,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_language_training(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
