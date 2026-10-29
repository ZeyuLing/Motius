"""Actual dual-modality/VAE/diffusion training through native prepared roots."""
import torch
from transformers import GPT2Config,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from motius.configuration.motiongpt3 import TokenLayout
from motius.models.base_model_bundle import ModelBundle
from motius.models.motiongpt3 import MotionGPT3Bundle
from motius.models.motiongpt3.networks import MotionGPT3LanguageNetwork
from motius.models.motiongpt3.vae import MldVae
from motius.motion.conditioning.motiongpt3 import MotionGPT3Processor
from motius.trainers.motiongpt3 import MotionGPT3Trainer


def bundle():
    value=MotionGPT3Bundle.__new__(MotionGPT3Bundle);ModelBundle.__init__(value)
    value.options=dict(stage='lm_pretrain',motion_holder_repeat=4,holder_num_in_input=4,motion_holder_seq_mode='withse',
        with_hid_norm=False,with_vae_latent_norm=True,multi_hidden=True,guidance_scale=3.,guidance_uncondp=.5,
        fake_latent_mode='learnable_zero',diffusion_batch_mul=2,max_length=32,framerate=20.,down_t=4,
        predict_ratio=.2,inbetween_ratio=.25,quota_ratio=.5,noise_density=.15,mean_noise_span_length=3)
    value.diffusion_config=dict(width=32,depth=1,multi_hidden=True,num_sampling_steps='10',noise_schedule='scaled_linear',learn_sigma=False,sigma_small=True,grad_checkpointing=False)
    value.diffusion_options={k:value.diffusion_config[k] for k in ['noise_schedule','learn_sigma','sigma_small']}
    value.vae=MldVae(ablation={'MLP_DIST':False,'PE_TYPE':'mld'},nfeats=263,latent_dim=[1,16],ff_size=32,num_layers=1,num_heads=2,dropout=0.,arch='encoder_decoder')
    value.lm=MotionGPT3LanguageNetwork(GPT2Config(vocab_size=66,n_embd=16,n_layer=1,n_head=2,n_positions=64,
        resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.),TokenLayout(64),value.options,value.diffusion_config,16,1,set())
    value.diffusion_head=value.lm.diffloss;del value.lm.diffloss
    value.diffusion_head.register_parameter('fake_latent',value.lm.fake_latent);del value.lm.fake_latent
    value.register_buffer('mean',torch.zeros(263));value.register_buffer('std',torch.ones(263))
    return value


def processor(options):
    vocab={'<unk>':0,'Generate':1,'motion':2,':':3,'walk':4,'jump':5}
    vocab.update({f'token{i}':i for i in range(6,63)});vocab['<endoftext>']=63
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',eos_token='<endoftext>',bos_token='<endoftext>')
    return MotionGPT3Processor(tokenizer,options,torch.device('cpu'))


def batch():
    return dict(motion=torch.randn(2,16,263),length=[16,12],text=['walk','jump'],tasks=[
        {'input':['Generate motion: <Caption_Placeholder>'],'output':['<Motion_Placeholder>'],'class':'t2m'}]*2)


def test_actual_three_network_training_graph_and_vae():
    for stage in ['lm_pretrain','vae']:
        torch.manual_seed(17);value=bundle();trainer=MotionGPT3Trainer(value,stage=stage)
        trainer._processor=processor(value.options);value.train()
        calls=[];handles=[]
        for name in value._trainable_modules:
            handles.append(getattr(value,name).register_forward_hook(lambda module,inputs,result,key=name:calls.append(key)))
        try:
            result=trainer.train_step(batch());result['loss'].backward()
            assert calls==value._trainable_modules and torch.isfinite(result['loss'])
            for name in value._trainable_modules:
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(value,name).parameters())
            if stage!='vae':assert result['diffusion_loss']>0
        finally:
            for handle in handles:handle.remove()


def _worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(17);value=bundle();trainer=MotionGPT3Trainer(value);trainer._processor=processor(value.options)
        value.train()
        for name in value._trainable_modules:setattr(value,name,DistributedDataParallel(getattr(value,name),find_unused_parameters=True))
        optimizer=torch.optim.AdamW([p for p in value.parameters() if p.requires_grad],lr=1e-3)
        for step in range(2):
            torch.manual_seed(19+rank+step);optimizer.zero_grad()
            trainer.train_step(batch())['loss'].backward();optimizer.step()
        for name in value._trainable_modules:
            parameter=next(p.detach() for p in getattr(value,name).parameters() if p.requires_grad)
            values=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(values,parameter)
            torch.testing.assert_close(*values,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_language_and_diffusion_training(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
