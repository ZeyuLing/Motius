"""Real UniMuMo token/caption graphs and distributed parameter updates."""
import copy
import torch
from torch import nn
from transformers import T5Config,T5ForConditionalGeneration
from motius.models.unimumo.bundle import UniMuMoBundle,DEFAULT_UNIMUMO_CONFIG
from motius.trainers.unimumo import UniMuMoTrainer
from test_unimumo import _generator_config


def bundle():
    config=copy.deepcopy(DEFAULT_UNIMUMO_CONFIG);config['generator']=_generator_config();config['text_hidden_size']=8
    config['motion_codec'].update(latent_dim=8,encoder_channels=[16,16,16,16,8],decoder_channels=[8,16,16,16,16],depth_per_block=1)
    captioner=T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=8,d_kv=4,d_ff=16,num_layers=1,
        num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,pad_token_id=0))
    return UniMuMoBundle(config,audio_codec=nn.Identity(),text_encoder=nn.Identity(),captioner=captioner,
        tokenizer_path=None,mean=torch.zeros(263),std=torch.ones(263))


def batch(stage):
    values=dict(music_codes=torch.randint(0,8,(2,2,6)),motion_codes=torch.randint(0,8,(2,2,6)),
        condition=torch.randn(2,4,8),condition_mask=torch.tensor([[[True,True,False,False],[False,False,True,True]]]*2))
    if stage=='train_caption':values.update(caption_mode='motion',labels=torch.randint(1,32,(2,5)),decoder_attention_mask=torch.ones(2,5,dtype=torch.long))
    return values


def test_two_actual_training_graphs_use_prepared_core():
    for stage in ['train_music_motion','train_caption']:
        torch.manual_seed(17);value=bundle();trainer=UniMuMoTrainer(value,stage=stage);value.train()
        calls=[];handle=value.core.register_forward_hook(lambda *args:calls.append(True))
        try:
            result=trainer.train_step(batch(stage));result['loss'].backward()
            assert calls==[True] and torch.isfinite(result['loss'])
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in value.core.parameters())
            assert all(p.grad is None for p in value.motion_codec.parameters())
            if stage=='train_caption':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in value.captioner.parameters())
        finally:handle.remove()


def _worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        for stage in ['train_music_motion','train_caption']:
            torch.manual_seed(17);value=bundle();trainer=UniMuMoTrainer(value,stage=stage);value.train()
            for name in value._trainable_modules:setattr(value,name,DistributedDataParallel(getattr(value,name),find_unused_parameters=True))
            optimizer=torch.optim.AdamW([p for p in value.parameters() if p.requires_grad],lr=1e-3)
            for step in range(2):
                torch.manual_seed(19+rank+step);optimizer.zero_grad()
                trainer.train_step(batch(stage))['loss'].backward();optimizer.step()
            for name in value._trainable_modules:
                parameter=next(p.detach() for p in getattr(value,name).parameters() if p.requires_grad)
                values=[torch.empty_like(parameter) for _ in range(2)];dist.all_gather(values,parameter)
                torch.testing.assert_close(*values,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_token_and_caption_training(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
