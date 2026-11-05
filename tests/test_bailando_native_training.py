"""Bailando training must route neural calls through its prepared shared module."""
import torch
from motius.models.bailando import BailandoBundle
from motius.trainers.bailando import BailandoTrainer
from test_bailando import _tiny_config


def test_token_trainer_backward():
    bundle=BailandoBundle(config=_tiny_config())
    trainer=BailandoTrainer(bundle,stage='gpt')
    bundle.train()
    calls=[]
    handle=bundle.gpt.register_forward_hook(lambda *args:calls.append(True))
    try:
        result=trainer.train_step(dict(tokens=(torch.randint(0,8,(2,3)),torch.randint(0,8,(2,3))),
            targets=(torch.randint(0,8,(2,3)),torch.randint(0,8,(2,3))),music_features=torch.randn(2,3,438)))
        result['loss'].backward()
        assert calls==[True] and torch.isfinite(result['loss'])
        assert bundle.gpt.gpt_head.head_up.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in bundle.gpt.critic_net.parameters())
    finally:handle.remove()


def test_root_velocity_trainer_backward():
    bundle=BailandoBundle(config=_tiny_config())
    trainer=BailandoTrainer(bundle,stage='root')
    bundle.train()
    calls=[]
    handle=bundle.vqvae.register_forward_hook(lambda *args:calls.append(True))
    try:
        result=trainer.train_step(dict(motion=torch.randn(2,8,72)))
        result['loss'].backward()
        assert calls==[True] and torch.isfinite(result['loss'])
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in bundle.vqvae.vqvae_down.decoders_root.parameters())
        assert all(p.grad is None for p in bundle.vqvae.vqvae_up.parameters())
        assert all(p.grad is None for p in bundle.vqvae.vqvae_down.encoders.parameters())
    finally:handle.remove()
