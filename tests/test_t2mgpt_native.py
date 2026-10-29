"""Stage selection must preserve the original fixed positional parameters."""

from types import SimpleNamespace
import torch
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.t2mgpt import T2MGPTBundle
from motius.models.t2mgpt.network.t2m_trans import Text2Motion_Transformer
from motius.models.t2mgpt.network.vqvae import HumanVQVAE
from motius.trainers.t2mgpt import T2MGPTTrainer


class TextTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(49408, 8)
        self.requires_grad_(False)

    def encode_text(self, tokens):
        return self.embedding(tokens).mean(1)


def bundle():
    model = T2MGPTBundle.__new__(T2MGPTBundle)
    ModelBundle.__init__(model)
    model.gpt = Text2Motion_Transformer(
        num_vq=8,
        embed_dim=16,
        clip_dim=8,
        block_size=8,
        num_layers=1,
        n_head=2,
        drop_out_rate=0.0,
    )
    options = SimpleNamespace(dataname="t2m", quantizer="ema_reset", mu=0.99)
    model.vqvae = HumanVQVAE(options, 8, 8, 8, 2, 2, 16, 1, 3, "relu", None)
    model.clip_model = TextTower()
    model.register_buffer("mean", torch.zeros(263))
    model.register_buffer("std", torch.ones(263))
    return model


def test_source_frozen_positions_and_stage_gradients():
    torch.manual_seed(11)
    model = bundle()
    fixed = [
        parameter
        for name, parameter in model.gpt.named_parameters()
        if "pos_embed.embed" in name
    ]
    assert fixed and all(not parameter.requires_grad for parameter in fixed)
    gpt = T2MGPTTrainer(model, pkeep=0.5, if_maxtest=True)
    model.train()
    tokens = torch.randint(0, 8, (2, 6))
    tokens[:, -1] = 8
    result = gpt.train_step(
        dict(tokens=tokens, token_lens=torch.tensor([5, 4]), text=["walk", "jump"])
    )
    result["loss"].backward()
    assert all(
        parameter.grad is None and not parameter.requires_grad for parameter in fixed
    )
    assert model.gpt.trans_head.head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.clip_model.parameters())
    model.zero_grad(set_to_none=True)
    vq = T2MGPTTrainer(model, stage="vqvae")
    model.train()
    vq.train_step(dict(motion=torch.randn(2, 20, 263)))["loss"].backward()
    assert model.vqvae.vqvae.encoder.model[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.gpt.parameters())
    T2MGPTTrainer(model)
    assert all(not parameter.requires_grad for parameter in fixed)
