"""Exercise the real denoiser through the native training and sampling interfaces."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from motius.models.base_model_bundle import ModelBundle
from motius.models.intergen import InterGenBundle
from motius.models.intergen.network.models.nets import InterDiffusion
from motius.pipelines.intergen import InterGenPipeline
from motius.trainers.intergen import InterGenTrainer


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(INPUT_DIM=262, LATENT_DIM=16, FF_SIZE=32,
            NUM_LAYERS=1, NUM_HEADS=2, DROPOUT=0., ACTIVATION="gelu", CFG_WEIGHT=3.5,
            DIFFUSION_STEPS=10, BETA_SCHEDULER="cosine", MOTION_REP="global",
            SAMPLER="uniform", T_BAR=7, STRATEGY="ddim2")
        self.decoder = InterDiffusion(self.cfg)
        self.text = nn.Embedding(1, 768)
        # Released denoisers start from zero output; nonzero weights also check text gradients.
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.01)

    def encode_text(self, texts):
        return self.text.weight.expand(len(texts), -1)

    def forward(self, x, timesteps, mask=None, cond=None):
        return self.decoder(x, timesteps, mask=mask, cond=cond)


def bundle(tmp_path):
    model = InterGenBundle.__new__(InterGenBundle)
    ModelBundle.__init__(model)
    model._model = TinyNetwork()
    model._save_ckpt_modules = ["_model"]
    model._trainable_modules = ["_model"]
    model.mean_path, model.std_path = tmp_path / 'mean.npy', tmp_path / 'std.npy'
    np.save(model.mean_path, np.zeros(262, dtype=np.float32))
    np.save(model.std_path, np.ones(262, dtype=np.float32))
    model.register_buffer('mean', torch.zeros(262))
    model.register_buffer('std', torch.ones(262))
    return model


def batch():
    return dict(text=["two people shake hands", "two people hug"],
                motions=torch.randn(2, 5, 524), motion_lens=torch.tensor([5, 3]))


def test_native_training_and_sampling(tmp_path):
    torch.manual_seed(7)
    model = bundle(tmp_path)
    trainer = InterGenTrainer(model)
    calls = []
    handle = model.model.register_forward_hook(lambda *args: calls.append(True))
    try:
        result = trainer.train_step(batch())
        result['loss'].backward()
        assert calls == [True]
        assert torch.isfinite(result['loss'])
        assert model.model.text.weight.grad.abs().sum() > 0
        assert not hasattr(model, 'generate')
        pipeline = InterGenPipeline(model)
        a = pipeline('two people shake hands', motion_len=15, seed=7, return_numpy=False)
        b = pipeline('two people shake hands', motion_len=15, seed=7, return_numpy=False)
        assert a.shape == (1, 15, 2, 262)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all()
        with pytest.raises(ValueError): pipeline([], motion_len=15)
    finally:
        handle.remove()


def _ddp_worker(rank, path, rendezvous):
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        model = bundle(Path(path) / str(rank))
        model._model = DistributedDataParallel(model._model)
        trainer = InterGenTrainer(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(2):
            torch.manual_seed(21 + rank + step)
            np.random.seed(21 + rank + step)
            optimizer.zero_grad()
            trainer.train_step(batch())['loss'].backward()
            optimizer.step()
        # Text encoding happens before the wrapped denoising forward; its gradients must sync too.
        for parameter in (model.network.text.weight, model.network.decoder.net.out.linear.weight):
            values = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(values, parameter.detach())
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_process_training_syncs_both_networks(tmp_path):
    for rank in range(2): (tmp_path / str(rank)).mkdir()
    torch.multiprocessing.spawn(_ddp_worker,
        args=(str(tmp_path), f'file://{tmp_path / "rendezvous"}'), nprocs=2, join=True)
