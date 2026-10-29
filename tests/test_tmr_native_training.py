"""The native TMR trainer must execute the shared network through its forward."""
from pathlib import Path

import torch

from motius.models.tmr import TMRBundle
from motius.trainers.tmr import TMRTrainer


def bundle():
    return TMRBundle(motion_nfeats=6, text_nfeats=8,
        arch=dict(latent_dim=16, ff_size=32, num_layers=1, num_heads=2, dropout=0.0))


def batch():
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2, [True] * 4 + [False]])
    return dict(text_x_dict=dict(x=torch.randn(3, 5, 8), mask=mask),
                motion_x_dict=dict(x=torch.randn(3, 5, 6), mask=mask), sent_emb=torch.eye(3))


def test_loss_and_backward_enter_network_forward():
    model = bundle()
    calls = []
    handle = model.tmr.register_forward_hook(lambda *args: calls.append(True))
    try:
        trainer = TMRTrainer(model, log_prefix="custom")
        result = trainer.train_step(batch())
        assert calls == [True]
        assert "custom_contrastive" in result
        result["loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum() for value in gradients) > 0
        assert not hasattr(model, "compute_loss")
    finally:
        handle.remove()


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        model = bundle()
        model.tmr = DistributedDataParallel(model.tmr)
        trainer = TMRTrainer(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(2):
            torch.manual_seed(40 + rank + step)
            optimizer.zero_grad()
            trainer.train_step(batch())["loss"].backward()
            optimizer.step()
        parameter = next(model.tmr.parameters()).detach()
        received = [torch.empty_like(parameter) for _ in range(2)]
        dist.all_gather(received, parameter)
        torch.testing.assert_close(received[0], received[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_process_training_synchronizes_parameters(tmp_path: Path):
    torch.multiprocessing.spawn(_ddp_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
