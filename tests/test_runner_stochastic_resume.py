"""Resume shuffled batches without advancing the checkpoint's model RNG."""

import random
import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, TensorDataset
from motius.runner.accelerate_runner import AccelerateRunner, _BundleOrphanCheckpoint
from motius.trainers.base_trainer import BaseTrainer
from test_runner_frozen_resume import _Bundle


class _Trainer(BaseTrainer):
    def train_step(self, batch):
        inputs = batch[0]
        target = torch.randn_like(inputs) + random.random() + np.random.random()
        return {"loss": (self.bundle(inputs) - target).square().mean()}


def _runner(directory):
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=17
        ),
    )
    set_seed(47)
    bundle = _Bundle()
    trainer = _Trainer(bundle)
    optimizer = torch.optim.AdamW(bundle.active.parameters(), lr=0.001)
    loader = DataLoader(
        TensorDataset(torch.arange(20).float().reshape(-1, 1)),
        shuffle=True,
        batch_size=2,
    )
    bundle.active, optimizer, loader = accelerator.prepare(
        bundle.active, optimizer, loader
    )
    accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
    return AccelerateRunner(
        bundle,
        trainer,
        loader,
        None,
        {"default": optimizer},
        {},
        accelerator,
        [],
        [],
        [],
        dict(by_epoch=False, max_iters=2),
        str(directory),
    )


def test_stochastic_resume_equals_uninterrupted_updates(tmp_path):
    original = _runner(tmp_path / "original")
    original._train_by_iter(dict(max_iters=2), None)
    original.save_checkpoint()
    resumed = _runner(tmp_path / "resumed")
    resumed._load(str(tmp_path / "original/checkpoint-iter_2"), load_scope="full")
    resumed._train_by_iter(dict(max_iters=3), None)
    expected = _runner(tmp_path / "uninterrupted")
    expected._train_by_iter(dict(max_iters=3), None)
    for name, value in resumed.bundle.state_dict().items():
        assert torch.equal(value, expected.bundle.state_dict()[name]), name
    left = resumed.optimizers["default"].state_dict()
    right = expected.optimizers["default"].state_dict()
    assert left["param_groups"] == right["param_groups"]
    for index, state in left["state"].items():
        for name, value in state.items():
            assert torch.equal(value, right["state"][index][name]), name
