from types import SimpleNamespace
import torch
from accelerate import Accelerator
from motius.models.base_model_bundle import ModelBundle
from motius.runner.accelerate_runner import AccelerateRunner, _BundleOrphanCheckpoint


class _Bundle(ModelBundle):
    def __init__(self):
        super().__init__()
        self.active = torch.nn.Linear(1, 1, bias=False)
        self.frozen = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.register_buffer("mean", torch.zeros(1))
        self._trainable_modules = ["active"]
        self._frozen_modules = ["frozen"]
        self._save_ckpt_modules = ["active", "frozen"]

    def forward(self, value):
        return self.active(self.frozen(value))


def _runner(directory):
    bundle = _Bundle()
    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.AdamW(bundle.active.parameters(), lr=0.001)
    bundle.active, optimizer = accelerator.prepare(bundle.active, optimizer)
    accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
    return AccelerateRunner(
        bundle,
        SimpleNamespace(),
        None,
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


def test_full_resume_restores_saved_frozen_networks(tmp_path):
    original = _runner(tmp_path / "original")
    with torch.no_grad():
        original.bundle.active.weight.fill_(3.1)
        original.bundle.frozen.weight.fill_(2.5)
        original.bundle.mean.fill_(0.25)
    original.global_step = 2
    expected = original.bundle(torch.tensor([[2.0]])).detach()
    original.save_checkpoint()
    restored = _runner(tmp_path / "restored")
    with torch.no_grad():
        restored.bundle.frozen.weight.zero_()
    restored._load(str(tmp_path / "original/checkpoint-iter_2"), load_scope="full")
    assert torch.equal(restored.bundle.frozen.weight, original.bundle.frozen.weight)
    assert torch.equal(restored.bundle.mean, original.bundle.mean)
    assert torch.equal(restored.bundle(torch.tensor([[2.0]])), expected)
    assert restored.global_step == 2
