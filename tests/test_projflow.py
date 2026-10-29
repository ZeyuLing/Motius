import json
from pathlib import Path

import numpy as np
import pytest
import torch

from motius.pipelines.projflow.sampler import ProjFlowSampler
from motius.pipelines.auto import PIPELINE_CLASS_PATHS
from motius.pipelines.projflow import ProjFlowPipeline


ROOT = Path(__file__).resolve().parents[1]


def test_projflow_sampler_enforces_hard_joint_positions():
    torch.manual_seed(3)
    initial = torch.randn(1, 3, 4, 22)
    mask = torch.zeros_like(initial)
    mask[:, :, 1, 20] = 1
    target = torch.zeros_like(initial)
    target[:, :, 1, 20] = torch.tensor([0.4, 1.2, -0.7])

    output = ProjFlowSampler().sample_projflow(num_steps=2)(
        initial,
        lambda value, time, **kwargs: torch.zeros_like(value),
        A=mask,
        y=target,
    )[-1]

    torch.testing.assert_close(
        output[:, :, 1, 20],
        target[:, :, 1, 20],
        atol=2e-6,
        rtol=0,
    )


def test_projflow_temporal_protocol_maps_to_exact_observed_frames():
    assert ProjFlowPipeline._frames_for_mode(
        "start_1f", 100, prefix_ratio=0.2, boundary_ratio=0.1, keyframes=None
    ) == [0]
    assert ProjFlowPipeline._frames_for_mode(
        "pre20", 100, prefix_ratio=0.2, boundary_ratio=0.1, keyframes=None
    ) == list(range(20))
    assert ProjFlowPipeline._frames_for_mode(
        "both_1f", 100, prefix_ratio=0.2, boundary_ratio=0.1, keyframes=None
    ) == [0, 99]
    assert ProjFlowPipeline._frames_for_mode(
        "adaptive_keyframes",
        100,
        prefix_ratio=0.2,
        boundary_ratio=0.1,
        keyframes=[0, 50, 99],
    ) == [0, 50, 99]
    assert ProjFlowPipeline._frames_for_mode(
        "mib10", 100, prefix_ratio=0.2, boundary_ratio=0.1, keyframes=None
    ) == list(range(10)) + list(range(90, 100))
    assert ProjFlowPipeline._frames_for_mode(
        "sparse", 100, prefix_ratio=0.2, boundary_ratio=0.1, keyframes=[0, 30, 60]
    ) == [0, 30, 60]


def test_projflow_lengths_are_validated_instead_of_silently_clipped():
    assert ProjFlowPipeline._validate_lengths([1, 196]) == [1, 196]
    with pytest.raises(ValueError, match="positive frame counts"):
        ProjFlowPipeline._validate_lengths([0])
    with pytest.raises(ValueError, match="at most 196"):
        ProjFlowPipeline._validate_lengths([197])


def test_projflow_hml263_output_preserves_requested_frame_count():
    joints = np.zeros((7, 22, 3), dtype=np.float32)
    joints[:, :, 1] = np.arange(22, dtype=np.float32)[None] * 0.03
    joints[:, 0, 2] = np.arange(7, dtype=np.float32) * 0.01

    encoded = ProjFlowPipeline._as_hml263(joints)

    assert encoded.shape == (7, 263)
    assert np.isfinite(encoded).all()


def test_projflow_stats_match_the_official_humanml3d_joint_space():
    assets = ROOT / "motius" / "motion" / "assets" / "projflow"

    np.testing.assert_allclose(
        np.load(assets / "joints_mean.npy"),
        [0.0001634518, 0.37663162, 0.31019124],
    )
    np.testing.assert_allclose(
        np.load(assets / "joints_std.npy"),
        [0.5147618, 0.6101322, 0.8715584],
    )


def test_projflow_is_a_trusted_self_describing_pipeline():
    assert PIPELINE_CLASS_PATHS["ProjFlowPipeline"] == (
        "motius.pipelines.projflow.ProjFlowPipeline"
    )
    assert ProjFlowPipeline.BUNDLE_CLS == "motius.models.projflow.ProjFlowBundle"


def test_projflow_runtime_has_no_external_repo_dependency():
    sources = "\n".join(
        path.read_text()
        for path in (ROOT / "motius" / "models" / "projflow").rglob("*.py")
    ) + "\n" + "\n".join(
        path.read_text()
        for path in (ROOT / "motius" / "pipelines" / "projflow").rglob("*.py")
    )

    assert "ref_repo" not in sources
    assert "repo_path" not in sources
    assert "from models." not in sources
    assert "from diffusions." not in sources


def test_projflow_release_contract_matches_checkpoint_and_temporal_results():
    manifest = json.loads(
        (ROOT / "docs" / "model_zoo" / "release_manifest.json").read_text()
    )["models"]["projflow"]
    assert manifest["checkpoint"] == (
        "https://huggingface.co/ZeyuLing/motius-projflow-humanml3d"
    )
    assert len(manifest["metrics"]) == 8
    assert len(manifest["preview_cases"]) == 3

    results = json.loads(
        (
            ROOT
            / "docs"
            / "leaderboards"
            / "hf_space_temporal_condition"
            / "temporal_control_results.json"
        ).read_text()
    )
    assert len(results["settings"]) == 8
    for setting in results["settings"]:
        projflow = [
            row for row in setting["methods"] if row["method_id"] == "projflow"
        ]
        assert len(projflow) == 1
        assert projflow[0]["samples"] == 4012
        assert np.isfinite(projflow[0]["metrics"]["fid"])
        assert setting["methods"][0]["method_id"] == "gt"


def test_native_projflow_trainer_backpropagates():
    from motius.trainers.projflow import ProjFlowTrainer
    class Net(torch.nn.Module):
        patches_per_frame = 1
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.clip_model = torch.nn.Linear(2, 2)
        def encode_text(self, tokens):
            return torch.zeros(len(tokens), 2)
        def forward(self, noisy, times, **kwargs):
            return noisy * self.weight
    class Bundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Net()
        @property
        def device(self):
            return self.net.weight.device
    bundle = Bundle()
    trainer = ProjFlowTrainer(bundle)
    out = trainer.train_step({'motion': torch.randn(2, 8, 22, 3), 'motion_lens': [8, 6],
                              'text': torch.zeros(2, 77, dtype=torch.long)})
    out['loss'].backward()
    assert torch.isfinite(out['loss'])
    assert bundle.net.weight.grad is not None
    assert not bundle.net.clip_model.weight.requires_grad
