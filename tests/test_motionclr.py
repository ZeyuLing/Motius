"""Focused contract tests for the native MotionCLR integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from motius.models.motionclr import MOTIONCLR_SOURCE_REVISION, MotionCLR, MotionCLRBundle
from motius.models.motionclr.bundle import _download_hub_layout
from motius.pipelines.motionclr import MotionCLRPipeline
from motius.registry import MODEL_BUNDLES, PIPELINES


def _tiny_network_config():
    return {
        "input_feats": 263,
        "base_dim": 8,
        "dim_mults": [1],
        "dims": None,
        "adagn": True,
        "zero": True,
        "dropout": 0.0,
        "no_eff": True,
        "time_dim": 8,
        "latent_dim": 8,
        "cond_mask_prob": 0.1,
        "clip_dim": 8,
        "clip_version": "ViT-B/32",
        "text_latent_dim": 8,
        "text_ff_size": 16,
        "text_num_heads": 2,
        "activation": "gelu",
        "num_text_layers": 1,
        "self_attention": True,
        "vis_attn": False,
    }


def _stats():
    mean = np.linspace(-1.0, 1.0, 263, dtype=np.float32)
    std = np.linspace(0.5, 1.5, 263, dtype=np.float32)
    return mean, std


def test_motionclr_registration_and_source_revision():
    assert MODEL_BUNDLES.get("MotionCLRBundle") is MotionCLRBundle
    assert PIPELINES.get("MotionCLRPipeline") is MotionCLRPipeline
    assert MOTIONCLR_SOURCE_REVISION == "a6f44a791940682fe335c82f1b436bae05a1cebb"
    assert set(MotionCLRBundle.SUPPORTED_TASKS) == {"text_to_motion"}


def test_tiny_motionclr_has_official_state_names_and_pads_length():
    model = MotionCLR(**_tiny_network_config(), load_clip=False).eval()
    keys = set(model.state_dict())
    assert "embed_text.weight" in keys
    assert "textTransEncoder.layers.0.self_attn.in_proj_weight" in keys
    assert "unet.time_mlp.0.pe" in keys
    assert "unet.downs.0.0.conv1d.blocks.0.block.weight" in keys

    output = model(
        torch.randn(1, 17, 263),
        torch.tensor([5]),
        enc_text=torch.randn(1, 4, 8),
    )
    assert output.shape == (1, 17, 263)


def test_humanml263_normalize_denormalize_roundtrip():
    mean, std = _stats()
    bundle = MotionCLRBundle(
        network_config=_tiny_network_config(),
        mean=mean,
        std=std,
        load_model=False,
        load_clip=False,
    )
    motion = np.arange(3 * 263, dtype=np.float32).reshape(3, 263) / 100.0
    np.testing.assert_allclose(
        bundle.denormalize(bundle.normalize(motion)), motion, rtol=1e-6, atol=1e-6
    )

    tensor = torch.from_numpy(motion)
    torch.testing.assert_close(bundle.denormalize(bundle.normalize(tensor)), tensor)


def test_motionclr_artifact_roundtrip_strips_clip_weights(tmp_path: Path):
    mean, std = _stats()
    network = MotionCLR(**_tiny_network_config(), load_clip=False)
    network.clip_model = nn.Linear(1, 1)
    bundle = MotionCLRBundle(
        network_config=_tiny_network_config(),
        mean=mean,
        std=std,
        network=network,
        load_model=False,
        load_clip=False,
    )
    artifact = tmp_path / "artifact"
    bundle.save_pretrained(artifact)

    from safetensors.torch import load_file

    saved_state = load_file(str(artifact / "model.safetensors"))
    saved_keys = set(saved_state)
    assert saved_keys
    assert not any(key.startswith("clip_model.") for key in saved_keys)
    metadata = json.loads((artifact / "motionclr_config.json").read_text())
    assert metadata["source_revision"] == MOTIONCLR_SOURCE_REVISION
    assert metadata["official_files"]["checkpoint_sha256"].startswith("5852e139")
    assert metadata["load_clip"] is False
    assert metadata["text_encoder"] == {
        "name": "ViT-B/32",
        "stored_in_artifact": False,
        "path": None,
    }
    assert metadata["inference"]["torch_dtype"] == "float32"
    assert (artifact / "model_index.json").is_file()
    assert (artifact / "Mean.npy").is_file()
    assert (artifact / "Std.npy").is_file()
    assert (artifact / "LICENSE").is_file()

    reloaded = MotionCLRBundle.from_pretrained(artifact, load_clip=False)
    assert reloaded.network_config["base_dim"] == 8
    assert reloaded.network is not None
    assert reloaded.network.clip_model is None
    for key, value in reloaded.network.state_dict().items():
        torch.testing.assert_close(value, saved_state[key])


def test_motionclr_artifact_missing_files_fails_early(tmp_path: Path):
    artifact = tmp_path / "broken"
    artifact.mkdir()
    (artifact / "motionclr_config.json").write_text(
        json.dumps(
            {
                "model_type": "motionclr",
                "network": _tiny_network_config(),
                "weights": "model.safetensors",
                "statistics": {"mean": "Mean.npy", "std": "Std.npy"},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="missing weights, mean, std"):
        MotionCLRBundle.from_pretrained(artifact, load_model=False)


def test_motius_hub_layout_downloads_packaged_clip(tmp_path: Path, monkeypatch):
    import huggingface_hub

    captured = {}
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id: ["motionclr_config.json", "clip/ViT-B-32.pt"],
    )

    def snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert _download_hub_layout("org/motionclr") == {"artifact": tmp_path}
    assert "clip/**" in captured["allow_patterns"]


def test_motionclr_artifact_can_store_local_clip_file(tmp_path: Path):
    mean, std = _stats()
    clip_path = tmp_path / "ViT-B-32.pt"
    clip_path.write_bytes(b"local clip fixture")
    bundle = MotionCLRBundle(
        network_config=_tiny_network_config(),
        mean=mean,
        std=std,
        network=MotionCLR(**_tiny_network_config(), load_clip=False),
        load_model=False,
        load_clip=False,
        clip_path=str(clip_path),
    )
    artifact = tmp_path / "artifact-with-clip"
    bundle.save_pretrained(artifact)
    metadata = json.loads((artifact / "motionclr_config.json").read_text())
    assert metadata["text_encoder"]["stored_in_artifact"] is True
    assert metadata["text_encoder"]["path"] == "clip/ViT-B-32.pt"
    assert (artifact / "clip" / "ViT-B-32.pt").read_bytes() == b"local clip fixture"


def test_fp16_inference_keeps_openai_clip_in_float32():
    mean, std = _stats()
    network = MotionCLR(**_tiny_network_config(), load_clip=False)
    network.clip_model = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8))
    bundle = MotionCLRBundle(
        network_config=_tiny_network_config(),
        mean=mean,
        std=std,
        network=network,
        load_model=False,
        load_clip=False,
        torch_dtype="fp16",
    )
    assert bundle.network.embed_text.weight.dtype == torch.float16
    assert next(bundle.network.clip_model.parameters()).dtype == torch.float32


def test_official_nested_layout_parses_without_building_full_network(tmp_path: Path):
    release = tmp_path / "release"
    (release / "model").mkdir(parents=True)
    (release / "meta").mkdir()
    torch.save({"encoder": {}}, release / "model" / "latest.tar")
    mean, std = _stats()
    np.save(release / "meta" / "mean.npy", mean)
    np.save(release / "meta" / "std.npy", std)
    (release / "opt.txt").write_text(
        "base_dim: 512\n"
        "dim_mults: [2, 2, 2, 2]\n"
        "no_adagn: False\n"
        "no_eff: True\n"
        "self_attention: True\n"
        "text_latent_dim: 256\n"
    )
    bundle = MotionCLRBundle.from_pretrained(
        release,
        load_model=False,
        load_clip=False,
        verify_hashes=False,
    )
    assert bundle.checkpoint_path == str(release / "model" / "latest.tar")
    assert bundle.network_config["base_dim"] == 512
    assert bundle.network_config["self_attention"] is True


class _MockNetwork(nn.Module):
    input_feats = 263
    cond_mask_prob = 0.0

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode_text(self, captions, device):
        return torch.zeros(len(captions), 1, 1, device=device)

    def forward(self, sample, timesteps, enc_text=None):
        del timesteps, enc_text
        return torch.zeros_like(sample)


class _MockBundle(nn.Module):
    diffuser_name = "ddim"
    num_inference_steps = 2
    guidance_scale = 2.5
    inference_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.network = _MockNetwork()
        self.register_buffer("mean", torch.zeros(263))
        self.register_buffer("std", torch.ones(263))

    @property
    def device(self):
        return self.mean.device

    def to_device(self, device):
        self.to(device)
        return self

    def denormalize(self, motion):
        return motion * self.std + self.mean


class _MockScheduler:
    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps - 1, -1, -1, device=device)

    def step(self, prediction, timestep, sample, generator=None):
        del prediction, timestep, generator
        return SimpleNamespace(prev_sample=sample)


@pytest.mark.parametrize(
    "name,class_name",
    [
        ("dpmsolver", "DPMSolverMultistepScheduler"),
        ("ddpm", "DDPMScheduler"),
        ("ddim", "DDIMScheduler"),
        ("deis", "DEISMultistepScheduler"),
        ("pndm", "PNDMScheduler"),
    ],
)
def test_pipeline_builds_official_diffusers_schedulers(name, class_name):
    scheduler = MotionCLRPipeline._build_scheduler(name)
    assert type(scheduler).__name__ == class_name
    assert scheduler.config.num_train_timesteps == 1000
    assert scheduler.config.beta_schedule == "linear"
    assert scheduler.config.prediction_type == "sample"


def test_pipeline_lengths_seed_and_rng_isolation():
    pipe = MotionCLRPipeline(_MockBundle(), scheduler=_MockScheduler())
    first = pipe.infer_t2m(
        ["walk", "jump"],
        [7, 13],
        seed=17,
        return_normalized=True,
    )
    second = pipe.infer_t2m(
        ["walk", "jump"],
        [7, 13],
        seed=17,
        return_normalized=True,
    )
    changed = pipe.infer_t2m(
        ["walk", "jump"],
        [7, 13],
        seed=18,
        return_normalized=True,
    )
    assert [motion.shape for motion in first] == [(7, 263), (13, 263)]
    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[0], changed[0])

    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    pipe.infer_t2m(["walk"], [4], seed=5, return_normalized=True)
    torch.testing.assert_close(torch.rand(3), expected)


@pytest.mark.parametrize(
    "captions,lengths,message",
    [
        ([], [], "at least one"),
        (["walk"], [], "equal length"),
        (["walk"], [0], "in \\[1, 196\\]"),
        (["walk"], [197], "in \\[1, 196\\]"),
    ],
)
def test_pipeline_rejects_invalid_lengths(captions, lengths, message):
    pipe = MotionCLRPipeline(_MockBundle(), scheduler=_MockScheduler())
    with pytest.raises(ValueError, match=message):
        pipe.infer_t2m(captions, lengths)
