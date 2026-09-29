import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

from motius.models.hymotion_v2m.bundle import (
    HYMOTION_V2M_ARTIFACT_FORMAT,
    HyMotionV2MBundle,
)
from motius.models.hymotion_v2m.vendor.hymotion.bodymodels.smpl_skeleton import (
    SMPLMesh,
    SMPLSkeleton,
)


class _BodyBuffers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("j_template", torch.ones(2))


class _ArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_transformer = torch.nn.Linear(2, 2)
        self.body_model = _BodyBuffers()
        self.vertex_loss = torch.nn.Module()
        self.vertex_loss.body_model = _BodyBuffers()


def _artifact_bundle(tmp_path: Path) -> HyMotionV2MBundle:
    bundle = HyMotionV2MBundle.__new__(HyMotionV2MBundle)
    torch.nn.Module.__init__(bundle)
    bundle.model = _ArtifactModel()
    bundle._artifact_network_module = "hymotion/network/Test"
    bundle._network_module_args = {"input_dim": 2}
    bundle._pipeline_args = {
        "mean_std": str(tmp_path / "stats.json"),
        "body_model_path": str(tmp_path / "restricted_model.npz"),
        "test_cfg": {"mean_std_dir": str(tmp_path / "stats.json")},
    }
    bundle._mean_std_path = tmp_path / "stats.json"
    bundle._mean_std_path.write_text("{}\n")
    bundle._checkpoint_path = None
    bundle._checkpoint_sha256 = "a" * 64
    return bundle


def test_hymotion_v2m_export_embeds_inference_body_buffers(tmp_path: Path):
    bundle = _artifact_bundle(tmp_path)
    output = tmp_path / "artifact"

    bundle.save_pretrained(str(output))

    state = load_file(str(output / "model.safetensors"))
    assert "motion_transformer.weight" in state
    assert "body_model.j_template" in state
    assert not any(key.startswith("vertex_loss.body_model.") for key in state)
    config = json.loads((output / "hymotion_v2m_config.json").read_text())
    assert config["artifact_format"] == HYMOTION_V2M_ARTIFACT_FORMAT
    assert config["source_checkpoint_sha256"] == "a" * 64
    assert config["pipeline_args"]["mean_std"] == "mean_std.json"
    assert "body_model_path" not in config["pipeline_args"]
    assert config["pipeline_args"]["test_cfg"]["mean_std_dir"] == "mean_std.json"
    assert config["components"]["smplh_kinematics"]["stored_in_artifact"] is True
    assert "external_assets" not in config
    model_index = json.loads((output / "model_index.json").read_text())
    assert "model.safetensors" in model_index["required_files"]
    assert "mean_std.json" in model_index["required_files"]
    for filename in (
        "README.md",
        "License.txt",
        "NOTICE",
        "ATTRIBUTIONS.md",
        "SAM3D_LICENSE",
        "DINOV3_LICENSE.md",
        "YOLOX_LICENSE",
    ):
        assert (output / filename).is_file()
    readme = (output / "README.md").read_text()
    assert "HyMotionV2MPipeline.from_pretrained" in readme
    assert "infer_monocular_motion_capture" in readme
    assert "rtol=0" in readme


def test_hymotion_v2m_export_packages_preprocessor_assets(tmp_path: Path):
    bundle = _artifact_bundle(tmp_path)
    assets = {}
    for key in ("sam3d_ckpt", "sam3d_mhr", "sam3d_config", "yolox_ckpt"):
        path = tmp_path / key
        path.write_bytes(key.encode("ascii"))
        assets[key] = str(path)

    output = tmp_path / "artifact"
    bundle.save_pretrained(str(output), preprocessor_assets=assets)

    config = json.loads((output / "hymotion_v2m_config.json").read_text())
    for key, relative_path in config["preprocessor_assets"].items():
        assert key in assets
        assert (output / relative_path).is_file()


def test_hymotion_v2m_loads_hf_symlinked_safetensors(tmp_path: Path):
    bundle = _artifact_bundle(tmp_path)
    blob = tmp_path / "extensionless-cache-blob"
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in bundle.model.state_dict().items()
        },
        str(blob),
    )
    logical_checkpoint = tmp_path / "model.safetensors"
    logical_checkpoint.symlink_to(blob.name)

    bundle.load_v2m_checkpoint(str(logical_checkpoint), strict=True)

    assert bundle._checkpoint_path == blob.resolve()


def test_hymotion_v2m_rebuilds_body_models_from_exact_buffers():
    skeleton_buffers = {
        "j_template": torch.randn(52, 3),
        "j_shapedirs": torch.randn(52, 3, 16),
        "parents": torch.arange(52),
    }
    mesh_buffers = {
        **skeleton_buffers,
        "v_template": torch.randn(12, 3),
        "shapedirs": torch.randn(12, 3, 16),
        "posedirs": torch.randn(459, 36),
        "lbs_weights": torch.randn(12, 52),
        "J_regressor": torch.randn(52, 12),
    }

    skeleton = SMPLSkeleton.from_buffers(skeleton_buffers)
    mesh = SMPLMesh.from_buffers(mesh_buffers)

    for name, expected in skeleton_buffers.items():
        assert torch.equal(dict(skeleton.named_buffers())[name], expected)
    for name, expected in mesh_buffers.items():
        assert torch.equal(dict(mesh.named_buffers())[name], expected)
    assert mesh.faces is None
