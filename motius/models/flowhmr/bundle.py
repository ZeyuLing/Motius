"""FlowHMR's official network, representation and losses in a Motius bundle."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

SOURCE_REVISION = "f12e6a2d46a63d771a66dbb6b5598a1be65b9eea"
ASSETS = Path(__file__).parent / "assets"
_BODY_PREFIXES = ("body_model.", "mesh_model.", "vertex_loss.body_model.")


def _external_body_key(key):
    return key == "J_regressor" or key.startswith(_BODY_PREFIXES)


@MODEL_BUNDLES.register_module()
class FlowHMRBundle(ModelBundle):
    """491-D official base/latest checkpoint; deliberately separate from legacy V2M.

    Body assets are explicit and excluded from exported inference weights.
    Runner checkpoints include the model, its statistics and optimizer state via
    Motius's normal checkpoint hook. ``save_pretrained`` exports inference weights.
    """

    def __init__(self, *, body_model_path, j_regressor_path,
                 config_path=None, config=None, mean_std_path=None,
                 ckpt_path=None, device=None):
        super().__init__()
        from .vendor.flowhmr.pipeline.pipeline_v2m import V2MPipeline

        if config is not None and config_path is not None:
            raise ValueError("Provide config or config_path, not both")
        self.config = deepcopy(config) if config is not None else yaml.safe_load(
            Path(config_path or ASSETS / "base_model.yml").read_text())
        network = self.config["network_module_args"]
        if network.get("input_dim") != 491:
            raise ValueError("FlowHMR requires the official 491-D model; legacy 349-D V2M is incompatible")
        self.mean_std_path = Path(mean_std_path or ASSETS / "motion_stats.json")
        args = deepcopy(self.config["train_pipeline_args"])
        args.update(mean_std=str(self.mean_std_path), smpl_model_path=str(body_model_path),
                    j_regressor_path=str(j_regressor_path))
        self.model = V2MPipeline(network_module=self.config["network_module"],
                                 network_module_args=deepcopy(network), **args)
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self._module_checkpoint_formats = {"model": "full"}
        self.checkpoint_sha256 = None
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        if device is not None:
            self.to(device)

    def load_checkpoint(self, path):
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("model_state_dict", state.get("state_dict", state))
        # Official releases intentionally omit licensed body buffers. Only those
        # omissions are allowed; every network/stat key is still checked strictly.
        current = self.inference_model.state_dict()
        for key, value in current.items():
            if _external_body_key(key) and key not in state:
                state[key] = value
        self.inference_model.load_state_dict(state, strict=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
            if digest is None:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, ckpt_name=None,
                        revision=None, cache_dir=None, token=None,
                        local_files_only=False, **kwargs):
        root = Path(pretrained_model_name_or_path).expanduser()
        if not root.exists():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(pretrained_model_name_or_path, revision=revision,
                        cache_dir=cache_dir, token=token, local_files_only=local_files_only))
        if root.is_file():
            checkpoint, root = root, root.parent
        elif ckpt_name:
            checkpoint = root / ckpt_name
        else:
            candidates = [p for p in (root / "model.safetensors", root / "flowhmr_base.ckpt",
                          root / "flowhmr_latest.ckpt") if p.is_file()]
            if len(candidates) != 1:
                raise ValueError("Select one checkpoint directory or supply ckpt_name explicitly")
            checkpoint = candidates[0]
        kwargs.setdefault("config_path", str(root / "config.yml"))
        if (root / "motion_stats.json").is_file():
            kwargs.setdefault("mean_std_path", str(root / "motion_stats.json"))
        bundle = cls(ckpt_path=checkpoint, **kwargs)
        bundle.artifact_root = root
        manifest = root / "model_index.json"
        if manifest.is_file():
            bundle.artifact_manifest = json.loads(manifest.read_text())
        return bundle

    def save_pretrained(self, save_directory, *, source_assets=None, **kwargs):
        from safetensors.torch import save_file
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        # Clone shared buffers so safetensors has no aliased storages.
        save_file({k: v.detach().cpu().contiguous().clone()
                   for k, v in self.inference_model.state_dict().items() if not _external_body_key(k)},
                  str(root / "model.safetensors"))
        cfg = deepcopy(self.config)
        for key in ("smpl_model_path", "j_regressor_path"):
            cfg["train_pipeline_args"].pop(key, None)
        cfg["train_pipeline_args"]["mean_std"] = "motion_stats.json"
        (root / "config.yml").write_text(yaml.safe_dump(cfg))
        if self.mean_std_path.resolve() != (root / "motion_stats.json").resolve():
            shutil.copyfile(self.mean_std_path, root / "motion_stats.json")
        asset_hashes = {}
        for relative, source in (source_assets or {}).items():
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Asset path must be relative: {relative}")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(source).resolve() != target.resolve():
                shutil.copyfile(source, target)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            asset_hashes[relative.as_posix()] = digest.hexdigest()
        (root / "model_index.json").write_text(json.dumps({
            "_class_name": "FlowHMRPipeline",
            "_library_name": "motius",
            "artifact_format": "motius-flowhmr-v1",
            "pipeline_class": "motius.pipelines.flowhmr.FlowHMRPipeline",
            "bundle_class": "motius.models.flowhmr.FlowHMRBundle",
            "source_revision": SOURCE_REVISION,
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "tasks": ["monocular_motion_capture"],
            "required_files": ["model.safetensors", "config.yml", "motion_stats.json", *asset_hashes],
            "asset_sha256": asset_hashes,
            "external_assets": {"body_model_path": "Licensed AMASS SMPL-H neutral, 16 betas",
                                "j_regressor_path": "SMPL neutral joint regressor"},
        }, indent=2))
        for name, source in {"LICENSE": Path(__file__).parent / "vendor/UPSTREAM_LICENSE",
                             "ATTRIBUTIONS.md": Path(__file__).parent / "ATTRIBUTIONS.md"}.items():
            shutil.copyfile(source, root / name)
        for source in (Path(__file__).parent / "vendor").glob("*LICENSE*"):
            shutil.copyfile(source, root / source.name)

    def training_forward(self, batch, global_step=0):
        # Upstream increments on entry. Derive its warmup counter from the runner
        # so full-state resume does not restart FK/translation loss schedules.
        return self.model(batch, global_step=global_step)

    @property
    def inference_model(self):
        """Unwrap ordinary DDP for inference/export, never for the training call."""
        from torch.nn.parallel import DistributedDataParallel
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    @torch.no_grad()
    def generate_from_feature(self, feature, camera_R, *, seeds=(0,), cfg_scale=1.0):
        if feature.ndim != 3 or feature.shape[0] != 1 or feature.shape[1] == 0:
            raise ValueError("Expected one nonempty feature sequence shaped (1,T,D)")
        expected = self.config["network_module_args"]["ctxt_input_dim"]["feature"]
        if feature.shape[-1] != expected or camera_R.shape != (*feature.shape[:2], 9):
            raise ValueError("Feature or camera_R dimensions do not match the model")
        if not seeds or cfg_scale < 1:
            raise ValueError("Provide at least one seed and cfg_scale >= 1")
        device = next(self.model.parameters()).device
        conditioning = {"feature": feature.to(device, torch.float32),
                        "camera_R": camera_R.to(device, torch.float32)}
        return self.inference_model.generate(conditioning, seeds=list(seeds),
                                   length=feature.shape[1], cfg_scale=cfg_scale)
