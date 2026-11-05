"""FlowHMR trainable networks and checkpoint I/O; no training or sampling graph."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

SOURCE_REVISION = "f12e6a2d46a63d771a66dbb6b5598a1be65b9eea"
ASSETS = Path(__file__).resolve().parents[2] / "configuration" / "flowhmr"
LICENSES = Path(__file__).resolve().parents[2] / "licenses" / "flowhmr"
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

    def __init__(self, *, body_model_path=None, j_regressor_path=None,
                 config_path=None, config=None, mean_std_path=None,
                 ckpt_path=None, device=None, _init_device=None):
        super().__init__()
        from .network.hymotion_mmdit_for_v2m import HunyuanMotionMMDiT
        from motius.utils.flowhmr_checkpoint import load_motion_statistics
        if config is not None and config_path is not None:
            raise ValueError("Provide config or config_path, not both")
        self.config = deepcopy(config) if config is not None else yaml.safe_load(
            Path(config_path or ASSETS / "base_model.yml").read_text())
        network = deepcopy(self.config["network_module_args"])
        if network.get("input_dim") != 491:
            raise ValueError("FlowHMR requires the official 491-D model; legacy 349-D V2M is incompatible")
        self.mean_std_path = Path(mean_std_path or ASSETS / "motion_stats.json")
        self.body_model_path = str(body_model_path) if body_model_path is not None else None
        self.j_regressor_path = str(j_regressor_path) if j_regressor_path is not None else None
        statistics = load_motion_statistics(self.mean_std_path)
        self._statistic_names = tuple(statistics)
        for name, value in statistics.items():
            self.register_buffer("motion_stat__" + name, value)
        # A fully loaded checkpoint overwrites every parameter. Avoid allocating
        # and randomly initializing 460M disposable CPU values on inference load.
        with torch.device(_init_device) if _init_device else nullcontext():
            self.model = HunyuanMotionMMDiT(**network)
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self._module_checkpoint_formats = {"model": "full"}
        self.checkpoint_sha256 = None
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        if device is not None:
            self.to(device)

    @property
    def network(self):
        """Unwrap for metadata/export; forward always enters the wrapped module."""
        from torch.nn.parallel import DistributedDataParallel
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def predict_motion(self, **inputs):
        return self.model(**inputs)

    @property
    def motion_statistics(self):
        return {name: getattr(self, "motion_stat__" + name) for name in self._statistic_names}

    def export_state_dict(self):
        """Official checkpoint names, independent of the native module layout."""
        return {**{"motion_transformer." + key: value for key, value in self.network.state_dict().items()},
                **self.motion_statistics}

    def load_checkpoint(self, path):
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("model_state_dict", state.get("state_dict", state))
        network_state = {}
        statistics = {}
        for key, value in state.items():
            if key.startswith("motion_transformer."):
                network_state[key.removeprefix("motion_transformer.")] = value
            elif key in self.motion_statistics:
                if value.shape != self.motion_statistics[key].shape:
                    raise ValueError(f"Incompatible FlowHMR statistic: {key}")
                statistics[key] = value
            elif not _external_body_key(key):
                raise RuntimeError(f"Unexpected FlowHMR checkpoint key: {key}")
        missing = self.motion_statistics.keys() - statistics.keys()
        if missing:
            raise RuntimeError(f"Missing FlowHMR statistics: {sorted(missing)}")
        assign = any(p.is_meta for p in self.network.parameters())
        if assign:
            targets = self.network.state_dict()
            # Match normal load_state_dict's dtype conversion instead of
            # silently changing inference precision to the checkpoint storage.
            network_state = {key: value.to(dtype=targets[key].dtype) if key in targets else value
                             for key, value in network_state.items()}
        self.network.load_state_dict(network_state, strict=True, assign=assign)
        if assign:
            # RoPE tables are deterministic, nonpersistent buffers, so they are
            # absent from checkpoints and must be rebuilt after a meta init.
            from .network.mmdit.positional_encoding import RotaryEmbedding
            for module in self.network.modules():
                if type(module) is RotaryEmbedding and any(b.is_meta for b in module.buffers()):
                    with torch.device('cpu'):
                        tables = RotaryEmbedding(module.num_feats, module.max_seq_len,
                            module.temperature, module.use_real, module.theta_rescale_factor,
                            module.interpolation_factor)
                    for name, value in tables._buffers.items():
                        setattr(module, name, value)
        if any(t.is_meta for t in (*self.network.parameters(), *self.network.buffers())):
            raise RuntimeError('FlowHMR checkpoint left unmaterialized tensors')
        for name, value in statistics.items():
            setattr(self, "motion_stat__" + name, value)
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, ckpt_name=None,
                        revision=None, cache_dir=None, token=None,
                        local_files_only=False, low_cpu_mem_usage=True, **kwargs):
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
        if low_cpu_mem_usage:
            kwargs['_init_device'] = 'meta'
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
                   for k, v in self.export_state_dict().items() if not _external_body_key(k)},
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
        for name, source in {"LICENSE": LICENSES / "UPSTREAM_LICENSE",
                             "ATTRIBUTIONS.md": LICENSES / "ATTRIBUTIONS.md"}.items():
            shutil.copyfile(source, root / name)
        for source in LICENSES.glob("*LICENSE*"):
            shutil.copyfile(source, root / source.name)
