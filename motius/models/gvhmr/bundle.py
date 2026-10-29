"""GVHMR neural architecture and released checkpoint I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .network import NetworkEncoderRoPE

OFFICIAL_REPOSITORY = "https://github.com/zju3dv/GVHMR"

OFFICIAL_RUNTIME_REVISION = "6ec3ca39336c50492c0fae65fba2fb831fc7d866"

GVHMR_REPO_ID = "ZeyuLing/Motius-GVHMR"

GVHMR_ARTIFACT_FORMAT = "motius-gvhmr-v1"


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a local artifact without loading or deserializing it."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    candidate = Path(pretrained_model_name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            token=token,
            local_files_only=local_files_only,
        )
    ).resolve()


@MODEL_BUNDLES.register_module()
class GVHMRBundle(ModelBundle):
    """Share only the motion network between the native trainer and pipeline."""
    def __init__(self, artifact_root=None, *, checkpoint_path=None, device=None,
                 config=None, body_models_root=None, manifest=None):
        super().__init__()
        self.artifact_root=Path(artifact_root or ".").expanduser().resolve()
        metadata_path=self.artifact_root/"gvhmr_config.json"
        self.manifest=dict(manifest or (json.loads(metadata_path.read_text()) if metadata_path.exists() else {}))
        revision=self.manifest.get("source_revision",OFFICIAL_RUNTIME_REVISION)
        if revision != OFFICIAL_RUNTIME_REVISION:
            raise ValueError("GVHMR artifact source revision does not match the supported architecture")
        self.config=dict(config or self.manifest.get("network",{}))
        self.model=NetworkEncoderRoPE(**self.config)
        if checkpoint_path is None:
            candidates=[self.artifact_root/"gvhmr.safetensors",
                        self.artifact_root/"inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
                        self.artifact_root/"gvhmr/gvhmr_siga24_release.ckpt"]
            checkpoint_path=next((p for p in candidates if p.is_file()),None)
        self.checkpoint_path=Path(checkpoint_path).resolve() if checkpoint_path is not None else None
        if self.checkpoint_path is not None:
            if self.checkpoint_path.suffix==".safetensors":
                from safetensors.torch import load_file
                state=load_file(str(self.checkpoint_path))
            else:
                state=torch.load(self.checkpoint_path,map_location="cpu",weights_only=True)
                state=state.get("state_dict",state)
                prefix="pipeline.denoiser3d."
                if any(k.startswith(prefix) for k in state):
                    other=[k for k in state if not k.startswith(prefix)]
                    if other:raise ValueError(f"Unexpected GVHMR checkpoint components: {other[:8]}")
                    state={k.removeprefix(prefix):v for k,v in state.items()}
            self.model.load_state_dict(state,strict=True)
        elif config is None:
            raise FileNotFoundError("GVHMR released network checkpoint was not found")
        # Asset locations are metadata; body geometry belongs to the pipeline.
        self.body_models_root=Path(body_models_root or self.artifact_root/"inputs/checkpoints/body_models").expanduser().resolve()
        self._trainable_modules=["model"]
        self._save_ckpt_modules=["model"]
        self.model.eval()
        if device is not None:self.to(device)

    @property
    def network(self):
        model=self.model
        while hasattr(model,"module"):model=model.module
        return model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def checkpoint_sha256(self):
        if self.checkpoint_path is None:raise ValueError("No pretrained checkpoint is attached")
        actual=sha256_file(self.checkpoint_path)
        expected=self.manifest.get("sha256",{}).get("inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt")
        if expected and self.checkpoint_path.suffix!=".safetensors" and expected!=actual:
            raise ValueError("GVHMR checkpoint SHA256 does not match its artifact manifest")
        return actual

    def predict_motion(self,*args,**kwargs):
        return self.model(*args,**kwargs)

    def forward(self,*args,**kwargs):
        return self.predict_motion(*args,**kwargs)

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,**kwargs):
        download={k:kwargs.pop(k) for k in ["revision","cache_dir","token","local_files_only"] if k in kwargs}
        return cls(artifact_root=_resolve_artifact(pretrained_model_name_or_path,**download),**kwargs)

    def save_pretrained(self,save_directory,**kwargs):
        from safetensors.torch import save_file
        destination=Path(save_directory);destination.mkdir(parents=True,exist_ok=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in self.network.state_dict().items()},str(destination/"gvhmr.safetensors"))
        (destination/"gvhmr_config.json").write_text(json.dumps({"artifact_format":GVHMR_ARTIFACT_FORMAT,
            "source_revision":OFFICIAL_RUNTIME_REVISION,"network":self.config},indent=2)+"\n")
        (destination/"model_index.json").write_text(json.dumps({"model_type":"gvhmr","_class_name":"GVHMRPipeline",
            "_library_name":"motius","bundle_class":"motius.models.gvhmr.GVHMRBundle",
            "pipeline_class":"motius.pipelines.gvhmr.GVHMRPipeline"},indent=2)+"\n")
        return save_directory
