"""Motius bundle for the subprocess-isolated official GEM-X runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from motius.models.base_model_bundle import ModelBundle
from motius.models.gem_x import runtime
from motius.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class GemXBundle(ModelBundle):
    """Own GEM-X/SOMA-77 runtime provenance without changing body topology."""

    def __init__(
        self,
        runtime_root: str,
        checkpoint: str,
        python_executable: Optional[str] = None,
        export_device: str = "cuda",
        soma_assets: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.runtime_root = str(Path(runtime_root).expanduser().resolve())
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.python_executable = python_executable
        self.export_device = str(export_device)
        self.soma_assets = str(
            (
                Path(soma_assets).expanduser()
                if soma_assets is not None
                else Path(self.runtime_root) / "inputs" / "soma_assets"
            ).resolve()
        )

    @property
    def checkpoint_sha256(self) -> str:
        return runtime.CHECKPOINT_SHA256

    @property
    def source_revision(self) -> str:
        return runtime.SOURCE_REVISION

    def verify(self) -> dict[str, str]:
        runtime.verify_runtime_checkout(self.runtime_root)
        runtime.verify_checkpoint(self.checkpoint)
        return {
            "source_revision": self.source_revision,
            "soma_source_revision": runtime.SOMA_SOURCE_REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> dict:
        config = dict(kwargs)
        config["checkpoint"] = pretrained_model_name_or_path
        return config

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "GEM-X runs in its isolated official environment. Use GemXPipeline."
        )


__all__ = ["GemXBundle"]
