"""Helpers for constructing OpenAI CLIP from artifact-local weights."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


def load_openai_clip(
    weights_path: str | Path,
    *,
    device: str | torch.device = "cpu",
):
    """Build CLIP without invoking ``clip.load`` or its download cache."""

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bundled CLIP weights are missing: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = dict(load_file(str(path), device="cpu"))
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    import clip

    model = clip.model.build_model(state).to(device)
    if str(device) == "cpu":
        model.float()
    return model.eval()


@contextmanager
def artifact_clip_load(
    weights_path: str | Path,
    *,
    expected_name: str | None = None,
) -> Iterator[None]:
    """Temporarily make ``clip.load`` resolve only artifact-local weights."""

    import clip

    original = clip.load

    def local_load(name, device="cpu", jit=False, download_root=None):
        del download_root
        if jit:
            raise ValueError("Artifact-local CLIP loading does not support JIT")
        if expected_name is not None and str(name) != expected_name:
            raise ValueError(
                f"Artifact contains CLIP {expected_name!r}, requested {name!r}"
            )
        return load_openai_clip(weights_path, device=device), None

    clip.load = local_load
    try:
        yield
    finally:
        clip.load = original


__all__ = ["artifact_clip_load", "load_openai_clip"]
