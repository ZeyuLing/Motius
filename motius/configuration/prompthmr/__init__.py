"""Pinned source provenance and released PromptHMR checkpoint inventory."""
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

PROMPTHMR_REPOSITORY = "https://github.com/yufu-wang/PromptHMR.git"


PROMPTHMR_REVISION = "3b566b7dbb28ce506c7ea972c18693f4c705ce8c"


PROMPTHMR_RUNTIME_CHECKPOINT = Path(
    "data/pretrain/phmr_vid/prhmr_release_002.ckpt"
)


PROMPTHMR_IMAGE_CHECKPOINT = Path("data/pretrain/phmr/checkpoint.ckpt")


@dataclass(frozen=True)
class PromptHMRCheckpoint:
    """One official BEDLAM video-head checkpoint."""

    name: str
    filename: str
    sha256: str
    url: str
    training_data: str


_BEDLAM_BASE = "https://download.is.tue.mpg.de/bedlam2/ml/videos"


PROMPTHMR_VIDEO_CHECKPOINTS: Dict[str, PromptHMRCheckpoint] = {
    "bedlam1": PromptHMRCheckpoint(
        name="bedlam1",
        filename="phmr_b1.ckpt",
        sha256="d06ae5ddc74ef74c252f4ec34e4e3092cd8fc18cba104af5aa978cdd2c669b5a",
        url=f"{_BEDLAM_BASE}/phmr_b1.ckpt",
        training_data="BEDLAM1",
    ),
    "bedlam1+2": PromptHMRCheckpoint(
        name="bedlam1+2",
        filename="phmr_b1b2.ckpt",
        sha256="2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5",
        url=f"{_BEDLAM_BASE}/phmr_b1b2.ckpt",
        training_data="BEDLAM1+BEDLAM2",
    ),
    "bedlam2": PromptHMRCheckpoint(
        name="bedlam2",
        filename="phmr_b2.ckpt",
        sha256="631433bf4dfd548dc5c6e2df037e11a11ce4a83c37367ee0f31b2f1627aa06d9",
        url=f"{_BEDLAM_BASE}/phmr_b2.ckpt",
        training_data="BEDLAM2",
    ),
}


_CHECKPOINT_ALIASES = {
    "b1": "bedlam1",
    "b1b2": "bedlam1+2",
    "b2": "bedlam2",
    "phmr_b1.ckpt": "bedlam1",
    "phmr_b1b2.ckpt": "bedlam1+2",
    "phmr_b2.ckpt": "bedlam2",
}


def sha256_file(path: Path) -> str:
    """Hash a local artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_name(value: str) -> str:
    normalized = _CHECKPOINT_ALIASES.get(value.lower(), value.lower())
    if normalized not in PROMPTHMR_VIDEO_CHECKPOINTS:
        options = ", ".join(PROMPTHMR_VIDEO_CHECKPOINTS)
        raise ValueError(
            f"Unknown PromptHMR video checkpoint {value!r}; choose one of {options}."
        )
    return normalized
