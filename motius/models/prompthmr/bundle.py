"""Metadata-only bundle for the isolated official PromptHMR-Video runtime."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


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


@MODEL_BUNDLES.register_module()
class PromptHMRBundle(ModelBundle):
    """Own paths and provenance for an out-of-process PromptHMR runtime.

    PromptHMR's released dependency stack is intentionally not imported into
    Motius. The official code runs in its own conda environment; this bundle
    verifies the pinned source revision and all checkpoints before inference.
    """

    SUPPORTED_TASKS = {
        "monocular_video_capture": (
            "multi-person PromptHMR-Video reconstruction from an RGB video"
        ),
        "official_output_conversion": (
            "convert the official results.pkl without loading model weights"
        ),
    }

    def __init__(
        self,
        upstream_dir: Optional[str] = None,
        video_checkpoint: str = "bedlam1+2",
        video_checkpoint_path: Optional[str] = None,
        image_checkpoint_path: Optional[str] = None,
        python_command: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.upstream_dir = (
            None if upstream_dir is None else str(Path(upstream_dir).expanduser())
        )
        self.video_checkpoint = _checkpoint_name(video_checkpoint)
        self.video_checkpoint_path = video_checkpoint_path
        self.image_checkpoint_path = image_checkpoint_path
        self.python_command: Tuple[str, ...] = tuple(python_command or ("python",))
        self._verified_checkpoint_cache: Optional[
            tuple[tuple[tuple[str, int, int], ...], Dict[str, str]]
        ] = None
        if not self.python_command:
            raise ValueError("python_command must contain at least one argument.")

    @property
    def checkpoint_spec(self) -> PromptHMRCheckpoint:
        return PROMPTHMR_VIDEO_CHECKPOINTS[self.video_checkpoint]

    @property
    def expected_checkpoint_sha256(self) -> str:
        return self.checkpoint_spec.sha256

    def resolved_video_checkpoint_path(self) -> Optional[Path]:
        if self.video_checkpoint_path:
            return Path(self.video_checkpoint_path).expanduser().resolve()
        if self.upstream_dir:
            return (
                Path(self.upstream_dir).expanduser().resolve()
                / "data"
                / "pretrain"
                / "phmr_vid"
                / self.checkpoint_spec.filename
            )
        return None

    def resolved_image_checkpoint_path(self) -> Optional[Path]:
        if self.image_checkpoint_path:
            return Path(self.image_checkpoint_path).expanduser().resolve()
        if self.upstream_dir:
            return (
                Path(self.upstream_dir).expanduser().resolve()
                / PROMPTHMR_IMAGE_CHECKPOINT
            )
        return None

    def verify_upstream_revision(self) -> Path:
        """Require the official checkout to be exactly the pinned commit."""

        if self.upstream_dir is None:
            raise FileNotFoundError(
                "upstream_dir is required to run PromptHMR inference."
            )
        upstream = Path(self.upstream_dir).expanduser().resolve()
        if not (upstream / "scripts" / "demo_video.py").is_file():
            raise FileNotFoundError(
                f"PromptHMR scripts/demo_video.py was not found under {upstream}."
            )

        marker = upstream / ".motius_prompthmr_revision"
        if (upstream / ".git").exists():
            try:
                revision = subprocess.run(
                    ["git", "-C", str(upstream), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                tracked_status = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(upstream),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.rstrip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError(
                    "Could not establish the PromptHMR source revision. Run "
                    "tools/setup_prompthmr_env.sh or provide a pinned git checkout."
                ) from exc
            modified_paths = {
                line[3:] for line in tracked_status.splitlines() if len(line) > 3
            }
            audited_runtime_patch = (
                os.environ.get("MOTIUS_PROMPTHMR_AUDITED_PATCH") == "1"
                and modified_paths
                <= {
                    "pipeline/utils_detectron2.py",
                    "pipeline/tools.py",
                    "pipeline/detector/segment.py",
                }
            )
            if tracked_status and not audited_runtime_patch:
                raise RuntimeError(
                    "PromptHMR checkout has modified tracked files; inference "
                    "requires the pristine pinned revision."
                )
        elif marker.is_file():
            revision = marker.read_text().strip()
        else:
            raise RuntimeError(
                "Could not establish the PromptHMR source revision. Run "
                "tools/setup_prompthmr_env.sh or provide a pinned git checkout."
            )
        if revision != PROMPTHMR_REVISION:
            raise RuntimeError(
                f"PromptHMR revision mismatch: expected {PROMPTHMR_REVISION}, "
                f"found {revision or '<empty>'}."
            )
        return upstream

    def verify_checkpoints(self) -> Dict[str, str]:
        """Hash required image and video weights and reject any mismatch."""

        video_path = self.resolved_video_checkpoint_path()
        image_path = self.resolved_image_checkpoint_path()
        if video_path is None or not video_path.is_file():
            raise FileNotFoundError(
                "PromptHMR video checkpoint is missing. Expected "
                f"{video_path or self.checkpoint_spec.filename}."
            )
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(
                "PromptHMR image checkpoint is missing. Expected "
                f"{image_path or PROMPTHMR_IMAGE_CHECKPOINT}."
            )

        signature = tuple(
            (
                str(path),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in (video_path, image_path)
        )
        if (
            self._verified_checkpoint_cache is not None
            and self._verified_checkpoint_cache[0] == signature
        ):
            return dict(self._verified_checkpoint_cache[1])

        video_sha256 = sha256_file(video_path)
        if video_sha256 != self.expected_checkpoint_sha256:
            raise ValueError(
                f"SHA256 mismatch for {video_path}: expected "
                f"{self.expected_checkpoint_sha256}, found {video_sha256}."
            )
        hashes = {
            "video_head": video_sha256,
            "image_model": sha256_file(image_path),
        }
        self._verified_checkpoint_cache = (signature, hashes)
        return dict(hashes)

    def stage_official_runtime_checkpoint(self) -> Path:
        """Expose the selected BEDLAM checkpoint at the official hard-coded path."""

        upstream = self.verify_upstream_revision()
        hashes = self.verify_checkpoints()
        source = self.resolved_video_checkpoint_path()
        assert source is not None
        destination = upstream / PROMPTHMR_RUNTIME_CHECKPOINT
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            try:
                if destination.samefile(source):
                    return destination
            except OSError:
                pass
            if destination.is_file() and sha256_file(destination) == hashes["video_head"]:
                return destination
            raise FileExistsError(
                f"{destination} already exists with different contents; refusing "
                "to overwrite the official runtime checkpoint."
            )
        try:
            destination.symlink_to(source)
        except FileExistsError:
            # Multiple 3DPW GPU shards stage the same immutable checkpoint at
            # startup. Another shard may win after our existence check.
            if destination.is_file() and sha256_file(destination) == hashes["video_head"]:
                return destination
            raise
        return destination

    @classmethod
    def _bundle_config_from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> dict:
        config = dict(kwargs)
        config.setdefault(
            "video_checkpoint_path", str(pretrained_model_name_or_path)
        )
        return config

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "PromptHMR runs out of process. Use PromptHMRPipeline instead."
        )


__all__ = [
    "PROMPTHMR_IMAGE_CHECKPOINT",
    "PROMPTHMR_REPOSITORY",
    "PROMPTHMR_REVISION",
    "PROMPTHMR_RUNTIME_CHECKPOINT",
    "PROMPTHMR_VIDEO_CHECKPOINTS",
    "PromptHMRBundle",
    "PromptHMRCheckpoint",
    "sha256_file",
]
