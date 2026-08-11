"""Launch the vendored SONIC PPO trainer without an upstream checkout."""

from __future__ import annotations

import os
import shutil
import runpy
import sys
from pathlib import Path
from typing import Optional, Sequence


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "outputs/training/sonic"
UPSTREAM_COMMIT = "4141c34280abb67c82e115342a8720f4a83d750d"
DEFAULT_EXPERIMENT = "manager/universal_token/all_modes/sonic_release"
EVAL_SCRIPT = VENDOR_ROOT / "gear_sonic/eval_agent_trl.py"


def _training_argv(arguments: Sequence[str]) -> list[str]:
    argv = list(arguments)
    if not any(value.startswith("+exp=") for value in argv):
        argv.insert(0, f"+exp={DEFAULT_EXPERIMENT}")
    if not any(value.startswith("base_dir=") for value in argv):
        argv.append(f"base_dir={OUTPUT_ROOT}")
    return argv


class SonicTrainer:
    """Adapter for SONIC's native Hydra/TRL/Isaac Lab training loop.

    SONIC owns environment stepping, PPO optimization, checkpointing, and
    distributed synchronization. The adapter deliberately preserves that
    loop and only supplies Motius-local source discovery and output defaults.
    """

    upstream_commit = UPSTREAM_COMMIT

    @classmethod
    def launch(cls, arguments: Optional[Sequence[str]] = None) -> None:
        main(arguments)

    @classmethod
    def export_policy(
        cls,
        checkpoint: str | Path,
        output_dir: str | Path,
        *,
        arguments: Optional[Sequence[str]] = None,
    ) -> Path:
        """Export and package one native checkpoint for Pipeline loading."""
        return export_policy(
            checkpoint,
            output_dir,
            arguments=arguments,
        )


def _exported_onnx_pair(checkpoint_dir: Path) -> tuple[Path, Path]:
    export_root = checkpoint_dir / "exported"
    encoders = sorted(
        export_root.glob("model_step_*_encoder.onnx"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not encoders:
        raise FileNotFoundError(
            f"SONIC export did not produce an encoder below {export_root}."
        )
    encoder = encoders[-1]
    prefix = encoder.name.removesuffix("_encoder.onnx")
    decoder = export_root / f"{prefix}_decoder.onnx"
    if not decoder.is_file():
        raise FileNotFoundError(
            f"SONIC export produced {encoder.name} without {decoder.name}."
        )
    return encoder, decoder


def export_policy(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    arguments: Optional[Sequence[str]] = None,
) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SONIC checkpoint does not exist: {checkpoint_path}")
    artifact_root = Path(output_dir).expanduser().resolve()
    argv = [
        f"+checkpoint={checkpoint_path}",
        "+num_envs=1",
        "+headless=true",
        "+export_onnx_only=true",
        *(arguments or ()),
    ]
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    try:
        os.chdir(VENDOR_ROOT)
        sys.argv = [str(EVAL_SCRIPT), *argv]
        try:
            runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv

    encoder, decoder = _exported_onnx_pair(checkpoint_path.parent)
    observation_config = checkpoint_path.parent / "model_config.yaml"
    if not observation_config.is_file():
        raise FileNotFoundError(
            "SONIC export did not produce model_config.yaml beside the checkpoint."
        )

    from motius.models.sonic import SONICBundle

    source = SONICBundle(
        file_paths={
            "encoder": encoder,
            "decoder": decoder,
            "observation_config": observation_config,
        }
    )
    source.save_pretrained(artifact_root)
    train_config = checkpoint_path.parent / "config.yaml"
    if train_config.is_file():
        shutil.copy2(train_config, artifact_root / "train_config.yaml")
    return artifact_root


def main(arguments: Optional[Sequence[str]] = None) -> None:
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    argv = _training_argv(sys.argv[1:] if arguments is None else arguments)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    try:
        os.chdir(VENDOR_ROOT)
        sys.argv = ["gear_sonic.train_agent_trl", *argv]
        runpy.run_module("gear_sonic.train_agent_trl", run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
