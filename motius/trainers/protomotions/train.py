"""Launch the vendored ProtoMotions trainer without an upstream checkout."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Optional, Sequence


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c"
DEFAULT_EXPERIMENT = VENDOR_ROOT / "examples/experiments/mimic/mlp.py"
OUTPUT_ROOT = REPO_ROOT / "outputs/training/protomotions"
PATH_ARGUMENTS = {"--checkpoint", "--motion-file", "--scenes-file"}


def _absolute_input_paths(arguments: Sequence[str], cwd: Path) -> list[str]:
    argv = list(arguments)
    for index, value in enumerate(argv):
        if value in PATH_ARGUMENTS and index + 1 < len(argv):
            path = Path(argv[index + 1]).expanduser()
            if not path.is_absolute():
                argv[index + 1] = str((cwd / path).resolve())
        else:
            for key in PATH_ARGUMENTS:
                prefix = f"{key}="
                if value.startswith(prefix):
                    path = Path(value[len(prefix):]).expanduser()
                    if not path.is_absolute():
                        argv[index] = prefix + str((cwd / path).resolve())
    return argv


def _training_argv(arguments: Sequence[str]) -> list[str]:
    argv = _absolute_input_paths(arguments, Path.cwd())
    if "--experiment-path" not in argv and not any(
        value.startswith("--experiment-path=") for value in argv
    ):
        argv.extend(["--experiment-path", str(DEFAULT_EXPERIMENT)])
    return argv


class ProtoMotionsTrainer:
    """Adapter for ProtoMotions' native PPO and simulator training loop."""

    upstream_commit = UPSTREAM_COMMIT

    @classmethod
    def launch(cls, arguments: Optional[Sequence[str]] = None) -> None:
        main(arguments)

    @classmethod
    def export_policy(
        cls,
        checkpoint: str | Path,
        output_dir: str | Path | None = None,
        *,
        validate: bool = True,
    ) -> Path:
        """Export a native checkpoint as the deployable Motius ONNX artifact."""
        vendor = str(VENDOR_ROOT)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from deployment.export_bm_tracker_onnx import export_tracker

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        export_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else checkpoint_path.parent / "compiled_models"
        )
        return export_tracker(
            checkpoint=str(checkpoint_path),
            output_dir=str(export_root),
            validate=validate,
        )


def main(arguments: Optional[Sequence[str]] = None) -> None:
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    argv = _training_argv(sys.argv[1:] if arguments is None else arguments)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    previous_output_root = os.environ.get("MOTIUS_PROTOMOTIONS_OUTPUT_ROOT")
    try:
        os.environ["MOTIUS_PROTOMOTIONS_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
        os.chdir(VENDOR_ROOT)
        sys.argv = ["protomotions.train_agent", *argv]
        runpy.run_module("protomotions.train_agent", run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        if previous_output_root is None:
            os.environ.pop("MOTIUS_PROTOMOTIONS_OUTPUT_ROOT", None)
        else:
            os.environ["MOTIUS_PROTOMOTIONS_OUTPUT_ROOT"] = previous_output_root


if __name__ == "__main__":
    main()
