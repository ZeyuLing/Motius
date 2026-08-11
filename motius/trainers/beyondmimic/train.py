"""Launch the vendored BeyondMimic Isaac Lab/RSL-RL trainer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import runpy
import sys
from typing import Optional, Sequence


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
TRAIN_SCRIPT = VENDOR_ROOT / "scripts/rsl_rl/train.py"
UPSTREAM_COMMIT = "cd65172032893724b445448818c34165846d847d"
DEFAULT_TASK = "Tracking-Flat-G1-v0"
DEFAULT_OUTPUT_ROOT = Path("outputs/training/beyondmimic")
DEFAULT_ASSET_ROOT = Path("checkpoints/robots")
DEFAULT_ISAAC_ASSET_DIR = Path("isaac_assets/g1")


@dataclass(frozen=True)
class TrainingLaunch:
    argv: tuple[str, ...]
    output_root: Path
    asset_root: Path
    isaac_asset_dir: Path
    resumed_from: Optional[Path]


def _has_option(arguments: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in arguments)


def _pop_value(arguments: list[str], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            arguments.pop(index)
            return value.split("=", 1)[1]
        if value == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"{name} requires a value")
            arguments.pop(index)
            return arguments.pop(index)
    return default


def _pop_flag(arguments: list[str], name: str) -> bool:
    enabled = False
    while name in arguments:
        arguments.remove(name)
        enabled = True
    return enabled


def _absolute_option(arguments: list[str], name: str, cwd: Path) -> None:
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            path = Path(value.split("=", 1)[1]).expanduser()
            if not path.is_absolute():
                path = cwd / path
            arguments[index] = f"{name}={path.resolve()}"
        elif value == name and index + 1 < len(arguments):
            path = Path(arguments[index + 1]).expanduser()
            if not path.is_absolute():
                path = cwd / path
            arguments[index + 1] = str(path.resolve())


def _option_value(arguments: Sequence[str], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def _latest_checkpoint(output_root: Path, experiment_name: str) -> Optional[Path]:
    run_root = output_root / "logs/rsl_rl" / experiment_name
    candidates = [
        path
        for path in run_root.glob("*/model_*.pt")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def build_launch(
    arguments: Sequence[str],
    *,
    cwd: Optional[Path] = None,
) -> TrainingLaunch:
    launch_cwd = (cwd or Path.cwd()).resolve()
    argv = list(arguments)
    output_value = _pop_value(argv, "--output-root", str(DEFAULT_OUTPUT_ROOT))
    asset_default = os.environ.get(
        "MOTIUS_BEYONDMIMIC_ASSET_ROOT",
        str(DEFAULT_ASSET_ROOT),
    )
    asset_value = _pop_value(argv, "--asset-root", asset_default)
    isaac_asset_value = _pop_value(
        argv,
        "--isaac-asset-dir",
        str(DEFAULT_ISAAC_ASSET_DIR),
    )
    auto_resume = _pop_flag(argv, "--auto-resume")

    output_root = Path(output_value).expanduser()
    if not output_root.is_absolute():
        output_root = launch_cwd / output_root
    output_root = output_root.resolve()

    asset_root = Path(asset_value).expanduser()
    if not asset_root.is_absolute():
        asset_root = launch_cwd / asset_root
    asset_root = asset_root.resolve()

    isaac_asset_dir = Path(isaac_asset_value).expanduser()
    if not isaac_asset_dir.is_absolute():
        isaac_asset_dir = output_root / isaac_asset_dir
    isaac_asset_dir = isaac_asset_dir.resolve()

    _absolute_option(argv, "--motion_file", launch_cwd)
    if not _has_option(argv, "--motion_file") and not _has_option(
        argv, "--registry_name"
    ):
        raise ValueError(
            "BeyondMimic training requires --motion_file or --registry_name"
        )
    if not _has_option(argv, "--task"):
        argv[:0] = ["--task", DEFAULT_TASK]

    resumed_from: Optional[Path] = None
    if auto_resume and not _has_option(argv, "--resume"):
        experiment_name = _option_value(argv, "--experiment_name", "g1_flat")
        resumed_from = _latest_checkpoint(output_root, experiment_name)
        if resumed_from is not None:
            argv.extend(
                [
                    "--resume",
                    "True",
                    "--load_run",
                    resumed_from.parent.name,
                    "--checkpoint",
                    resumed_from.name,
                ]
            )

    return TrainingLaunch(
        argv=tuple(argv),
        output_root=output_root,
        asset_root=asset_root,
        isaac_asset_dir=isaac_asset_dir,
        resumed_from=resumed_from,
    )


def _validate_assets(asset_root: Path) -> None:
    urdf = asset_root / "unitree_description/urdf/g1/main.urdf"
    mesh_dir = asset_root / "unitree_description/meshes/g1"
    if not urdf.is_file() or not mesh_dir.is_dir():
        raise FileNotFoundError(
            "BeyondMimic G1 assets are missing. Run "
            "`python tools/download_beyondmimic_assets.py` or pass "
            "`--asset-root PATH`."
        )


class BeyondMimicTrainer:
    """Motius adapter for BeyondMimic's native Isaac Lab PPO loop."""

    upstream_commit = UPSTREAM_COMMIT

    @classmethod
    def launch(cls, arguments: Optional[Sequence[str]] = None) -> None:
        main(arguments)

    @classmethod
    def prepare_motion(cls, arguments: Sequence[str]) -> None:
        from .prepare_motion import main as prepare_main

        prepare_main(arguments)

    @classmethod
    def export_policy(cls, arguments: Sequence[str]) -> None:
        from .export_policy import main as export_main

        export_main(arguments)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    launch = build_launch(
        sys.argv[1:] if arguments is None else arguments,
        cwd=Path.cwd(),
    )
    _validate_assets(launch.asset_root)
    launch.output_root.mkdir(parents=True, exist_ok=True)
    launch.isaac_asset_dir.mkdir(parents=True, exist_ok=True)

    vendor = str(VENDOR_ROOT)
    script_root = str(TRAIN_SCRIPT.parent)
    for path in (script_root, vendor):
        if path not in sys.path:
            sys.path.insert(0, path)

    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    previous_asset_root = os.environ.get("MOTIUS_BEYONDMIMIC_ASSET_ROOT")
    previous_isaac_asset_dir = os.environ.get("BEYONDMIMIC_ISAAC_ASSET_DIR")
    try:
        os.environ["MOTIUS_BEYONDMIMIC_ASSET_ROOT"] = str(launch.asset_root)
        os.environ["BEYONDMIMIC_ISAAC_ASSET_DIR"] = str(
            launch.isaac_asset_dir
        )
        os.chdir(launch.output_root)
        sys.argv = [str(TRAIN_SCRIPT), *launch.argv]
        runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        if previous_asset_root is None:
            os.environ.pop("MOTIUS_BEYONDMIMIC_ASSET_ROOT", None)
        else:
            os.environ["MOTIUS_BEYONDMIMIC_ASSET_ROOT"] = previous_asset_root
        if previous_isaac_asset_dir is None:
            os.environ.pop("BEYONDMIMIC_ISAAC_ASSET_DIR", None)
        else:
            os.environ["BEYONDMIMIC_ISAAC_ASSET_DIR"] = (
                previous_isaac_asset_dir
            )


if __name__ == "__main__":
    main()
