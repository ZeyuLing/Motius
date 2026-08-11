"""Export a trained BeyondMimic RSL-RL checkpoint to the official ONNX."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
from typing import Optional, Sequence

from .train import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_ISAAC_ASSET_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TASK,
    VENDOR_ROOT,
    _absolute_option,
    _has_option,
    _pop_value,
    _validate_assets,
)


EXPORT_SCRIPT = VENDOR_ROOT / "scripts/rsl_rl/play.py"


def build_export_argv(
    arguments: Sequence[str],
    *,
    cwd: Optional[Path] = None,
) -> tuple[tuple[str, ...], Path, Path, Path]:
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

    output_root = Path(output_value).expanduser()
    if not output_root.is_absolute():
        output_root = launch_cwd / output_root
    asset_root = Path(asset_value).expanduser()
    if not asset_root.is_absolute():
        asset_root = launch_cwd / asset_root
    isaac_asset_dir = Path(isaac_asset_value).expanduser()
    if not isaac_asset_dir.is_absolute():
        isaac_asset_dir = output_root / isaac_asset_dir

    _absolute_option(argv, "--motion_file", launch_cwd)
    if not _has_option(argv, "--motion_file"):
        raise ValueError("Local checkpoint export requires --motion_file")
    if not _has_option(argv, "--task"):
        argv[:0] = ["--task", DEFAULT_TASK]
    if "--export_only" not in argv:
        argv.append("--export_only")
    return (
        tuple(argv),
        output_root.resolve(),
        asset_root.resolve(),
        isaac_asset_dir.resolve(),
    )


def main(arguments: Optional[Sequence[str]] = None) -> None:
    argv, output_root, asset_root, isaac_asset_dir = build_export_argv(
        sys.argv[1:] if arguments is None else arguments,
        cwd=Path.cwd(),
    )
    _validate_assets(asset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    isaac_asset_dir.mkdir(parents=True, exist_ok=True)

    vendor = str(VENDOR_ROOT)
    script_root = str(EXPORT_SCRIPT.parent)
    for path in (script_root, vendor):
        if path not in sys.path:
            sys.path.insert(0, path)

    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    previous_asset_root = os.environ.get("MOTIUS_BEYONDMIMIC_ASSET_ROOT")
    previous_isaac_asset_dir = os.environ.get("BEYONDMIMIC_ISAAC_ASSET_DIR")
    try:
        os.environ["MOTIUS_BEYONDMIMIC_ASSET_ROOT"] = str(asset_root)
        os.environ["BEYONDMIMIC_ISAAC_ASSET_DIR"] = str(isaac_asset_dir)
        os.chdir(output_root)
        sys.argv = [str(EXPORT_SCRIPT), *argv]
        runpy.run_path(str(EXPORT_SCRIPT), run_name="__main__")
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
