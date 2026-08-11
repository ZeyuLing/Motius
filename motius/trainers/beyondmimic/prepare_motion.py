"""Run BeyondMimic's official Isaac Lab motion preprocessing locally."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
from typing import Optional, Sequence

from .train import (
    DEFAULT_ASSET_ROOT,
    VENDOR_ROOT,
    _absolute_option,
    _pop_value,
    _validate_assets,
)


PREPARE_SCRIPT = VENDOR_ROOT / "scripts/csv_to_npz.py"


def build_prepare_argv(
    arguments: Sequence[str],
    *,
    cwd: Optional[Path] = None,
) -> tuple[tuple[str, ...], Path]:
    launch_cwd = (cwd or Path.cwd()).resolve()
    argv = list(arguments)
    asset_default = os.environ.get(
        "MOTIUS_BEYONDMIMIC_ASSET_ROOT",
        str(DEFAULT_ASSET_ROOT),
    )
    asset_value = _pop_value(argv, "--asset-root", asset_default)
    asset_root = Path(asset_value).expanduser()
    if not asset_root.is_absolute():
        asset_root = launch_cwd / asset_root
    for name in ("--input_file", "--output_file"):
        _absolute_option(argv, name, launch_cwd)
    return tuple(argv), asset_root.resolve()


def main(arguments: Optional[Sequence[str]] = None) -> None:
    argv, asset_root = build_prepare_argv(
        sys.argv[1:] if arguments is None else arguments,
        cwd=Path.cwd(),
    )
    _validate_assets(asset_root)
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

    previous_argv = sys.argv
    previous_asset_root = os.environ.get("MOTIUS_BEYONDMIMIC_ASSET_ROOT")
    try:
        os.environ["MOTIUS_BEYONDMIMIC_ASSET_ROOT"] = str(asset_root)
        sys.argv = [str(PREPARE_SCRIPT), *argv]
        runpy.run_path(str(PREPARE_SCRIPT), run_name="__main__")
    finally:
        sys.argv = previous_argv
        if previous_asset_root is None:
            os.environ.pop("MOTIUS_BEYONDMIMIC_ASSET_ROOT", None)
        else:
            os.environ["MOTIUS_BEYONDMIMIC_ASSET_ROOT"] = previous_asset_root


if __name__ == "__main__":
    main()
