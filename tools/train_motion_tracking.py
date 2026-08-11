#!/usr/bin/env python3
"""Launch a simulator-owned motion-tracking trainer from a Motius config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAINING_MODULES = {
    "beyondmimic": "motius.trainers.beyondmimic.train",
    "sonic": "motius.trainers.sonic.train",
    "protomotions": "motius.trainers.protomotions.train",
}


def load_command(
    config_path: Path,
    *,
    num_processes: Optional[int] = None,
    overrides: Optional[List[str]] = None,
) -> list[str]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    method = str(payload.get("method", "")).lower()
    expected_module = TRAINING_MODULES.get(method)
    module = payload.get("module")
    if expected_module is None or module != expected_module:
        raise ValueError(
            f"Unsupported trainer declaration: method={method!r}, module={module!r}"
        )

    arguments = [
        os.path.expandvars(str(value))
        for value in payload.get("arguments", [])
    ]
    unresolved = [value for value in arguments if "${" in value]
    if unresolved:
        variables = sorted(
            {
                token.split("}", 1)[0]
                for value in unresolved
                for token in value.split("${")[1:]
            }
        )
        raise ValueError(
            "Set the required training data environment variable(s): "
            + ", ".join(variables)
        )
    arguments.extend(overrides or [])

    processes = (
        int(payload.get("num_processes", 1))
        if num_processes is None
        else int(num_processes)
    )
    if processes < 1:
        raise ValueError("num_processes must be positive")
    if processes == 1:
        return [sys.executable, "-m", module, *arguments]
    return [
        "accelerate",
        "launch",
        "--num_processes",
        str(processes),
        "--module",
        module,
        *arguments,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    command = load_command(
        config_path,
        num_processes=args.num_processes,
        overrides=overrides,
    )
    print(shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
