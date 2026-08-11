#!/usr/bin/env python3
"""Keep a watcher-managed GPU holder off a node until a target process starts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from tools.gentrack.coevolve import release_occupy_process_groups


def process_exists(substring: str, command_kind: str) -> bool:
    own_pid = os.getpid()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit() or int(proc_dir.name) == own_pid:
            continue
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError):
            continue
        if "release_occupy_until_process.py" in command:
            continue
        if command_kind in command and substring in command:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-substring", required=True)
    parser.add_argument("--command-kind", default="train_agent.py")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--post-start-seconds", type=float, default=120.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_seconds
    released = set()
    while time.monotonic() < deadline:
        released.update(release_occupy_process_groups(grace_seconds=0.5))
        if process_exists(args.process_substring, args.command_kind):
            post_start_deadline = time.monotonic() + args.post_start_seconds
            while time.monotonic() < post_start_deadline:
                released.update(release_occupy_process_groups(grace_seconds=0.5))
                time.sleep(args.interval_seconds)
            print(json.dumps({"status": "target_started", "released_groups": sorted(released)}))
            return
        time.sleep(args.interval_seconds)
    raise SystemExit(f"timed out waiting for process containing {args.process_substring!r}")


if __name__ == "__main__":
    main()
