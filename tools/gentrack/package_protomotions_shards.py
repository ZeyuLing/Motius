#!/usr/bin/env python3
"""Package a ProtoMotions directory into one native motion library per rank."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def _retry_ceph_metadata(operation, path: Path, attempts: int = 12):
    """Retry transient metadata misses observed immediately after Ceph publish."""
    last_error: Optional[OSError] = None
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(min(0.25 * (attempt + 1), 2.0))
    assert last_error is not None
    raise last_error


def _stat(path: Path):
    return _retry_ceph_metadata(path.stat, path)


def _resolve(path: Path) -> Path:
    return _retry_ceph_metadata(lambda: path.resolve(strict=True), path)


def _acquire_package_lock(package_dir: Path):
    lock_path = package_dir / ".package.lock"
    lock_handle = lock_path.open("a")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return lock_handle


def discover_motions(motion_dir: Path) -> list[Path]:
    motions = sorted(motion_dir.rglob("*.motion"))
    if not motions:
        raise ValueError(f"no .motion files found under {motion_dir}")
    return motions


def balanced_shards(motions: list[Path], num_shards: int) -> list[list[Path]]:
    """Balance both motion count and on-disk bytes across DDP ranks."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if len(motions) < num_shards:
        raise ValueError(
            f"cannot split {len(motions)} motions across {num_shards} non-empty shards"
        )

    base, remainder = divmod(len(motions), num_shards)
    capacities = [base + (rank < remainder) for rank in range(num_shards)]
    shards: list[list[Path]] = [[] for _ in range(num_shards)]
    shard_bytes = [0] * num_shards
    sized = sorted(
        ((_stat(motion).st_size, motion) for motion in motions),
        key=lambda item: (-item[0], item[1].as_posix()),
    )
    for size, motion in sized:
        eligible = [
            rank for rank in range(num_shards)
            if len(shards[rank]) < capacities[rank]
        ]
        rank = min(eligible, key=lambda idx: (shard_bytes[idx], len(shards[idx]), idx))
        shards[rank].append(motion)
        shard_bytes[rank] += size
    return shards


def source_signature(motion_dir: Path, motions: list[Path]) -> str:
    digest = hashlib.sha256()
    for motion in motions:
        stat = _stat(motion)
        digest.update(motion.relative_to(motion_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def output_path(pattern: Path, rank: int) -> Path:
    if "slurmrank" not in pattern.name:
        raise ValueError("output pattern must contain 'slurmrank'")
    return pattern.with_name(pattern.name.replace("slurmrank", str(rank)))


def package_is_current(
    manifest_path: Path,
    pattern: Path,
    signature: str,
    num_shards: int,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("source_signature") != signature:
        return False
    if manifest.get("num_shards") != num_shards:
        return False
    return all(
        output_path(pattern, rank).is_file()
        and _stat(output_path(pattern, rank)).st_size > 0
        for rank in range(num_shards)
    )


def _package_one(
    rank: int,
    shard: list[Path],
    pattern: Path,
    python: Path,
    vendor_root: Path,
    inputs_root: Path,
    logs_root: Path,
    env: dict[str, str],
) -> dict:
    input_dir = inputs_root / f"rank_{rank}"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    for index, motion in enumerate(sorted(shard)):
        # Motion names can contain an entire caption and exceed NAME_MAX after
        # the rank prefix is added. Keep package inputs short and deterministic;
        # source provenance remains available through the package signature.
        resolved_motion = _resolve(motion)
        source_id = hashlib.sha1(str(resolved_motion).encode("utf-8")).hexdigest()[:12]
        link = input_dir / f"{index:06d}_{source_id}{motion.suffix}"
        link.symlink_to(resolved_motion)

    final_output = output_path(pattern, rank)
    temp_output = final_output.with_name(final_output.stem + ".tmp.pt")
    temp_output.unlink(missing_ok=True)
    command = [
        str(python),
        "-m",
        "protomotions.components.motion_lib",
        "--motion-path",
        str(input_dir),
        "--output-file",
        str(temp_output),
        "--device",
        "cpu",
    ]
    log_path = logs_root / f"rank_{rank}.log"
    with log_path.open("w") as log_file:
        result = subprocess.run(
            command,
            cwd=vendor_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"rank {rank} package failed with rc={result.returncode}:\n{tail}"
        )
    os.replace(temp_output, final_output)
    return {
        "rank": rank,
        "num_motions": len(shard),
        "source_bytes": sum(_stat(motion).st_size for motion in shard),
        "package_bytes": _stat(final_output).st_size,
        "output": str(final_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--output-pattern", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--vendor-root", required=True, type=Path)
    args = parser.parse_args()

    motion_dir = args.motion_dir.resolve(strict=True)
    pattern = args.output_pattern.resolve()
    pattern.parent.mkdir(parents=True, exist_ok=True)
    # Multiple elastic retries can overlap briefly while the scheduler tears
    # down the previous container. Serialize the complete transaction;
    # otherwise one process can delete rank inputs while another reads them.
    package_lock = _acquire_package_lock(pattern.parent)
    print(
        json.dumps(
            {
                "event": "package_lock_acquired",
                "lock": str(Path(package_lock.name).resolve()),
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    manifest_path = pattern.parent / "package_manifest.json"
    motions = discover_motions(motion_dir)
    signature = source_signature(motion_dir, motions)
    if package_is_current(
        manifest_path, pattern, signature, args.num_shards
    ):
        print(
            json.dumps(
                {
                    "event": "package_reused",
                    "num_motions": len(motions),
                    "num_shards": args.num_shards,
                    "pattern": str(pattern),
                }
            ),
            flush=True,
        )
        return

    shards = balanced_shards(motions, args.num_shards)
    inputs_root = pattern.parent / "inputs"
    logs_root = pattern.parent / "logs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(args.vendor_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.num_shards) as executor:
        futures = {
            executor.submit(
                _package_one,
                rank,
                shard,
                pattern,
                args.python,
                args.vendor_root,
                inputs_root,
                logs_root,
                env,
            ): rank
            for rank, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: int(item["rank"]))
    manifest = {
        "schema_version": 1,
        "motion_dir": str(motion_dir),
        "source_signature": signature,
        "num_motions": len(motions),
        "num_shards": args.num_shards,
        "output_pattern": str(pattern),
        "shards": results,
    }
    temp_manifest = manifest_path.with_suffix(".tmp.json")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp_manifest, manifest_path)
    print(json.dumps({"event": "package_done", **manifest}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ProtoMotions rank packaging failed: {exc}", file=sys.stderr)
        raise
