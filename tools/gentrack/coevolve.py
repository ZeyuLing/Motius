#!/usr/bin/env python3
"""PhysFlow online-adversarial *co-evolution* orchestrator.

This closes the loop the Stage-1 setup was missing: the trainee tracker's
improvement is fed *back* into the judge that scores the generator, so the
generator never "finishes" -- it keeps being pushed by an ever-stronger tracker.

One OUTER round = one adversarial exchange:

    round r:
      1. build the judge ensemble for this round (see --judge-mode);
      2. GENERATOR phase  : FlowGRPO-post-train HYMotion-G1 for --gen-iters steps
                            against the current judge -> accepted motions stream
                            into the shared per-arm pool;
      3. TRAINEE phase    : PPO+AMP+BeyondMimic the G1 tracker for --trainee-epochs
                            on a snapshot of the pool, warm-started from the
                            previous round's tracker;
      4. JUDGE SYNC       : export the new tracker -> ONNX and make it (part of)
                            next round's judge.

"外层步数" (outer steps) == --num-rounds (number of adversarial exchanges).
"内层步数" (inner steps) == --gen-iters (generator GRPO iters/round) and
                            --trainee-epochs (tracker PPO epochs/round).

Judge-mode ABLATION (set by config, NOT assumed):
  * frozen  : judge is always the released frozen tracker (control / Stage-1).
  * trainee : judge is fully replaced by the latest trainee each round
              (pure adversarial; tests whether a co-adapting judge helps).
  * anchor  : judge = blend of frozen (weight=--anchor-alpha) + latest trainee
              (weight=1-alpha); keeps an unbiased anchor so the generator cannot
              reward-hack a drifting tracker.
  * lagged  : the paper-facing same-data loop. A one-round-lagged tracker gives
              the generator reward; the current trainee is evaluated with zero
              reward weight and is used only for frontier difficulty.

The generator runs in the HYMotion environment; the tracker + ONNX export run in
the IsaacGym py3.8 env. The orchestrator itself is dependency-light.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickletools
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from motius.models.gentrack.tracker_paths import PROTOMOTIONS_ROOT

PROTO = PROTOMOTIONS_ROOT
DEFAULT_PROTO_OUTPUT_ROOT = (
    ROOT / "outputs" / "training" / "protomotions"
)
FROZEN_ONNX = (
    PROTO / "data" / "pretrained_models" / "motion_tracker"
    / "g1-bones-deploy" / "compiled_models" / "unified_pipeline.onnx"
)
NUM_STEPS_PER_EPOCH = 32  # ProtoMotions PPO rollout horizon (base_agent num_steps)
_FILE_IDENTITY_CACHE = {}
_PICKLE_MARK = object()
_MAX_GENERATOR_META_PICKLE_BYTES = 1024 * 1024
_MAX_GENERATOR_META_PICKLE_OPS = 10_000
PAPER_ROUND_LOCAL_GENERATED_COUNT = 13_337
PAPER_ROUND_LOCAL_PUBLIC_COUNT = 13_337


def _lexical_absolute(path: Path) -> str:
    """Return an absolute path without resolving storage-mount aliases."""
    return os.path.abspath(os.fspath(path))


def file_identity(path: Path) -> dict:
    """Return a content identity for one required file.

    Paths remain lexical so the identity is stable across equivalent Ceph mount
    aliases; content equality is established by SHA-256 rather than ``resolve``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    cache_key = (
        _lexical_absolute(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    digest = _FILE_IDENTITY_CACHE.get(cache_key)
    if digest is None:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        _FILE_IDENTITY_CACHE[cache_key] = digest
    return {
        "path": _lexical_absolute(path),
        "sha256": digest,
        "size_bytes": int(stat.st_size),
    }


def optional_file_identity(path: Path | None) -> dict | None:
    return None if path is None else file_identity(path)


def checkpoint_identity(path: Path) -> dict:
    """Bind a generator checkpoint directory to its model-weight file."""
    path = Path(path)
    if path.is_file():
        return file_identity(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    identity_file = None
    for filename in ("model.safetensors", "model.pt"):
        candidate = path / filename
        if candidate.is_file():
            identity_file = candidate
            break
    if identity_file is None:
        raise FileNotFoundError(
            f"checkpoint has no model.safetensors or model.pt: {path}"
        )
    identity = file_identity(identity_file)
    return {
        "path": _lexical_absolute(path),
        "identity_file": identity["path"],
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _pop_pickle_mark(stack: list) -> list:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is _PICKLE_MARK:
            values = stack[index + 1 :]
            del stack[index:]
            return values
    raise RuntimeError("primitive pickle has no matching MARK")


def _validate_primitive_pickle_value(
    value,
    *,
    active_containers: set[int] | None = None,
) -> None:
    """Reject non-primitive or cyclic values without invoking pickle."""
    if type(value) in (type(None), bool, int, float, str, bytes):
        return
    if type(value) not in (dict, list, tuple):
        raise RuntimeError(
            f"primitive pickle contains unsupported value type: {type(value).__name__}"
        )
    active_containers = active_containers or set()
    identity = id(value)
    if identity in active_containers:
        raise RuntimeError("primitive pickle contains a cyclic container")
    active_containers.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) not in (str, int, bool):
                    raise RuntimeError(
                        "primitive pickle contains a non-primitive dict key"
                    )
                _validate_primitive_pickle_value(
                    item,
                    active_containers=active_containers,
                )
        else:
            for item in value:
                _validate_primitive_pickle_value(
                    item,
                    active_containers=active_containers,
                )
    finally:
        active_containers.remove(identity)


def safe_load_primitive_pickle(data: bytes) -> dict:
    """Decode a small primitive-only pickle with a non-executing stack machine.

    This intentionally does not call :func:`pickle.loads`. Any opcode capable
    of importing globals, constructing objects, invoking reducers, loading
    persistent IDs, or extensions is outside the whitelist and fails closed.
    """
    if not data or len(data) > _MAX_GENERATOR_META_PICKLE_BYTES:
        raise RuntimeError(
            "generator meta pickle is empty or exceeds the 1 MiB safety limit"
        )
    if data[-1:] != b".":
        raise RuntimeError("generator meta pickle has trailing or missing data")

    stack: list = []
    memo: dict[int, object] = {}
    saw_stop = False
    for operation_index, (opcode, argument, position) in enumerate(
        pickletools.genops(data),
        start=1,
    ):
        if operation_index > _MAX_GENERATOR_META_PICKLE_OPS:
            raise RuntimeError("generator meta pickle exceeds the opcode limit")
        name = opcode.name
        if operation_index == 1 and name != "PROTO":
            raise RuntimeError("generator meta pickle must begin with PROTO")

        if name == "PROTO":
            if type(argument) is not int or not 0 <= argument <= 5:
                raise RuntimeError(
                    f"unsupported generator meta pickle protocol: {argument!r}"
                )
        elif name == "FRAME":
            if type(argument) is not int or argument < 0:
                raise RuntimeError("generator meta pickle has an invalid FRAME")
        elif name == "MARK":
            stack.append(_PICKLE_MARK)
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4"):
            if type(argument) is not int:
                raise RuntimeError(f"{name} did not decode to an integer")
            stack.append(argument)
        elif name in ("FLOAT", "BINFLOAT"):
            if type(argument) is not float or not math.isfinite(argument):
                raise RuntimeError(f"{name} did not decode to a finite float")
            stack.append(argument)
        elif name in (
            "UNICODE",
            "BINUNICODE",
            "SHORT_BINUNICODE",
            "BINUNICODE8",
        ):
            if type(argument) is not str:
                raise RuntimeError(f"{name} did not decode to text")
            stack.append(argument)
        elif name in ("BINBYTES", "SHORT_BINBYTES", "BINBYTES8"):
            if type(argument) is not bytes:
                raise RuntimeError(f"{name} did not decode to bytes")
            stack.append(argument)
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "DICT":
            items = _pop_pickle_mark(stack)
            if len(items) % 2:
                raise RuntimeError("primitive pickle DICT has an odd item count")
            result = {}
            for index in range(0, len(items), 2):
                result[items[index]] = items[index + 1]
            stack.append(result)
        elif name == "SETITEM":
            if len(stack) < 3 or type(stack[-3]) is not dict:
                raise RuntimeError("primitive pickle SETITEM has no dict")
            value = stack.pop()
            key = stack.pop()
            stack[-1][key] = value
        elif name == "SETITEMS":
            items = _pop_pickle_mark(stack)
            if not stack or type(stack[-1]) is not dict or len(items) % 2:
                raise RuntimeError("primitive pickle SETITEMS is malformed")
            for index in range(0, len(items), 2):
                stack[-1][items[index]] = items[index + 1]
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "LIST":
            stack.append(_pop_pickle_mark(stack))
        elif name == "APPEND":
            if len(stack) < 2 or type(stack[-2]) is not list:
                raise RuntimeError("primitive pickle APPEND has no list")
            stack[-2].append(stack.pop())
        elif name == "APPENDS":
            items = _pop_pickle_mark(stack)
            if not stack or type(stack[-1]) is not list:
                raise RuntimeError("primitive pickle APPENDS has no list")
            stack[-1].extend(items)
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "TUPLE":
            stack.append(tuple(_pop_pickle_mark(stack)))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            count = int(name[-1])
            if len(stack) < count:
                raise RuntimeError(f"primitive pickle {name} underflows")
            values = stack[-count:]
            del stack[-count:]
            stack.append(tuple(values))
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            if not stack or type(argument) is not int or argument < 0:
                raise RuntimeError(f"primitive pickle {name} is malformed")
            memo[argument] = stack[-1]
        elif name == "MEMOIZE":
            if not stack:
                raise RuntimeError("primitive pickle MEMOIZE has an empty stack")
            memo[len(memo)] = stack[-1]
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            if type(argument) is not int or argument not in memo:
                raise RuntimeError(f"primitive pickle {name} has an unknown memo")
            stack.append(memo[argument])
        elif name == "POP":
            if not stack:
                raise RuntimeError("primitive pickle POP underflows")
            stack.pop()
        elif name == "POP_MARK":
            _pop_pickle_mark(stack)
        elif name == "DUP":
            if not stack:
                raise RuntimeError("primitive pickle DUP underflows")
            stack.append(stack[-1])
        elif name == "STOP":
            if len(stack) != 1:
                raise RuntimeError(
                    f"primitive pickle STOP has stack depth {len(stack)}"
                )
            if position != len(data) - 1:
                raise RuntimeError(
                    "generator meta pickle contains data after STOP"
                )
            saw_stop = True
            break
        else:
            raise RuntimeError(
                f"unsafe or unsupported generator meta pickle opcode: {name}"
            )

    if not saw_stop:
        raise RuntimeError("generator meta pickle has no STOP opcode")
    value = stack[0]
    _validate_primitive_pickle_value(value)
    if type(value) is not dict:
        raise RuntimeError("generator meta pickle root is not a dict")
    return value


def validate_generator_checkpoint_metadata(
    checkpoint: Path,
    expected_global_step: int,
) -> dict:
    """Safely bind and validate the MMEngine ``meta.pt`` completion record."""
    checkpoint = Path(checkpoint)
    meta_path = checkpoint / "meta.pt"
    meta_identity = file_identity(meta_path)
    try:
        with zipfile.ZipFile(meta_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("generator meta archive has duplicate members")
            pickle_members = [
                name for name in names if name.endswith("/data.pkl")
            ]
            if len(pickle_members) != 1:
                raise RuntimeError(
                    "generator meta archive must contain exactly one */data.pkl"
                )
            info = archive.getinfo(pickle_members[0])
            if info.flag_bits & 0x1:
                raise RuntimeError("generator meta archive is encrypted")
            if (
                info.file_size < 1
                or info.file_size > _MAX_GENERATOR_META_PICKLE_BYTES
            ):
                raise RuntimeError(
                    "generator meta pickle exceeds the safe size limit"
                )
            payload = safe_load_primitive_pickle(archive.read(info))
    except (OSError, zipfile.BadZipFile, ValueError, EOFError, KeyError) as error:
        raise RuntimeError(
            f"cannot safely decode generator checkpoint metadata: {meta_path}"
        ) from error

    global_step = payload.get("global_step")
    if type(global_step) is not int:
        raise RuntimeError(
            f"generator checkpoint metadata lacks integer global_step: {meta_path}"
        )
    if global_step != expected_global_step:
        raise RuntimeError(
            f"generator checkpoint global_step {global_step} does not equal "
            f"exact target {expected_global_step}: {meta_path}"
        )
    current_epoch = payload.get("current_epoch")
    if current_epoch is not None and type(current_epoch) is not int:
        raise RuntimeError(
            f"generator checkpoint current_epoch is not an integer: {meta_path}"
        )
    return {
        "identity": meta_identity,
        "global_step": global_step,
        "current_epoch": current_epoch,
    }


def write_immutable_json(path: Path, payload: dict) -> dict:
    """Create a deterministic JSON artifact once, or verify an identical retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    f"immutable JSON artifact differs from existing file: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return file_identity(path)


def _require_path_within(path: Path, root: Path, label: str) -> str:
    resolved_path = Path(path).resolve(strict=True)
    resolved_root = Path(root).resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(
            f"{label} escapes its required source root: "
            f"{resolved_path} not under {resolved_root}"
        ) from error
    return relative.as_posix()


def write_round_local_snapshot_manifest(
    manifest_path: Path,
    *,
    round_index: int,
    snapshot_dir: Path,
    generated_sources: dict[str, Path],
    public_sources: dict[str, Path],
    generated_root: Path,
    public_root: Path,
    expected_generated_count: int = PAPER_ROUND_LOCAL_GENERATED_COUNT,
    expected_public_count: int = PAPER_ROUND_LOCAL_PUBLIC_COUNT,
) -> dict:
    """Validate and attest an exact generated/public symlink snapshot."""
    snapshot_dir = Path(snapshot_dir)
    generated_root = Path(generated_root)
    public_root = Path(public_root)
    if len(generated_sources) != expected_generated_count:
        raise RuntimeError(
            f"round {round_index}: generated snapshot source count "
            f"{len(generated_sources)} != {expected_generated_count}"
        )
    if len(public_sources) != expected_public_count:
        raise RuntimeError(
            f"round {round_index}: public snapshot source count "
            f"{len(public_sources)} != {expected_public_count}"
        )
    overlap = set(generated_sources) & set(public_sources)
    if overlap:
        raise RuntimeError(
            f"round {round_index}: generated/public snapshot names overlap: "
            f"{sorted(overlap)[:3]}"
        )

    expected_names = set(generated_sources) | set(public_sources)
    actual_names = {
        path.name for path in snapshot_dir.glob("*.motion")
    } if snapshot_dir.is_dir() else set()
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"round {round_index}: snapshot manifest membership is not exact: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_examples={missing[:3]} "
            f"unexpected_examples={unexpected[:3]}"
        )

    entries = []
    for name in sorted(expected_names):
        if name in generated_sources:
            source_kind = "generated"
            expected_source = Path(generated_sources[name])
            source_root = generated_root
        else:
            source_kind = "public"
            expected_source = Path(public_sources[name])
            source_root = public_root
        source_relative_path = _require_path_within(
            expected_source,
            source_root,
            f"round {round_index} {source_kind} source {name}",
        )
        link = snapshot_dir / name
        if not link.is_symlink():
            raise RuntimeError(
                f"round {round_index}: snapshot member is not a symlink: {link}"
            )
        lexical_target = os.readlink(link)
        target_path = Path(lexical_target)
        if not target_path.is_absolute():
            target_path = link.parent / target_path
        if os.path.realpath(target_path) != os.path.realpath(expected_source):
            raise RuntimeError(
                f"round {round_index}: snapshot target mismatch for {name}: "
                f"{lexical_target!r} != {expected_source}"
            )
        target_relative_path = _require_path_within(
            target_path,
            source_root,
            f"round {round_index} {source_kind} target {name}",
        )
        if target_relative_path != source_relative_path:
            raise RuntimeError(
                f"round {round_index}: snapshot source/target relative paths "
                f"differ for {name}"
            )
        entries.append(
            {
                "name": name,
                "source_kind": source_kind,
                "lexical_symlink_target": lexical_target,
                "source_relative_path": source_relative_path,
            }
        )

    payload = {
        "schema_version": 1,
        "round": round_index,
        "policy": "round-local-replacement",
        "snapshot_path": _lexical_absolute(snapshot_dir),
        "generated_root": _lexical_absolute(generated_root),
        "public_root": _lexical_absolute(public_root),
        "generated_count": len(generated_sources),
        "public_count": len(public_sources),
        "total_count": len(entries),
        "entries": entries,
    }
    return write_immutable_json(manifest_path, payload)


def _require_record_matches_file(record: dict, label: str) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise RuntimeError(f"{label} identity is missing or malformed")
    actual = file_identity(Path(record["path"]))
    if (
        actual["sha256"] != record.get("sha256")
        or actual["size_bytes"] != record.get("size_bytes")
    ):
        raise RuntimeError(f"{label} identity does not match its file")
    return actual


def write_judge_export_manifest(
    manifest_path: Path,
    *,
    round_index: int,
    source_tracker_identity: dict,
    onnx_path: Path,
    yaml_path: Path,
) -> dict:
    """Bind one persistent tracker checkpoint to its ONNX/YAML export."""
    source = _require_record_matches_file(
        source_tracker_identity,
        "judge export source tracker",
    )
    onnx_identity = file_identity(onnx_path)
    yaml_identity = file_identity(yaml_path)
    payload = {
        "schema_version": 1,
        "round": round_index,
        "source_tracker_checkpoint": source,
        "onnx": onnx_identity,
        "yaml": yaml_identity,
    }
    manifest_identity = write_immutable_json(manifest_path, payload)
    return {
        "identity": manifest_identity,
        "onnx": onnx_identity,
        "yaml": yaml_identity,
    }


def ephemeral_checkpoint_blockers(path: Path) -> list[str]:
    """Return paper/resume blockers for a checkpoint on node-local storage."""
    absolute = Path(_lexical_absolute(Path(path)))
    for root in (Path("/tmp"), Path("/dev/shm")):
        try:
            absolute.relative_to(root)
        except ValueError:
            continue
        return [f"tracker_output_checkpoint_is_ephemeral:{absolute}"]
    return []


def persistent_tracker_checkpoint_path(
    checkpoint_root: Path,
    round_index: int,
) -> Path:
    """Return the immutable, paper-facing tracker checkpoint for one round."""
    return Path(checkpoint_root) / f"r{round_index}" / "last.ckpt"


def persist_file_atomic_copy(source: Path, destination: Path) -> dict:
    """Publish an independent, content-verified copy without partial visibility.

    The source may live on node-local storage and may itself be a hard link.
    Copying into a same-directory temporary file before atomically linking the
    temporary name into place guarantees that the persistent artifact neither
    aliases the source inode nor becomes visible before the copy is complete.
    Existing destinations are immutable: an identical retry is accepted, while
    different bytes fail closed.
    """
    source = Path(source)
    destination = Path(destination)
    source_before = file_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination_identity = file_identity(destination)
        if (
            destination_identity["sha256"] != source_before["sha256"]
            or destination_identity["size_bytes"] != source_before["size_bytes"]
        ):
            raise RuntimeError(
                "persistent checkpoint differs from existing immutable artifact: "
                f"{destination}"
            )
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_dev,
            source_stat.st_ino,
        ) == (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            raise RuntimeError(
                "existing persistent tracker checkpoint aliases its runtime "
                f"source: {destination}"
            )
        return destination_identity

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())

        source_after = file_identity(source)
        temporary_identity = file_identity(temporary)
        expected = (
            source_before["sha256"],
            source_before["size_bytes"],
        )
        if (
            source_after["sha256"],
            source_after["size_bytes"],
        ) != expected:
            raise RuntimeError(
                f"tracker checkpoint changed while being persisted: {source}"
            )
        if (
            temporary_identity["sha256"],
            temporary_identity["size_bytes"],
        ) != expected:
            raise RuntimeError(
                f"persistent tracker checkpoint copy failed verification: {temporary}"
            )

        try:
            os.link(temporary, destination)
        except FileExistsError:
            destination_identity = file_identity(destination)
            if (
                destination_identity["sha256"],
                destination_identity["size_bytes"],
            ) != expected:
                raise RuntimeError(
                    "persistent checkpoint was concurrently published with "
                    f"different bytes: {destination}"
                )

        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

    destination_identity = file_identity(destination)
    if (
        destination_identity["sha256"],
        destination_identity["size_bytes"],
    ) != (
        source_before["sha256"],
        source_before["size_bytes"],
    ):
        raise RuntimeError(
            f"persistent tracker checkpoint failed post-publish verification: "
            f"{destination}"
        )
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (
        source_stat.st_dev,
        source_stat.st_ino,
    ) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise RuntimeError(
            f"persistent tracker checkpoint aliases its runtime source: {destination}"
        )
    return destination_identity


def validate_persistent_tracker_boundary(
    arm: Path,
    checkpoint_root: Path,
    round_index: int,
    *,
    require_exact_budget: bool = False,
) -> Path:
    """Require a complete attestation and its immutable persistent checkpoint."""
    arm = Path(arm)
    checkpoint = persistent_tracker_checkpoint_path(
        checkpoint_root,
        round_index,
    )
    attestation_path = arm / "round_attestations" / f"r{round_index}.json"
    if not attestation_path.is_file():
        raise FileNotFoundError(
            f"round {round_index}: complete round attestation is missing: "
            f"{attestation_path}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"round {round_index}: persistent tracker checkpoint is missing: "
            f"{checkpoint}"
        )

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("status") != "complete":
        raise RuntimeError(
            f"round {round_index}: attestation status is not complete"
        )
    if attestation.get("round") != round_index:
        raise RuntimeError(
            f"round {round_index}: attestation records round "
            f"{attestation.get('round')!r}"
        )
    blockers = attestation.get("paper_blockers")
    if not isinstance(blockers, list) or blockers:
        raise RuntimeError(
            f"round {round_index}: attestation has paper blockers: {blockers!r}"
        )

    tracker = attestation.get("tracker")
    if not isinstance(tracker, dict) or tracker.get("skipped") is not False:
        raise RuntimeError(
            f"round {round_index}: attestation has no completed tracker update"
        )
    if tracker.get("output_persisted") is not True:
        raise RuntimeError(
            f"round {round_index}: tracker output is not attested as persistent"
        )
    recorded_identity = tracker.get("output")
    if not isinstance(recorded_identity, dict):
        raise RuntimeError(
            f"round {round_index}: tracker output identity is missing"
        )
    actual_identity = file_identity(checkpoint)
    for key in ("sha256", "size_bytes"):
        if recorded_identity.get(key) != actual_identity[key]:
            raise RuntimeError(
                f"round {round_index}: persistent tracker checkpoint {key} "
                "does not match its attestation"
            )

    if (
        require_exact_budget
        and tracker.get("exact_budget_required") is not True
    ):
        raise RuntimeError(
            f"round {round_index}: exact tracker budget was not required"
        )
    if tracker.get("exact_budget_required"):
        required_fields = (
            "input_epoch",
            "output_epoch",
            "input_step_count",
            "output_step_count",
            "requested_epochs",
            "requested_transition_steps",
            "actual_transition_steps",
            "num_envs",
            "world_size",
            "steps_per_epoch",
        )
        if any(type(tracker.get(field)) is not int for field in required_fields):
            raise RuntimeError(
                f"round {round_index}: exact-budget attestation is incomplete"
            )
        expected_steps_per_epoch = (
            tracker["num_envs"]
            * NUM_STEPS_PER_EPOCH
            * tracker["world_size"]
        )
        expected_steps = tracker["requested_epochs"] * expected_steps_per_epoch
        exact_budget_satisfied = (
            tracker.get("exact_budget_satisfied") is True
            and tracker["steps_per_epoch"] == expected_steps_per_epoch
            and tracker["output_epoch"]
            == tracker["input_epoch"] + tracker["requested_epochs"]
            and tracker["requested_transition_steps"] == expected_steps
            and tracker["actual_transition_steps"] == expected_steps
            and tracker["output_step_count"] - tracker["input_step_count"]
            == expected_steps
        )
        if not exact_budget_satisfied:
            raise RuntimeError(
                f"round {round_index}: exact tracker budget is not satisfied"
            )
    return checkpoint


def select_tracker_input_checkpoint(
    arm: Path,
    round_index: int,
    trainee_init_checkpoint: Path,
    restart_each_round: bool,
    persistent_checkpoint_root: Path | None = None,
    proto_root: Path = PROTO,
    runtime_checkpoint_root: Path | None = None,
    require_exact_budget: bool = False,
) -> Path:
    """Select T_r, validating a persistent completed boundary when enabled."""
    if round_index == 0 or restart_each_round:
        return Path(trainee_init_checkpoint)
    if persistent_checkpoint_root is not None:
        return validate_persistent_tracker_boundary(
            arm,
            persistent_checkpoint_root,
            round_index - 1,
            require_exact_budget=require_exact_budget,
        )
    checkpoint_root = (
        Path(runtime_checkpoint_root)
        if runtime_checkpoint_root is not None
        else Path(proto_root) / "results"
    )
    return (
        checkpoint_root
        / f"{Path(arm).name}_co_r{round_index - 1}"
        / "last.ckpt"
    )


def sanitize_pythonpath_for_py38(value: str) -> str:
    """Remove parent interpreter packages before launching ProtoMotions py3.8."""
    compatible = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        normalized = entry.replace("\\", "/")
        if normalized.rstrip("/").endswith("/gentrack_mujoco_site"):
            continue
        if "/site-packages" in normalized and "/python3.8/" not in normalized:
            continue
        compatible.append(entry)
    return os.pathsep.join(dict.fromkeys(compatible))


def training_max_steps_for_epochs(
    current_epoch: int,
    added_epochs: int,
    num_envs: int,
    world_size: int,
) -> int:
    """Match ProtoMotions' epoch conversion for single- or multi-GPU runs."""
    return (
        (current_epoch + added_epochs)
        * num_envs
        * NUM_STEPS_PER_EPOCH
        * world_size
    )


def log(state_file: Path, event: str, **kw):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    with open(state_file, "a") as f:
        f.write(line + "\n")


def run(cmd, env, log_path: Path, cwd=None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as lf:
        lf.write(f"\n=== {time.strftime('%H:%M:%S')} RUN: {' '.join(map(str, cmd))} ===\n")
        lf.flush()
        p = subprocess.run(list(map(str, cmd)), env=env, cwd=cwd,
                           stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode


def release_occupy_process_groups(grace_seconds: float = 3.0) -> list[int]:
    """Release only ``occupy.py`` process groups before an all-GPU trainee.

    The fixed A100 pool has an external watcher that starts ``occupy.py`` while
    the single-GPU generator is running.  ``occupy.py`` uses multiprocessing,
    so terminating only its parent leaves one CUDA holder per GPU alive.  The
    watcher itself belongs to another process group and is deliberately left
    untouched.
    """
    own_group = os.getpgrp()
    groups = set()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            argv = (proc_dir / "cmdline").read_bytes().split(b"\0")
            if not any(Path(os.fsdecode(arg)).name == "occupy.py" for arg in argv if arg):
                continue
            group = os.getpgid(int(proc_dir.name))
            if group > 1 and group != own_group:
                groups.add(group)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue

    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    remaining = set(groups)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        for group in list(remaining):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                remaining.remove(group)
    for group in remaining:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return sorted(groups)


def newest_ckpt_dir(gen_work: Path):
    cks = sorted(gen_work.glob("checkpoint-iter_*"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None


def state_has_round_event(state_path: Path, event: str, round_index: int) -> bool:
    if not state_path.is_file():
        return False
    for raw_line in state_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if (
            payload.get("event") == event
            and payload.get("round") == round_index
        ):
            return True
    return False


def completed_generator_checkpoint(
    gen_work: Path,
    expected_iter: int,
    state_path: Path,
    round_index: int,
):
    """Return only a checkpoint proven to represent a completed generator round.

    Elastic preemption can leave an otherwise valid checkpoint at an intermediate
    save interval. Treating that directory as round completion silently shortens
    the generator budget and advances the tracker on a partial round.
    """
    expected = gen_work / f"checkpoint-iter_{expected_iter}"
    checkpoints = sorted(
        path for path in gen_work.glob("checkpoint-iter_*") if path.is_dir()
    )
    has_done_event = state_has_round_event(state_path, "gen_done", round_index)
    if expected.is_dir() and has_done_event:
        return expected
    if expected.is_dir():
        raise RuntimeError(
            f"round {round_index}: exact generator checkpoint {expected} exists "
            "without a matching gen_done event; refusing an ambiguous resume"
        )
    if checkpoints:
        names = ", ".join(path.name for path in checkpoints)
        raise RuntimeError(
            f"round {round_index}: partial generator checkpoints found ({names}); "
            f"expected checkpoint-iter_{expected_iter} plus gen_done"
        )
    if has_done_event:
        raise RuntimeError(
            f"round {round_index}: gen_done exists but "
            f"checkpoint-iter_{expected_iter} is missing"
        )
    return None


def select_generator_input_checkpoint(
    arm: Path,
    round_index: int,
    gen_init_checkpoint: Path,
    gen_iters: int,
    state_path: Path,
    reset_each_round: bool,
) -> Path:
    """Select G_r for round ``r`` and fail closed on a broken continuation."""
    if round_index == 0 or reset_each_round:
        return Path(gen_init_checkpoint)
    previous = completed_generator_checkpoint(
        Path(arm) / "gen" / f"r{round_index - 1}",
        gen_iters,
        state_path,
        round_index - 1,
    )
    if previous is None:
        raise FileNotFoundError(
            f"round {round_index}: completed generator checkpoint for round "
            f"{round_index - 1} is missing"
        )
    return previous


def validate_refreshed_generator_bank(
    ready_path: Path,
    checkpoint: Path,
    expected_count: int,
) -> tuple[Path, dict]:
    """Require a complete round-local qpos bank from the selected generator."""
    payload = json.loads(ready_path.read_text(encoding="utf-8"))
    recorded_checkpoint = Path(payload["checkpoint"]).resolve()
    if recorded_checkpoint != checkpoint.resolve():
        raise RuntimeError(
            "refreshed generator bank checkpoint mismatch: "
            f"{recorded_checkpoint} != {checkpoint.resolve()}"
        )
    count = int(payload["count"])
    if count != expected_count:
        raise RuntimeError(
            f"refreshed generator bank has {count} references; "
            f"expected exactly {expected_count}"
        )
    replay_manifest = Path(payload["replay_manifest"])
    replay = json.loads(replay_manifest.read_text(encoding="utf-8"))
    if int(replay["count"]) != expected_count:
        raise RuntimeError(
            f"replay manifest has {replay['count']} references; "
            f"expected exactly {expected_count}"
        )
    qpos_dir = replay_manifest.parent / "qpos_npz"
    if not qpos_dir.is_dir():
        raise FileNotFoundError(qpos_dir)
    return qpos_dir, payload


def validate_refreshed_proto_bank(
    done_path: Path,
    qpos_dir: Path,
    expected_count: int,
) -> tuple[Path, dict]:
    """Require an exact, source-matched ProtoMotions conversion."""
    payload = json.loads(done_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"ProtoMotions conversion is not complete: {done_path}")
    if int(payload["count"]) != expected_count:
        raise RuntimeError(
            f"ProtoMotions conversion has {payload['count']} references; "
            f"expected exactly {expected_count}"
        )
    manifest_path = Path(payload["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["converted_count"]) != expected_count:
        raise RuntimeError(
            f"ProtoMotions manifest has {manifest['converted_count']} references; "
            f"expected exactly {expected_count}"
        )
    recorded_input = Path(manifest["input_dir"]).resolve()
    if recorded_input != qpos_dir.resolve():
        raise RuntimeError(
            "ProtoMotions conversion input mismatch: "
            f"{recorded_input} != {qpos_dir.resolve()}"
        )
    motion_dir = manifest_path.parent
    motion_count = len(list(motion_dir.glob("*.motion")))
    if motion_count != expected_count:
        raise RuntimeError(
            f"ProtoMotions conversion directory has {motion_count} motions; "
            f"expected exactly {expected_count}"
        )
    return motion_dir, payload


def ckpt_progress(path: Path):
    import torch
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0)), int(ck.get("step_count", 0))
    except Exception:
        return 0, 0


def ckpt_epoch(path: Path) -> int:
    return ckpt_progress(path)[0]


def validate_trainee_checkpoint_progress(
    path: Path,
    target_epoch: int,
    previous_step_count: int,
    *,
    require_exact: bool = False,
    expected_added_steps: int | None = None,
):
    """Fail closed unless a nominally successful trainee reached its budget."""
    if not path.is_file():
        raise RuntimeError(f"trainee checkpoint does not exist: {path}")
    epoch, step_count = ckpt_progress(path)
    if require_exact and expected_added_steps is None:
        raise ValueError(
            "expected_added_steps is required when exact budget validation is enabled"
        )
    if require_exact and epoch != target_epoch:
        raise RuntimeError(
            f"trainee checkpoint epoch {epoch} does not equal exact target "
            f"{target_epoch}: {path}"
        )
    if not require_exact and epoch < target_epoch:
        raise RuntimeError(
            f"trainee checkpoint epoch {epoch} is below target {target_epoch}: "
            f"{path}"
        )
    step_delta = step_count - previous_step_count
    if require_exact and step_delta != expected_added_steps:
        raise RuntimeError(
            f"trainee checkpoint step delta {step_delta} does not equal exact "
            f"budget {expected_added_steps}: {path}"
        )
    if not require_exact and step_delta <= 0:
        raise RuntimeError(
            f"trainee checkpoint step_count {step_count} did not advance beyond "
            f"{previous_step_count}: {path}"
        )
    return epoch, step_count


def build_judge_spec(
    mode: str,
    alpha: float,
    quality_onnx,
    trainee_onnx,
    spec_path: Path,
):
    if mode == "frozen":
        judges = [{"onnx": str(FROZEN_ONNX), "weight": 1.0, "name": "frozen"}]
    elif mode == "lagged":
        if quality_onnx is None:
            raise ValueError("lagged judge mode requires --initial-judge-onnx")
        judges = [{"onnx": str(quality_onnx), "weight": 1.0, "name": "quality"}]
        if trainee_onnx is not None and Path(trainee_onnx) != Path(quality_onnx):
            judges.append(
                {"onnx": str(trainee_onnx), "weight": 0.0, "name": "trainee"}
            )
    elif trainee_onnx is None:
        raise ValueError(
            f"judge mode {mode!r} requires a same-data tracker judge. "
            "Pass --initial-judge-onnx for round 0 or resume from a round that "
            "already exported judge_onnx/r*/unified_pipeline.onnx."
        )
    elif mode == "trainee":
        judges = [{"onnx": str(trainee_onnx), "weight": 1.0, "name": "trainee"}]
    elif mode == "anchor":
        judges = [
            {"onnx": str(FROZEN_ONNX), "weight": float(alpha), "name": "frozen"},
            {"onnx": str(trainee_onnx), "weight": float(1.0 - alpha), "name": "trainee"},
        ]
    else:
        raise ValueError(f"bad judge mode {mode}")
    write_immutable_json(spec_path, {"judges": judges})
    return judges


def initial_judge_clock(mode: str, initial_judge: Path | None):
    """Return the Q/T state before round 0."""
    quality = initial_judge
    trainee = None if mode == "lagged" else initial_judge
    return quality, trainee


def advance_judge_clock(
    mode: str,
    quality_judge: Path | None,
    trainee_judge: Path | None,
    updated_trainee: Path,
):
    """Advance the two-clock state after exporting T_{r+1}."""
    if mode == "lagged":
        quality_judge = trainee_judge or quality_judge
    return quality_judge, updated_trainee


def resumed_judge_clock(
    mode: str,
    initial_judge: Path | None,
    arm: Path,
    start_round: int,
):
    """Reconstruct Q/T at a complete round boundary without silent fallback."""
    quality, trainee = initial_judge_clock(mode, initial_judge)
    if start_round <= 0 or mode == "frozen":
        return quality, trainee

    latest = (
        Path(arm)
        / "judge_onnx"
        / f"r{start_round - 1}"
        / "unified_pipeline.onnx"
    )
    if not latest.is_file():
        raise FileNotFoundError(
            f"round {start_round}: resumed trainee judge is missing: {latest}"
        )
    trainee = latest
    if mode == "lagged" and start_round > 1:
        quality = (
            Path(arm)
            / "judge_onnx"
            / f"r{start_round - 2}"
            / "unified_pipeline.onnx"
        )
        if not quality.is_file():
            raise FileNotFoundError(
                f"round {start_round}: resumed quality judge is missing: {quality}"
            )
    return quality, trainee


def judge_spec_path(arm: Path, round_index: int) -> Path:
    return Path(arm) / "judge_specs" / f"r{round_index}.json"


def snapshot_membership_is_ready(
    mode: str,
    required_names: set[str],
    existing_names: set[str],
) -> bool:
    if not required_names:
        return False
    if mode == "round-local-replacement":
        return existing_names == required_names
    return required_names.issubset(existing_names)


def validate_snapshot_membership(
    mode: str,
    required_names: set[str],
    actual_names: set[str],
) -> None:
    if mode != "round-local-replacement":
        return
    if actual_names == required_names:
        return
    missing = sorted(required_names - actual_names)
    unexpected = sorted(actual_names - required_names)
    raise RuntimeError(
        "round-local replacement snapshot is not exact: "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"missing_examples={missing[:3]} unexpected_examples={unexpected[:3]}"
    )


def parse_round_cfg_options(spec: str):
    """Parse ``round:cfg,cfg;round:cfg`` generator override fragments."""
    out = {}
    for chunk in (spec or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "bad --gen-cfg-options-by-round entry "
                f"{chunk!r}; expected ROUND:KEY=VALUE[,KEY=VALUE]"
            )
        round_s, opts_s = chunk.split(":", 1)
        r = int(round_s.strip())
        opts = [x.strip() for x in opts_s.split(",") if x.strip()]
        out.setdefault(r, []).extend(opts)
    return out


def copy_motion_dir(
    src: Path,
    dst: Path,
    prefix: str = "",
    symlink: bool = False,
) -> int:
    """Flatten motions into a snapshot without name collisions."""
    copied = 0
    if not src.is_dir():
        return copied
    for m in sorted(src.rglob("*.motion")):
        relative = m.relative_to(src).as_posix()
        digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
        out_name = f"{prefix}{digest}_{m.name}"
        output = dst / out_name
        if symlink:
            output.symlink_to(m)
        else:
            shutil.copy2(m, output)
        copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", required=True)
    ap.add_argument(
        "--judge-mode",
        required=True,
        choices=["frozen", "trainee", "anchor", "lagged"],
    )
    ap.add_argument("--anchor-alpha", type=float, default=0.5,
                    help="weight on the frozen judge in anchor mode (trainee gets 1-alpha)")
    ap.add_argument("--num-rounds", type=int, default=12)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--gen-iters", type=int, default=120)
    ap.add_argument(
        "--gen-checkpoint-interval",
        type=int,
        default=120,
        help=(
            "Save intermediate generator checkpoints at this optimizer-step "
            "interval. Values <= 0 save only at the end of each round."
        ),
    )
    ap.add_argument(
        "--gen-checkpoint-max-keep",
        type=int,
        default=5,
        help="Maximum intermediate generator checkpoints retained per round.",
    )
    ap.add_argument("--trainee-epochs", type=int, default=150)
    ap.add_argument(
        "--require-exact-trainee-budget",
        action="store_true",
        help=(
            "Require the tracker checkpoint epoch and step delta to equal the "
            "requested round budget exactly. Legacy mode accepts overshoot."
        ),
    )
    ap.add_argument(
        "--skip-trainee-update",
        action="store_true",
        help="Update only the generator and keep the initial tracker frozen.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--trainee-ngpu", type=int, default=1)
    ap.add_argument(
        "--trainee-visible-devices",
        default="",
        help=("Comma-separated physical GPU ids for ProtoMotions DDP. "
              "Defaults to the generator --gpu for single-GPU compatibility."),
    )
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument(
        "--release-occupy-before-trainee",
        action="store_true",
        help=("Terminate occupy.py process groups immediately before launching "
              "the all-GPU ProtoMotions trainee. The external watcher is not touched."),
    )
    ap.add_argument(
        "--trainee-overrides",
        default="",
        help=("Extra ProtoMotions --overrides entries, comma-separated. "
              "Example: agent.model.actor_optimizer.lr=2e-6,agent.num_mini_epochs=1"),
    )
    ap.add_argument(
        "--gen-config",
        default="configs/gentrack/train_gentrack_g1.py",
    )
    ap.add_argument(
        "--gen-cfg-options",
        default="",
        help=("Extra MMEngine --cfg-options entries for the generator, comma-separated. "
              "Example: trainer.frontier_t_high=0.98,trainer.num_samples=8"),
    )
    ap.add_argument(
        "--gen-cfg-options-by-round",
        default="",
        help=("Per-round MMEngine cfg-options fragments. Format: "
              "'1:trainer.num_samples=8;2:trainer.num_samples=8,trainer.frontier_t_high=0.98'."),
    )
    ap.add_argument(
        "--gen-init-ckpt",
        default=os.environ.get(
            "MOTIUS_GENTRACK_G0_CHECKPOINT",
            "checkpoints/models/hymotion_g1/g0",
        ),
    )
    ap.add_argument(
        "--gen-reset-each-round",
        action="store_true",
        help=(
            "Initialize every generator round from --gen-init-ckpt. This keeps "
            "the frozen reference policy immutable and prevents semantic drift "
            "from compounding across rounds."
        ),
    )
    ap.add_argument(
        "--trainee-init-ckpt",
        default=os.environ.get(
            "MOTIUS_GENTRACK_TRAINEE_CHECKPOINT",
            str(DEFAULT_PROTO_OUTPUT_ROOT / "gentrack_g1_t0" / "last.ckpt"),
        ),
    )
    ap.add_argument(
        "--initial-judge-onnx",
        default="",
        help=("Required at round 0 for trainee/anchor modes. This must be the "
              "deployment export of --trainee-init-ckpt, trained on the same "
              "public-data protocol; it must not silently fall back to a "
              "released external tracker."),
    )
    ap.add_argument("--trainee-restart-each-round", action="store_true",
                    help="Start every trainee round from --trainee-init-ckpt instead of previous round.")
    ap.add_argument(
        "--persistent-tracker-checkpoint-root",
        default="",
        help=(
            "Optional persistent root for immutable round tracker checkpoints. "
            "When set, runtime last.ckpt is atomically copied to "
            "<root>/rN/last.ckpt and only that verified copy may be exported, "
            "attested, or used by the next round."
        ),
    )
    ap.add_argument("--trainee-snapshot-mode", default="cumulative",
                    choices=[
                        "cumulative",
                        "base-plus-latest",
                        "latest-only",
                        "round-local-replacement",
                    ],
                    help=("Which pool subset to train the tracker on. cumulative copies the full "
                          "pool; base-plus-latest copies r0_snap plus motions added since the "
                          "previous round snapshot, preventing unbounded hard-pool growth; "
                          "latest-only copies only motions added since the previous round snapshot; "
                          "round-local-replacement requires a complete refreshed generator bank "
                          "and admits exactly that round's generated names plus fixed extras."))
    ap.add_argument(
        "--trainee-snapshot-root",
        default="",
        help=(
            "Optional node-local directory for tracker snapshot metadata. "
            "Checkpoints and audit artifacts remain under --root."
        ),
    )
    ap.add_argument(
        "--trainee-extra-motion-dir",
        default="",
        help=("Optional deterministic replay bank of .motion files injected into every "
              "trainee snapshot after the normal snapshot is built."),
    )
    ap.add_argument(
        "--trainee-extra-motion-prefix",
        default="extra_",
        help="Prefix used when copying --trainee-extra-motion-dir files into snapshots.",
    )
    ap.add_argument(
        "--trainee-generated-target-fraction",
        type=float,
        default=0.0,
        help=("Target generated-motion fraction in each tracker snapshot. "
              "Implemented by deterministic source-balanced resampling; 0 keeps "
              "the natural motion-count ratio."),
    )
    ap.add_argument(
        "--trainee-bootstrap-motion-dir",
        default="",
        help=(
            "Optional precomputed train-only generated .motion coverage bank. "
            "It is counted as generated support, separately from public GT."
        ),
    )
    ap.add_argument(
        "--refresh-generator-bank",
        action="store_true",
        help=(
            "After each generator update, regenerate the complete train-only "
            "reference budget from that round's G_t and convert it to "
            "ProtoMotions format. This forbids a frozen G0 bootstrap bank."
        ),
    )
    ap.add_argument("--refresh-generator-count", type=int, default=13337)
    ap.add_argument(
        "--refresh-generator-command",
        default=os.environ.get("MOTIUS_GENTRACK_REFRESH_COMMAND", ""),
        help=(
            "Optional shell command that materializes COVERAGE_READY.json from "
            "the refresh environment exported by this orchestrator. Required "
            "only with --refresh-generator-bank; ordinary online-pool rounds "
            "do not use it."
        ),
    )
    ap.add_argument(
        "--refresh-generator-gpu-list",
        default="",
        help=(
            "Comma-separated GPUs used for the round-local generator refresh. "
            "Defaults to --trainee-visible-devices."
        ),
    )
    ap.add_argument(
        "--refresh-generator-index-root",
        default=(
            "outputs/evaluation/gentrack/"
            "tracker_training/g0_coverage_13337_seed0/coverage_indices"
        ),
    )
    ap.add_argument(
        "--refresh-generator-annotation",
        default=(
            "data/training/hymotion_g1/train.json"
        ),
    )
    ap.add_argument(
        "--refresh-generator-prompt-bank",
        default=(
            "outputs/evaluation/gentrack/"
            "splits/prompts_train_prompt_bank.jsonl"
        ),
    )
    ap.add_argument("--refresh-generator-num-shards", type=int, default=64)
    ap.add_argument("--refresh-generator-batch-size", type=int, default=8)
    ap.add_argument("--refresh-generator-sample-steps", type=int, default=50)
    ap.add_argument("--refresh-convert-workers", type=int, default=32)
    ap.add_argument(
        "--trainee-min-unique-generated",
        type=int,
        default=0,
        help=(
            "Fail before tracker training unless this many distinct generated "
            "motion files are available (online plus bootstrap)."
        ),
    )
    ap.add_argument(
        "--trainee-require-unique-balance",
        action="store_true",
        help=(
            "Disallow repeating generated files to reach the requested source "
            "fraction. A paper-facing coverage gate should always enable this."
        ),
    )
    ap.add_argument(
        "--trainee-exp",
        default=str(
            PROTO
            / "examples"
            / "experiments"
            / "mimic"
            / "gentrack_g1_xy_offset.py"
        ),
    )
    ap.add_argument("--root", default="outputs/training/gentrack_coevolve")
    ap.add_argument("--py310", default=sys.executable)
    ap.add_argument("--py38", default=sys.executable)
    ap.add_argument("--hf-home", default="checkpoints/huggingface")
    args = ap.parse_args()

    if not 0.0 <= args.trainee_generated_target_fraction < 1.0:
        raise ValueError("--trainee-generated-target-fraction must be in [0, 1)")

    # All paths handed to the py38 trainee subprocess run with cwd=PROTO, so they
    # must be ABSOLUTE (resolve anything relative against the repo root).
    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else (ROOT / p)

    args.gen_init_ckpt = str(_abs(args.gen_init_ckpt))
    args.trainee_init_ckpt = str(_abs(args.trainee_init_ckpt))
    initial_judge_onnx = _abs(args.initial_judge_onnx) if args.initial_judge_onnx else None
    args.trainee_exp = str(_abs(args.trainee_exp))
    extra_motion_dir = _abs(args.trainee_extra_motion_dir) if args.trainee_extra_motion_dir else None
    trainee_snapshot_root = (
        _abs(args.trainee_snapshot_root)
        if args.trainee_snapshot_root
        else None
    )
    persistent_tracker_checkpoint_root = (
        _abs(args.persistent_tracker_checkpoint_root)
        if args.persistent_tracker_checkpoint_root
        else None
    )
    bootstrap_motion_dir = (
        _abs(args.trainee_bootstrap_motion_dir)
        if args.trainee_bootstrap_motion_dir
        else None
    )
    refresh_index_root = _abs(args.refresh_generator_index_root)
    refresh_annotation = _abs(args.refresh_generator_annotation)
    refresh_prompt_bank = _abs(args.refresh_generator_prompt_bank)
    if (
        args.trainee_snapshot_mode == "round-local-replacement"
        and not args.refresh_generator_bank
    ):
        raise ValueError(
            "--trainee-snapshot-mode=round-local-replacement requires "
            "--refresh-generator-bank"
        )
    if args.refresh_generator_bank:
        if not args.refresh_generator_command.strip():
            raise ValueError(
                "--refresh-generator-bank requires "
                "--refresh-generator-command (or "
                "MOTIUS_GENTRACK_REFRESH_COMMAND)"
            )
        if bootstrap_motion_dir is not None:
            raise ValueError(
                "--refresh-generator-bank forbids "
                "--trainee-bootstrap-motion-dir"
            )
        if args.refresh_generator_count < 1:
            raise ValueError("--refresh-generator-count must be positive")
        if args.trainee_min_unique_generated != args.refresh_generator_count:
            raise ValueError(
                "--trainee-min-unique-generated must equal "
                "--refresh-generator-count in full refresh mode"
            )
        if args.refresh_convert_workers < 1:
            raise ValueError("--refresh-convert-workers must be positive")
        for path in (
            refresh_index_root,
            refresh_annotation,
            refresh_prompt_bank,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
    round_cfg_options = parse_round_cfg_options(args.gen_cfg_options_by_round)
    root = _abs(args.root)
    arm = root / args.arm_name
    pool = arm / "pool"
    state = arm / "state.jsonl"
    arm.mkdir(parents=True, exist_ok=True)
    pool.mkdir(parents=True, exist_ok=True)

    # gcc toolset for any gymtorch JIT in the py38 trainee subprocess
    gcc_env = {}
    for v in ("14", "13", "12", "11", "10", "9"):
        r = f"/opt/rh/gcc-toolset-{v}/root/usr"
        if os.path.isdir(r + "/bin"):
            gcc_env = {"PATH": f"{r}/bin:" + os.environ.get("PATH", ""),
                       "CC": f"{r}/bin/gcc", "CXX": f"{r}/bin/g++",
                       "LD_LIBRARY_PATH": f"{r}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")}
            break
    py38_ld = os.environ.get("PHYSFLOW_PY38_LD_LIBRARY_PATH", "")
    py38_pythonpath = sanitize_pythonpath_for_py38(
        os.environ.get(
            "PHYSFLOW_PY38_PYTHONPATH", os.environ.get("PYTHONPATH", "")
        )
    )
    py38_extensions = os.environ.get(
        "PHYSFLOW_PY38_TORCH_EXTENSIONS_DIR", ""
    )
    trainee_visible_devices = args.trainee_visible_devices or str(args.gpu)
    visible_device_count = len(
        [device for device in trainee_visible_devices.split(",") if device.strip()]
    )
    refresh_gpu_list = (
        args.refresh_generator_gpu_list or trainee_visible_devices
    )
    if args.trainee_ngpu < 1 or visible_device_count < args.trainee_ngpu:
        raise ValueError(
            f"trainee-ngpu={args.trainee_ngpu} requires at least that many "
            f"visible devices, got {trainee_visible_devices!r}"
        )

    def apply_py38_ld(env):
        if not py38_ld:
            return env
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = py38_ld + ((os.pathsep + current) if current else "")
        return env

    log(state, "orchestrator_start", arm=args.arm_name, mode=args.judge_mode,
        alpha=args.anchor_alpha, rounds=args.num_rounds, gen_iters=args.gen_iters,
        trainee_epochs=args.trainee_epochs, seed=args.seed, gpu=args.gpu,
        trainee_ngpu=args.trainee_ngpu,
        trainee_visible_devices=trainee_visible_devices,
        trainee_restart_each_round=args.trainee_restart_each_round,
        require_exact_trainee_budget=args.require_exact_trainee_budget,
        persistent_tracker_checkpoint_root=(
            str(persistent_tracker_checkpoint_root)
            if persistent_tracker_checkpoint_root
            else None
        ),
        trainee_snapshot_mode=args.trainee_snapshot_mode,
        trainee_snapshot_root=(
            str(trainee_snapshot_root) if trainee_snapshot_root else None
        ),
        gen_reset_each_round=args.gen_reset_each_round,
        trainee_generated_target_fraction=(
            args.trainee_generated_target_fraction),
        trainee_bootstrap_motion_dir=(
            str(bootstrap_motion_dir) if bootstrap_motion_dir else None
        ),
        trainee_min_unique_generated=args.trainee_min_unique_generated,
        trainee_require_unique_balance=args.trainee_require_unique_balance,
        refresh_generator_bank=args.refresh_generator_bank,
        refresh_generator_count=args.refresh_generator_count,
        trainee_extra_motion_dir=str(extra_motion_dir) if extra_motion_dir else None,
        skip_trainee_update=args.skip_trainee_update,
        initial_judge_onnx=str(initial_judge_onnx) if initial_judge_onnx else None,
        gen_cfg_options_by_round=round_cfg_options)

    if (
        args.start_round > 0
        and persistent_tracker_checkpoint_root is not None
        and not args.skip_trainee_update
    ):
        validate_persistent_tracker_boundary(
            arm,
            persistent_tracker_checkpoint_root,
            args.start_round - 1,
            require_exact_budget=args.require_exact_trainee_budget,
        )

    # Resume the two-clock judge state. In the paper-facing lagged mode, Q is
    # one tracker generation behind T: for round r, Q=T_{r-1}, T=T_r. Round 0
    # has only Q=T0, so all Q-valid samples can seed the first tracker update.
    quality_onnx, trainee_onnx = resumed_judge_clock(
        args.judge_mode,
        initial_judge_onnx,
        arm,
        args.start_round,
    )

    for label, path in (("quality", quality_onnx), ("trainee", trainee_onnx)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"initial/resumed {label} judge ONNX not found: {path}")

    for r in range(args.start_round, args.num_rounds):
        round_seed = args.seed + r * 1000
        spec_path = judge_spec_path(arm, r)
        judges = build_judge_spec(
            args.judge_mode,
            args.anchor_alpha,
            quality_onnx,
            trainee_onnx,
            spec_path,
        )
        quality_identity = optional_file_identity(quality_onnx)
        trainee_identity = optional_file_identity(trainee_onnx)
        spec_identity = file_identity(spec_path)
        log(state, "round_start", round=r, judges=[j["name"] for j in judges],
            quality_onnx=str(quality_onnx) if quality_onnx else None,
            trainee_onnx=str(trainee_onnx) if trainee_onnx else None,
            quality_identity=quality_identity,
            trainee_identity=trainee_identity,
            judge_spec_identity=spec_identity)

        # ---------------------------------------------------------- GENERATOR
        gen_work = arm / "gen" / f"r{r}"
        gen_work.mkdir(parents=True, exist_ok=True)
        load_from = select_generator_input_checkpoint(
            arm,
            r,
            Path(args.gen_init_ckpt),
            args.gen_iters,
            state,
            args.gen_reset_each_round,
        )
        generator_input_identity = checkpoint_identity(load_from)
        gen_env = dict(os.environ)
        gen_env.update({
            "HF_HOME": args.hf_home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(ROOT) + ":" + os.environ.get("PYTHONPATH", ""),
            "PHYSFLOW_JUDGE_SPEC": str(spec_path),
            "PHYSFLOW_CONVERT_PYTHON": args.py38,
        })
        checkpoint_interval = (
            args.gen_checkpoint_interval
            if args.gen_checkpoint_interval > 0
            else args.gen_iters
        )
        checkpoint_interval = min(checkpoint_interval, args.gen_iters)
        gen_cmd = [
            args.py310, "tools/train.py", args.gen_config,
            "--work-dir", str(gen_work),
            "--load-from", str(load_from), "--load-scope", "model",
            "--cfg-options",
            f"train_cfg.max_iters={args.gen_iters}",
            f"trainer.tracker_pool_dir={pool}",
            f"default_hooks.checkpoint.interval={checkpoint_interval}",
            f"default_hooks.checkpoint.max_keep_ckpts={args.gen_checkpoint_max_keep}",
            f"seed={round_seed}",
        ]
        gen_opts = []
        if args.gen_cfg_options.strip():
            gen_opts.extend(x.strip() for x in args.gen_cfg_options.split(",") if x.strip())
        gen_opts.extend(round_cfg_options.get(r, []))
        if gen_opts:
            gen_cmd.extend(gen_opts)
            log(state, "gen_cfg_options", round=r, options=gen_opts)
        # Resume guard: skip only a generator round that reached its exact target
        # checkpoint and durably logged gen_done. Intermediate interval
        # checkpoints left by elastic preemption are deliberately fail-closed.
        existing_gen = completed_generator_checkpoint(
            gen_work,
            args.gen_iters,
            state,
            r,
        )
        if existing_gen is not None:
            log(state, "gen_skip", round=r, ckpt=str(existing_gen),
                pool=len(list(pool.glob("*.motion"))),
                generator_input_identity=generator_input_identity)
        else:
            log(state, "gen_launch", round=r, load_from=str(load_from),
                generator_input_identity=generator_input_identity)
            rc = run(gen_cmd, gen_env, gen_work / "gen.log", cwd=str(ROOT))
            if rc != 0:
                log(state, "gen_failed", round=r, rc=rc)
                sys.exit(2)
            pool_n = len(list(pool.glob("*.motion")))
            log(state, "gen_done", round=r, rc=rc, pool=pool_n)

        selected_generator_checkpoint = completed_generator_checkpoint(
            gen_work,
            args.gen_iters,
            state,
            r,
        )
        if selected_generator_checkpoint is None:
            raise FileNotFoundError(
                f"round {r}: no generator checkpoint under {gen_work}"
            )
        generator_output_identity = checkpoint_identity(
            selected_generator_checkpoint
        )
        generator_metadata = validate_generator_checkpoint_metadata(
            selected_generator_checkpoint,
            args.gen_iters,
        )
        log(
            state,
            "generator_checkpoint_attested",
            round=r,
            generator_input=generator_input_identity,
            generator_output=generator_output_identity,
            generator_metadata=generator_metadata,
        )

        refreshed_bank_manifest = None
        refreshed_bank_identity = None
        refreshed_replay_manifest = None
        conversion_identity = None
        conversion_done_path = None
        conversion_manifest_path = None
        generated_motion_root = None
        generated_motion_dir = pool
        if args.refresh_generator_bank:
            refresh_root = arm / "generator_coverage" / f"r{r}"
            ready_path = refresh_root / "COVERAGE_READY.json"
            if not ready_path.is_file():
                refresh_env = dict(os.environ)
                refresh_env.update(
                    {
                        "ROOT": str(ROOT),
                        "CHECKPOINT": str(selected_generator_checkpoint),
                        "CHECKPOINT_METADATA": str(
                            selected_generator_checkpoint
                        ),
                        "OUT_ROOT": str(refresh_root),
                        "INDEX_ROOT": str(refresh_index_root),
                        "ANNO": str(refresh_annotation),
                        "PROMPT_BANK": str(refresh_prompt_bank),
                        "PYTHON": args.py310,
                        "GPU_LIST": refresh_gpu_list,
                        "COUNT": str(args.refresh_generator_count),
                        "MIN_UNIQUE": str(args.refresh_generator_count),
                        "NUM_SHARDS": str(args.refresh_generator_num_shards),
                        "BASE_SEED": str(round_seed),
                        "BATCH_SIZE": str(args.refresh_generator_batch_size),
                        "SAMPLE_STEPS": str(
                            args.refresh_generator_sample_steps
                        ),
                        "ONE_LOAD_PER_GPU": "1",
                    }
                )
                log(
                    state,
                    "generator_bank_refresh_start",
                    round=r,
                    checkpoint=str(selected_generator_checkpoint),
                    expected_count=args.refresh_generator_count,
                    gpu_list=refresh_gpu_list,
                )
                rc = run(
                    shlex.split(args.refresh_generator_command),
                    refresh_env,
                    refresh_root / "generation.log",
                    cwd=str(ROOT),
                )
                if rc != 0:
                    log(state, "generator_bank_refresh_failed", round=r, rc=rc)
                    raise RuntimeError(
                        f"round {r}: generator bank refresh failed with rc={rc}"
                    )
            qpos_dir, refresh_payload = validate_refreshed_generator_bank(
                ready_path,
                selected_generator_checkpoint,
                args.refresh_generator_count,
            )
            proto_motion_dir = refresh_root / "proto_motion_pool"
            conversion_done = proto_motion_dir / "_CONVERSION_DONE.json"
            if not conversion_done.is_file():
                convert_env = dict(os.environ)
                convert_env.update(gcc_env)
                apply_py38_ld(convert_env)
                convert_env.update(
                    {
                        "PYTHONPATH": str(PROTO) + ":" + py38_pythonpath,
                        "MUJOCO_GL": "disable",
                        "CUDA_VISIBLE_DEVICES": "",
                    }
                )
                log(
                    state,
                    "generator_bank_proto_conversion_start",
                    round=r,
                    qpos_dir=str(qpos_dir),
                    expected_count=args.refresh_generator_count,
                    workers=args.refresh_convert_workers,
                )
                rc = run(
                    [
                        args.py38,
                        ROOT
                        / "tools/gentrack/convert_qpos_to_protomotions.py",
                        "--input-dir",
                        qpos_dir,
                        "--output-dir",
                        proto_motion_dir,
                        "--python",
                        args.py38,
                        "--workers",
                        args.refresh_convert_workers,
                        "--expected-count",
                        args.refresh_generator_count,
                    ],
                    convert_env,
                    refresh_root / "proto_conversion.log",
                    cwd=str(ROOT),
                )
                if rc != 0:
                    log(
                        state,
                        "generator_bank_proto_conversion_failed",
                        round=r,
                        rc=rc,
                    )
                    raise RuntimeError(
                        f"round {r}: ProtoMotions conversion failed with rc={rc}"
                    )
            generated_motion_dir, conversion_payload = (
                validate_refreshed_proto_bank(
                    conversion_done,
                    qpos_dir,
                    args.refresh_generator_count,
                )
            )
            refreshed_bank_manifest = str(ready_path.resolve())
            refreshed_bank_identity = file_identity(ready_path)
            refreshed_replay_manifest = str(
                Path(refresh_payload["replay_manifest"]).resolve(strict=True)
            )
            conversion_identity = file_identity(conversion_done)
            conversion_done_path = str(conversion_done.resolve(strict=True))
            conversion_manifest_path = str(
                Path(conversion_payload["manifest"]).resolve(strict=True)
            )
            generated_motion_root = str(
                generated_motion_dir.resolve(strict=True)
            )
            log(
                state,
                "generator_bank_refresh_done",
                round=r,
                checkpoint=str(selected_generator_checkpoint),
                count=int(refresh_payload["count"]),
                ready_manifest=refreshed_bank_manifest,
                conversion_manifest=conversion_payload["manifest"],
                motion_dir=str(generated_motion_dir),
            )

        round_attestation = {
            "schema_version": 1,
            "round": r,
            "seed": round_seed,
            "arm": args.arm_name,
            "judge_mode": args.judge_mode,
            "judge_clock_before_generator": {
                "quality": quality_identity,
                "trainee": trainee_identity,
                "spec": spec_identity,
            },
            "generator": {
                "input": generator_input_identity,
                "output": generator_output_identity,
                "metadata": generator_metadata,
                "requested_updates": args.gen_iters,
                "continued_from_previous_round": bool(
                    r > 0 and not args.gen_reset_each_round
                ),
            },
            "fresh_bank": {
                "enabled": args.refresh_generator_bank,
                "policy": args.trainee_snapshot_mode,
                "expected_generated_count": (
                    args.refresh_generator_count
                    if args.refresh_generator_bank
                    else None
                ),
                "ready": refreshed_bank_identity,
                "ready_manifest_path": refreshed_bank_manifest,
                "replay_manifest_path": refreshed_replay_manifest,
                "conversion": conversion_identity,
                "conversion_done_path": conversion_done_path,
                "conversion_manifest_path": conversion_manifest_path,
                "generated_motion_root": generated_motion_root,
            },
            "run_contract": {
                "path": os.environ.get("RUN_CONTRACT") or None,
                "sha256": os.environ.get("RUN_CONTRACT_SHA256") or None,
                "persistent_tracker_checkpoint_root": (
                    str(persistent_tracker_checkpoint_root)
                    if persistent_tracker_checkpoint_root
                    else None
                ),
                "require_exact_trainee_budget": (
                    args.require_exact_trainee_budget
                ),
            },
            "paper_blockers": [],
        }

        if args.skip_trainee_update:
            round_attestation["tracker"] = {"skipped": True}
            round_attestation["status"] = "complete"
            attestation_path = (
                arm / "round_attestations" / f"r{r}.json"
            )
            attestation_identity = write_immutable_json(
                attestation_path,
                round_attestation,
            )
            log(
                state,
                "trainee_skipped",
                round=r,
                checkpoint=str(args.trainee_init_ckpt),
            )
            log(
                state,
                "round_done",
                round=r,
                refreshed_generator_bank=refreshed_bank_manifest,
                attestation=attestation_identity,
            )
            continue

        # ------------------------------------------------------------ TRAINEE
        snapshot_base = trainee_snapshot_root or (arm / "trainee")
        snap = snapshot_base / f"r{r}_snap"
        if args.refresh_generator_bank:
            online_pool_files = {}
            for motion in generated_motion_dir.rglob("*.motion"):
                relative = motion.relative_to(generated_motion_dir).as_posix()
                digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
                online_pool_files[
                    f"generated_refresh_{digest}_{motion.name}"
                ] = motion
        else:
            online_pool_files = {m.name: m for m in pool.glob("*.motion")}
        bootstrap_files = {}
        if bootstrap_motion_dir is not None:
            for motion in bootstrap_motion_dir.rglob("*.motion"):
                relative = motion.relative_to(bootstrap_motion_dir).as_posix()
                digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
                bootstrap_files[f"generated_bootstrap_{digest}_{motion.name}"] = motion
        pool_files = {**bootstrap_files, **online_pool_files}
        if len(pool_files) < args.trainee_min_unique_generated:
            log(
                state,
                "snapshot_failed_unique_coverage",
                round=r,
                online_unique=len(online_pool_files),
                bootstrap_unique=len(bootstrap_files),
                generated_unique=len(pool_files),
                required=args.trainee_min_unique_generated,
            )
            raise RuntimeError(
                f"round {r}: only {len(pool_files)} unique generated references; "
                f"requires {args.trainee_min_unique_generated}"
            )
        extra_files = {}
        if extra_motion_dir is not None:
            for m in extra_motion_dir.rglob("*.motion"):
                relative = m.relative_to(extra_motion_dir).as_posix()
                digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
                out_name = f"{args.trainee_extra_motion_prefix}{digest}_{m.name}"
                extra_files[out_name] = m
        balanced_files = {}
        target_fraction = args.trainee_generated_target_fraction
        if target_fraction > 0.0 and pool_files and extra_files:
            target_generated = int(
                math.ceil(target_fraction * len(extra_files) / (1.0 - target_fraction))
            )
            duplicate_count = max(0, target_generated - len(pool_files))
            if (
                duplicate_count
                and args.trainee_snapshot_mode == "round-local-replacement"
            ):
                raise RuntimeError(
                    f"round {r}: round-local replacement forbids generated "
                    f"repeats, but source balance needs {duplicate_count}"
                )
            if duplicate_count and args.trainee_require_unique_balance:
                log(
                    state,
                    "snapshot_failed_unique_balance",
                    round=r,
                    generated_unique=len(pool_files),
                    public_unique=len(extra_files),
                    target_generated=target_generated,
                    missing_unique=duplicate_count,
                )
                raise RuntimeError(
                    f"round {r}: source ratio requires {target_generated} unique "
                    f"generated references, found {len(pool_files)}; refusing "
                    "duplicate replay under the unique-balance contract"
                )
            generated_sources = [pool_files[name] for name in sorted(pool_files)]
            for index in range(duplicate_count):
                source = generated_sources[index % len(generated_sources)]
                digest = hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:12]
                out_name = f"generated_balance_{index:06d}_{digest}.motion"
                balanced_files[out_name] = source
        required_names = set(pool_files) | set(extra_files) | set(balanced_files)
        existing_names = (
            {m.name for m in snap.glob("*.motion")} if snap.is_dir() else set()
        )
        snapshot_ready = snapshot_membership_is_ready(
            args.trainee_snapshot_mode,
            required_names,
            existing_names,
        )

        if snapshot_ready:
            log(state, "snapshot_reused", round=r, mode=args.trainee_snapshot_mode,
                motions=len(existing_names), required=len(required_names))
        else:
            if snap.exists():
                shutil.rmtree(snap)
            snap.mkdir(parents=True, exist_ok=True)
            if (
                args.trainee_snapshot_mode
                in ("cumulative", "round-local-replacement")
                or r == 0
            ):
                for name, motion in pool_files.items():
                    if args.refresh_generator_bank:
                        (snap / name).symlink_to(motion.resolve(strict=True))
                    else:
                        shutil.copy2(motion, snap / name)
                log(state, "snapshot_built", round=r, mode=args.trainee_snapshot_mode,
                    motions=len(pool_files))
            elif args.trainee_snapshot_mode in ("base-plus-latest", "latest-only"):
                base_snap = snapshot_base / "r0_snap"
                prev_snap = snapshot_base / f"r{r-1}_snap"
                if (args.trainee_snapshot_mode == "base-plus-latest" and not base_snap.is_dir()) or not prev_snap.is_dir():
                    log(state, "snapshot_failed", round=r, mode=args.trainee_snapshot_mode,
                        base_exists=base_snap.is_dir(), prev_exists=prev_snap.is_dir())
                    sys.exit(5)
                prev_names = {m.name for m in prev_snap.glob("*.motion")}
                copied_names = set()
                base_count = 0
                latest_count = 0
                if args.trainee_snapshot_mode == "base-plus-latest":
                    for m in base_snap.glob("*.motion"):
                        shutil.copy2(m, snap / m.name)
                        copied_names.add(m.name)
                        base_count += 1
                for name in sorted(set(pool_files) - prev_names):
                    if name in copied_names:
                        continue
                    shutil.copy2(pool_files[name], snap / name)
                    copied_names.add(name)
                    latest_count += 1
                log(state, "snapshot_built", round=r, mode=args.trainee_snapshot_mode,
                    base=base_count, latest=latest_count, motions=len(copied_names))
            else:
                raise ValueError(f"bad snapshot mode {args.trainee_snapshot_mode}")

            if extra_motion_dir is not None:
                for name, source in extra_files.items():
                    (snap / name).symlink_to(source)
                extra_count = len(extra_files)
                log(state, "snapshot_extra_injected", round=r,
                    source=str(extra_motion_dir), prefix=args.trainee_extra_motion_prefix,
                    extra=extra_count, motions=len(list(snap.glob("*.motion"))))

            if balanced_files:
                for name, source in balanced_files.items():
                    (snap / name).symlink_to(source.resolve(strict=True))
                generated_count = len(pool_files) + len(balanced_files)
                total_count = generated_count + len(extra_files)
                log(state, "snapshot_generated_balanced", round=r,
                    online_unique=len(online_pool_files),
                    bootstrap_unique=len(bootstrap_files),
                    original_generated=len(pool_files), duplicates=len(balanced_files),
                    generated=generated_count, public=len(extra_files),
                    total=total_count,
                    actual_fraction=generated_count / total_count)
            else:
                total_count = len(pool_files) + len(extra_files)
                log(
                    state,
                    "snapshot_source_counts",
                    round=r,
                    online_unique=len(online_pool_files),
                    bootstrap_unique=len(bootstrap_files),
                    generated_unique=len(pool_files),
                    generated_repeats=0,
                    public_unique=len(extra_files),
                    total=total_count,
                    actual_fraction=(
                        len(pool_files) / total_count if total_count else 0.0
                    ),
                )

        actual_snapshot_names = {m.name for m in snap.glob("*.motion")}
        validate_snapshot_membership(
            args.trainee_snapshot_mode,
            required_names,
            actual_snapshot_names,
        )
        snapshot_manifest_identity = None
        if args.trainee_snapshot_mode == "round-local-replacement":
            if extra_motion_dir is None:
                raise RuntimeError(
                    "round-local replacement requires a fixed public motion root"
                )
            snapshot_manifest_identity = write_round_local_snapshot_manifest(
                arm / "snapshot_manifests" / f"r{r}.json",
                round_index=r,
                snapshot_dir=snap,
                generated_sources=pool_files,
                public_sources=extra_files,
                generated_root=generated_motion_dir,
                public_root=extra_motion_dir,
            )
            log(
                state,
                "snapshot_manifest_attested",
                round=r,
                manifest=snapshot_manifest_identity,
            )
        round_attestation["fresh_bank"].update(
            {
                "generated_unique": len(pool_files),
                "public_unique": len(extra_files),
                "generated_balance_repeats": len(balanced_files),
                "snapshot_count": len(actual_snapshot_names),
                "snapshot_required_count": len(required_names),
                "snapshot_exact": actual_snapshot_names == required_names,
                "snapshot_manifest": snapshot_manifest_identity,
            }
        )

        prev_trainee = select_tracker_input_checkpoint(
            arm,
            r,
            Path(args.trainee_init_ckpt),
            args.trainee_restart_each_round,
            persistent_checkpoint_root=persistent_tracker_checkpoint_root,
            proto_root=PROTO,
            runtime_checkpoint_root=arm / "trainee_runtime",
            require_exact_budget=args.require_exact_trainee_budget,
        )
        tracker_input_identity = file_identity(prev_trainee)
        E, current_steps = ckpt_progress(prev_trainee)
        # ProtoMotions converts training_max_steps to max_epochs once at agent
        # construction, then stops on ``current_epoch < max_epochs``. A resumed
        # checkpoint keeps its epoch even when this round changes num_envs, so
        # the target must be expressed in the *new run's* epoch units. Using the
        # checkpoint's cumulative step_count here can otherwise turn a one-epoch
        # smoke test into hundreds of thousands of epochs.
        target_epoch = E + args.trainee_epochs
        steps_per_epoch = (
            args.num_envs * NUM_STEPS_PER_EPOCH * args.trainee_ngpu
        )
        max_steps = training_max_steps_for_epochs(
            E,
            args.trainee_epochs,
            args.num_envs,
            args.trainee_ngpu,
        )
        added_steps = args.trainee_epochs * steps_per_epoch
        exp = f"{args.arm_name}_co_r{r}"
        tr_env = dict(os.environ)
        tr_env.update(gcc_env)
        apply_py38_ld(tr_env)
        tr_env.update({
            "PYTHONPATH": str(PROTO) + ":" + py38_pythonpath,
            "MOTIUS_PROTOMOTIONS_OUTPUT_ROOT": str(
                arm / "trainee_runtime"
            ),
            "PYTHONPYCACHEPREFIX": os.environ.get(
                "PHYSFLOW_PY38_PYCACHE",
                f"/tmp/physflow_py38_pycache_{os.getuid()}",
            ),
            "ACCEPT_EULA": "Y",
            "CUDA_VISIBLE_DEVICES": trainee_visible_devices,
            # dm_control (imported by pose_lib) inits a GL backend at import; the
            # generator needs MUJOCO_GL=egl for its native-mujoco rollout, but
            # dm_control's pyopengl-EGL path fails headless here. The headless
            # IsaacGym trainee never renders dm_control, so disable its GL.
            "MUJOCO_GL": "disable",
        })
        if py38_extensions:
            tr_env["TORCH_EXTENSIONS_DIR"] = py38_extensions

        motion_file = snap
        if args.trainee_ngpu > 1:
            # A directory-backed MotionLib is fully duplicated by every DDP
            # rank. With the 13k public replay bank this reaches about 1 TB of
            # host memory before the first PPO step. ProtoMotions natively
            # supports rank-local packaged libraries through a filename that
            # contains ``slurmrank``; build those packages once per snapshot.
            package_dir = arm / "trainee" / f"r{r}_pack_n{args.trainee_ngpu}"
            package_pattern = package_dir / "motions_slurmrank.pt"
            package_cmd = [
                args.py38,
                ROOT / "tools" / "gentrack" / "package_protomotions_shards.py",
                "--motion-dir", snap,
                "--output-pattern", package_pattern,
                "--num-shards", str(args.trainee_ngpu),
                "--python", args.py38,
                "--vendor-root", PROTO,
            ]
            log(
                state,
                "trainee_package_start",
                round=r,
                motions=len(list(snap.glob("*.motion"))),
                shards=args.trainee_ngpu,
                pattern=str(package_pattern),
            )
            rc = run(
                package_cmd,
                tr_env,
                arm / "trainee" / f"r{r}_package.log",
                cwd=str(ROOT),
            )
            package_manifest = package_dir / "package_manifest.json"
            if rc != 0 or not package_manifest.is_file():
                log(
                    state,
                    "trainee_package_failed",
                    round=r,
                    rc=rc,
                    manifest_exists=package_manifest.is_file(),
                )
                sys.exit(6)
            package_data = json.loads(package_manifest.read_text())
            if package_data.get("num_motions") != len(list(snap.glob("*.motion"))):
                log(
                    state,
                    "trainee_package_failed",
                    round=r,
                    rc=rc,
                    reason="motion_count_mismatch",
                    package_motions=package_data.get("num_motions"),
                )
                sys.exit(6)
            motion_file = package_pattern
            log(
                state,
                "trainee_package_done",
                round=r,
                motions=package_data["num_motions"],
                shards=package_data["num_shards"],
                pattern=str(package_pattern),
            )
        tr_cmd = [
            args.py38, "protomotions/train_agent.py",
            "--robot-name", "g1", "--simulator", "isaacgym",
            "--experiment-path", args.trainee_exp,
            "--experiment-name", exp,
            "--motion-file", str(motion_file),
            "--checkpoint", str(prev_trainee),
            "--num-envs", str(args.num_envs), "--batch-size", str(args.batch_size),
            "--ngpu", str(args.trainee_ngpu),
            "--seed", str(round_seed),
            "--training-max-steps", str(max_steps),
            "--headless", "True",
            # Held-out evaluation runs separately from saved round checkpoints.
            # Avoid ProtoMotions' one-off warm-start evaluator here: on the
            # 17k-motion mixed replay snapshot it allocates a multi-gigabyte
            # metric tensor after the first PPO epoch and can OOM an otherwise
            # healthy 40-GB training process.
            "--skip-initial-eval",
            "--disable-training-eval",
        ]
        # ProtoMotions defines --overrides with nargs="*": pass the flag once,
        # followed by all key=value entries. Repeating the flag keeps only the
        # final occurrence, while comma-joining entries turns them into one bad
        # key=value token.
        # Co-evolution rounds are short fine-tuning bursts (often 5-20 epochs).
        # Saving every 50 epochs can finish a round without producing last.ckpt,
        # which makes the orchestrator treat a successful run as failed.
        override_entries = ["agent.save_last_checkpoint_every=1"]
        if args.trainee_overrides.strip():
            override_entries.extend(
                x.strip() for x in args.trainee_overrides.split(",") if x.strip()
            )
        if override_entries:
            tr_cmd.append("--overrides")
            tr_cmd.extend(override_entries)
        holder_guard = None
        holder_guard_log = None
        if args.release_occupy_before_trainee:
            released_groups = release_occupy_process_groups()
            log(
                state,
                "occupy_release_before_trainee",
                round=r,
                process_groups=released_groups,
            )
            holder_guard_log = open(
                arm / "trainee" / f"r{r}_occupy_guard.log", "a"
            )
            holder_guard = subprocess.Popen(
                [
                    sys.executable,
                    ROOT / "tools" / "gentrack" / "release_occupy_until_process.py",
                    "--command-kind", "train_agent.py",
                    "--process-substring", exp,
                    "--timeout-seconds", "600",
                    "--post-start-seconds", "120",
                ],
                cwd=str(ROOT),
                stdout=holder_guard_log,
                stderr=subprocess.STDOUT,
            )
            log(state, "occupy_guard_launch", round=r, pid=holder_guard.pid)
        log(state, "trainee_launch", round=r, exp=exp, warm_epoch=E,
            current_steps=current_steps, added_steps=added_steps,
            target_epoch=target_epoch, target_steps=max_steps,
            requested_epochs=args.trainee_epochs,
            motions=len(list(snap.glob("*.motion"))),
            motion_file=str(motion_file),
            tracker_input_identity=tracker_input_identity)
        rc = run(tr_cmd, tr_env, arm / "trainee" / f"r{r}.log", cwd=str(PROTO))
        if holder_guard is not None and holder_guard.poll() is None:
            holder_guard.terminate()
            try:
                holder_guard.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder_guard.kill()
                holder_guard.wait()
        if holder_guard_log is not None:
            holder_guard_log.close()
        runtime_trainee_ckpt = (
            arm / "trainee_runtime" / exp / "last.ckpt"
        )
        if rc != 0 or not runtime_trainee_ckpt.is_file():
            log(
                state,
                "trainee_failed",
                round=r,
                rc=rc,
                ckpt_exists=runtime_trainee_ckpt.is_file(),
            )
            sys.exit(3)
        try:
            completed_epoch, completed_steps = (
                validate_trainee_checkpoint_progress(
                    runtime_trainee_ckpt,
                    target_epoch,
                    current_steps,
                    require_exact=args.require_exact_trainee_budget,
                    expected_added_steps=added_steps,
                )
            )
        except RuntimeError as error:
            log(
                state,
                "trainee_incomplete",
                round=r,
                rc=rc,
                target_epoch=target_epoch,
                previous_step_count=current_steps,
                expected_added_steps=added_steps,
                require_exact=args.require_exact_trainee_budget,
                checkpoint=str(runtime_trainee_ckpt),
                reason=str(error),
            )
            raise

        runtime_tracker_output_identity = file_identity(runtime_trainee_ckpt)
        if persistent_tracker_checkpoint_root is not None:
            trainee_ckpt = persistent_tracker_checkpoint_path(
                persistent_tracker_checkpoint_root,
                r,
            )
            tracker_output_identity = persist_file_atomic_copy(
                runtime_trainee_ckpt,
                trainee_ckpt,
            )
            if (
                tracker_output_identity["sha256"]
                != runtime_tracker_output_identity["sha256"]
                or tracker_output_identity["size_bytes"]
                != runtime_tracker_output_identity["size_bytes"]
            ):
                raise RuntimeError(
                    f"round {r}: persistent tracker checkpoint identity differs "
                    "from its validated runtime source"
                )
            tracker_output_persisted = True
        else:
            trainee_ckpt = runtime_trainee_ckpt
            tracker_output_identity = runtime_tracker_output_identity
            tracker_output_persisted = False
        tracker_output_blockers = ephemeral_checkpoint_blockers(trainee_ckpt)
        log(
            state,
            "trainee_done",
            round=r,
            rc=rc,
            epoch=completed_epoch,
            step_count=completed_steps,
            target_epoch=target_epoch,
            checkpoint=str(trainee_ckpt),
            runtime_checkpoint=str(runtime_trainee_ckpt),
            tracker_output_persisted=tracker_output_persisted,
            require_exact_budget=args.require_exact_trainee_budget,
            tracker_input_identity=tracker_input_identity,
            tracker_output_identity=tracker_output_identity,
            runtime_tracker_output_identity=runtime_tracker_output_identity,
            paper_blockers=tracker_output_blockers,
        )

        # --------------------------------------------------------- JUDGE SYNC
        judge_output_identity = None
        judge_export_manifest_identity = None
        if args.judge_mode != "frozen":
            out = arm / "judge_onnx" / f"r{r}"
            out.mkdir(parents=True, exist_ok=True)
            exp_env = dict(os.environ)
            exp_env.update(gcc_env)
            apply_py38_ld(exp_env)
            exp_env.update({
                "PYTHONPATH": str(PROTO) + ":" + py38_pythonpath,
                "ACCEPT_EULA": "Y",
                "CUDA_VISIBLE_DEVICES": trainee_visible_devices.split(",")[0],
                "MUJOCO_GL": "disable",
            })
            if py38_extensions:
                exp_env["TORCH_EXTENSIONS_DIR"] = py38_extensions
            exp_cmd = [
                args.py38, "deployment/export_bm_tracker_onnx.py",
                "--checkpoint", str(trainee_ckpt), "--output", str(out),
            ]
            log(state, "judge_export", round=r)
            rc = run(exp_cmd, exp_env, arm / "trainee" / f"r{r}_export.log", cwd=str(PROTO))
            onnx = out / "unified_pipeline.onnx"
            yaml = out / "unified_pipeline.yaml"
            if rc != 0 or not onnx.is_file() or not yaml.is_file():
                log(
                    state,
                    "judge_export_failed",
                    round=r,
                    rc=rc,
                    onnx_exists=onnx.is_file(),
                    yaml_exists=yaml.is_file(),
                )
                sys.exit(4)
            export_evidence = write_judge_export_manifest(
                out / "EXPORT_MANIFEST.json",
                round_index=r,
                source_tracker_identity=tracker_output_identity,
                onnx_path=onnx,
                yaml_path=yaml,
            )
            judge_output_identity = export_evidence["onnx"]
            judge_export_manifest_identity = export_evidence["identity"]
            quality_onnx, trainee_onnx = advance_judge_clock(
                args.judge_mode,
                quality_onnx,
                trainee_onnx,
                onnx,
            )
            log(
                state,
                "judge_synced",
                round=r,
                onnx=str(onnx),
                judge_output_identity=judge_output_identity,
                judge_export_manifest_identity=(
                    judge_export_manifest_identity
                ),
                next_quality_identity=optional_file_identity(quality_onnx),
                next_trainee_identity=optional_file_identity(trainee_onnx),
            )

        round_attestation["tracker"] = {
            "skipped": False,
            "input": tracker_input_identity,
            "output": tracker_output_identity,
            "runtime_output": runtime_tracker_output_identity,
            "output_persisted": tracker_output_persisted,
            "input_epoch": E,
            "input_step_count": current_steps,
            "output_epoch": completed_epoch,
            "output_step_count": completed_steps,
            "requested_epochs": args.trainee_epochs,
            "requested_transition_steps": added_steps,
            "actual_transition_steps": completed_steps - current_steps,
            "num_envs": args.num_envs,
            "world_size": args.trainee_ngpu,
            "steps_per_epoch": steps_per_epoch,
            "exact_budget_required": args.require_exact_trainee_budget,
            "exact_budget_satisfied": (
                completed_epoch == target_epoch
                and completed_steps - current_steps == added_steps
            ),
            "judge_output": judge_output_identity,
            "judge_export_manifest": judge_export_manifest_identity,
        }
        round_attestation["paper_blockers"].extend(tracker_output_blockers)
        round_attestation["status"] = "complete"
        attestation_path = arm / "round_attestations" / f"r{r}.json"
        attestation_identity = write_immutable_json(
            attestation_path,
            round_attestation,
        )
        log(
            state,
            "round_done",
            round=r,
            refreshed_generator_bank=refreshed_bank_manifest,
            attestation=attestation_identity,
            paper_blockers=round_attestation["paper_blockers"],
        )

    log(state, "orchestrator_done", rounds=args.num_rounds)


if __name__ == "__main__":
    main()
