#!/usr/bin/env python3
"""Collect and compare immutable GenTrack runtime receipts.

The receipt intentionally separates two kinds of identity:

* ``canonical_fingerprint`` describes the runtime that must be identical
  across paired arms and collection stages.
* ``binding`` and ``volatile`` bind the observation to a contract/segment
  without making host names, task IDs, timestamps, or GPU UUIDs part of the
  comparable runtime fingerprint.

An image tag is evidence, not an OCI digest.  In particular, ``:latest`` is
recorded with ``registry_digest_available=false`` unless the explicit image
reference also contains an ``@sha256:<64 hex>`` digest.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


RECEIPT_SCHEMA = "gentrack.taiji-runtime-receipt.v1"
FINGERPRINT_SCHEMA = "gentrack.taiji-runtime-fingerprint.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"@sha256:([0-9a-fA-F]{64})$")

STABLE_ENV_KEYS = (
    "ACCEPT_EULA",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_CACHE_PATH",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "ISAACSIM_ACCEPT_EULA",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
    "OMNI_KIT_ACCEPT_EULA",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
)
STABLE_ENV_PREFIXES = ("NCCL_",)
VOLATILE_ENV_PREFIXES = (
    "JOB_",
    "KUBERNETES_",
    "POD_",
    "SLURM_",
    "TAIJI_",
    "TASK_",
)
VOLATILE_ENV_KEYS = (
    "HOSTNAME",
    "HOST",
    "INSTANCE_ID",
    "JOB_ID",
    "POD_NAME",
    "TASK_ID",
    "TASK_NAME",
)
SECRET_MARKERS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
SECRET_ARG_NAMES = {
    "api-key",
    "apikey",
    "auth-token",
    "password",
    "secret",
    "token",
}
VOLATILE_ARG_NAMES = {
    "captured-at",
    "captured_at",
    "gpu-uuid",
    "gpu-uuids",
    "gpu_uuid",
    "gpu_uuids",
    "host",
    "hostname",
    "instance-id",
    "instance_id",
    "job-id",
    "job_id",
    "pod-name",
    "pod_name",
    "segment-id",
    "segment_id",
    "task-flag",
    "task_flag",
    "task-id",
    "task_id",
    "task-name",
    "task_name",
    "timestamp",
}
CRITICAL_LIBRARY_PATTERNS = (
    "ld-linux",
    "libc.so",
    "libcuda",
    "libcudart",
    "libcudnn",
    "libegl",
    "libgcc",
    "libgl",
    "libgomp",
    "libnvidia",
    "libnvrtc",
    "libpython",
    "libstdc++",
    "libtorch",
    "libvulkan",
)

PYTHON_PROBE = r"""
import json
import os
import platform
import sys

result = {
    "base_prefix": sys.base_prefix,
    "implementation": platform.python_implementation(),
    "prefix": sys.prefix,
    "python_version": platform.python_version(),
    "sys_executable": sys.executable,
}
try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    result["torch"] = {
        "available": True,
        "cuda_available": cuda_available,
        "cuda_device_count": (
            int(torch.cuda.device_count()) if cuda_available else 0
        ),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "native_extension": getattr(torch._C, "__file__", None),
        "version": torch.__version__,
    }
    loaded = set()
    try:
        with open("/proc/self/maps", "r", encoding="utf-8") as handle:
            for line in handle:
                candidate = line.rstrip().split(None, 5)[-1]
                if candidate.startswith("/") and os.path.isfile(candidate):
                    loaded.add(candidate)
    except OSError:
        pass
    result["torch"]["loaded_native_libraries"] = sorted(loaded)
except Exception as exc:
    result["torch"] = {
        "available": False,
        "error_type": type(exc).__name__,
    }
print(json.dumps(result, sort_keys=True))
""".strip()


class RuntimeReceiptError(RuntimeError):
    """Raised when runtime evidence cannot be collected safely."""


class ReceiptValidationError(RuntimeReceiptError):
    """Raised when a receipt fails its self-hash or schema checks."""


class ReceiptMismatchError(RuntimeReceiptError):
    """Raised when receipts do not share one canonical fingerprint."""


Runner = Callable[..., Any]
NowFn = Callable[[], Union[str, dt.datetime]]
ReceiptInput = Union[Mapping[str, Any], str, os.PathLike]


@dataclass(frozen=True)
class CollectorConfig:
    """Explicit inputs required to bind a runtime observation."""

    output: Path
    image_ref: str
    py310_executable: Path
    py38_executable: Path
    contract_sha256: str
    segment_id: str
    orchestrator_argv: Sequence[str]
    package_sources: Mapping[str, Path]
    py310_wrapper: Optional[Path] = None
    native_libraries: Sequence[Path] = field(default_factory=tuple)
    collection_stage: str = "node-preflight"


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single byte representation used by every receipt hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def rendered_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path_value: Union[str, os.PathLike]) -> str:
    path = Path(path_value)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path(path_value: Union[str, os.PathLike]) -> Path:
    # Do not resolve symlinks: Ceph mount aliases are operationally meaningful
    # and resolving one can make otherwise identical receipts host-dependent.
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))


def file_identity(path_value: Union[str, os.PathLike]) -> Dict[str, Any]:
    path = _absolute_path(path_value)
    if not path.is_file():
        raise RuntimeReceiptError("required file is missing: {}".format(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def source_identity(
    label: str,
    path_value: Union[str, os.PathLike],
) -> Dict[str, Any]:
    """Hash an allow-listed file or a deterministic directory tree."""

    path = _absolute_path(path_value)
    if not path.exists() and not path.is_symlink():
        raise RuntimeReceiptError(
            "package source {!r} is missing: {}".format(label, path)
        )
    if path.is_file():
        record = file_identity(path)
        record.update(
            {
                "file_count": 1,
                "kind": "file",
                "label": label,
                "symlink_count": 0,
                "tree_size_bytes": record["size_bytes"],
            }
        )
        return record
    if path.is_symlink():
        target = os.readlink(str(path))
        digest = sha256_bytes(
            canonical_json_bytes(
                [{"path": ".", "target": target, "type": "symlink"}]
            )
        )
        return {
            "file_count": 0,
            "kind": "symlink",
            "label": label,
            "path": str(path),
            "sha256": digest,
            "symlink_count": 1,
            "target": target,
            "tree_size_bytes": 0,
        }
    if not path.is_dir():
        raise RuntimeReceiptError(
            "package source {!r} is not a file or directory: {}".format(
                label,
                path,
            )
        )

    entries: List[Dict[str, Any]] = []
    file_count = 0
    symlink_count = 0
    tree_size_bytes = 0
    for current, dir_names, file_names in os.walk(
        str(path),
        topdown=True,
        followlinks=False,
    ):
        dir_names.sort()
        file_names.sort()
        current_path = Path(current)
        relative_current = current_path.relative_to(path)

        retained_dirs: List[str] = []
        for name in dir_names:
            child = current_path / name
            relative = (relative_current / name).as_posix()
            if child.is_symlink():
                entries.append(
                    {
                        "path": relative,
                        "target": os.readlink(str(child)),
                        "type": "symlink",
                    }
                )
                symlink_count += 1
            else:
                entries.append({"path": relative, "type": "directory"})
                retained_dirs.append(name)
        dir_names[:] = retained_dirs

        for name in file_names:
            child = current_path / name
            relative = (relative_current / name).as_posix()
            if child.is_symlink():
                entries.append(
                    {
                        "path": relative,
                        "target": os.readlink(str(child)),
                        "type": "symlink",
                    }
                )
                symlink_count += 1
                continue
            if not child.is_file():
                raise RuntimeReceiptError(
                    "unsupported package-source entry: {}".format(child)
                )
            size = child.stat().st_size
            entries.append(
                {
                    "path": relative,
                    "sha256": sha256_file(child),
                    "size_bytes": size,
                    "type": "file",
                }
            )
            file_count += 1
            tree_size_bytes += size

    entries.sort(key=lambda item: (item["path"], item["type"]))
    return {
        "file_count": file_count,
        "kind": "directory",
        "label": label,
        "path": str(path),
        "sha256": sha256_bytes(canonical_json_bytes(entries)),
        "symlink_count": symlink_count,
        "tree_size_bytes": tree_size_bytes,
    }


def _run_text(
    command: Sequence[str],
    *,
    runner: Runner,
    environ: Mapping[str, str],
    timeout: int = 60,
) -> str:
    try:
        completed = runner(
            list(command),
            check=True,
            env=dict(environ),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeReceiptError(
            "command failed: {} ({})".format(
                " ".join(map(str, command)),
                type(exc).__name__,
            )
        ) from exc
    return str(getattr(completed, "stdout", ""))


def _last_json_object(output: str, purpose: str) -> Dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeReceiptError(
        "{} did not emit a JSON object".format(purpose)
    )


def _same_file(left: Union[str, os.PathLike], right: Union[str, os.PathLike]) -> bool:
    try:
        return os.path.samefile(os.fspath(left), os.fspath(right))
    except (FileNotFoundError, OSError):
        return os.path.normcase(str(_absolute_path(left))) == os.path.normcase(
            str(_absolute_path(right))
        )


def collect_python_runtime(
    *,
    name: str,
    executable: Path,
    wrapper: Optional[Path],
    runner: Runner,
    environ: Mapping[str, str],
) -> Dict[str, Any]:
    executable_record = file_identity(executable)
    wrapper_record = file_identity(wrapper) if wrapper is not None else None
    launcher = wrapper if wrapper is not None else executable
    output = _run_text(
        [str(_absolute_path(launcher)), "-c", PYTHON_PROBE],
        runner=runner,
        environ=environ,
    )
    probe = _last_json_object(output, "{} probe".format(name))
    reported = probe.get("sys_executable")
    if not isinstance(reported, str) or not reported:
        raise RuntimeReceiptError(
            "{} probe omitted sys.executable".format(name)
        )
    if not _same_file(reported, executable):
        raise RuntimeReceiptError(
            "{} wrapper executed unexpected Python: {!r}, expected {!r}".format(
                name,
                reported,
                str(_absolute_path(executable)),
            )
        )
    torch_record = probe.get("torch")
    if not isinstance(torch_record, dict):
        raise RuntimeReceiptError(
            "{} probe omitted torch evidence".format(name)
        )
    loaded_paths = torch_record.get("loaded_native_libraries")
    if loaded_paths is not None:
        if not isinstance(loaded_paths, list) or not all(
            isinstance(value, str) for value in loaded_paths
        ):
            raise RuntimeReceiptError(
                "{} probe emitted invalid loaded native libraries".format(
                    name
                )
            )
        torch_record["loaded_native_libraries"] = sorted(
            {
                value
                for value in loaded_paths
                if _is_critical_library(Path(value).name)
            }
        )
    return {
        "executable": executable_record,
        "name": name,
        "probe": probe,
        "wrapper": wrapper_record,
    }


def _is_critical_library(name: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in CRITICAL_LIBRARY_PATTERNS)


def _parse_ldd(output: str) -> List[Tuple[str, Optional[str]]]:
    parsed: List[Tuple[str, Optional[str]]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=>" in line:
            soname, remainder = line.split("=>", 1)
            soname = soname.strip()
            target = remainder.strip().split(" ", 1)[0]
            parsed.append(
                (soname, None if target == "not" else target)
            )
            continue
        first = line.split(" ", 1)[0]
        if first.startswith("/"):
            parsed.append((Path(first).name, first))
    return parsed


def collect_native_libraries(
    *,
    runtimes: Mapping[str, Mapping[str, Any]],
    explicit_paths: Sequence[Path],
    runner: Runner,
    environ: Mapping[str, str],
) -> Dict[str, Any]:
    sources: List[Tuple[str, Path]] = []
    libraries_by_path: Dict[str, Dict[str, Any]] = {}
    for runtime_name in sorted(runtimes):
        runtime = runtimes[runtime_name]
        sources.append(
            (
                "{}:python".format(runtime_name),
                Path(runtime["executable"]["path"]),
            )
        )
        torch_record = runtime["probe"]["torch"]
        extension = torch_record.get("native_extension")
        if torch_record.get("available") is True and extension:
            extension_path = _absolute_path(extension)
            if not extension_path.is_file():
                raise RuntimeReceiptError(
                    "{} torch native extension is missing: {}".format(
                        runtime_name,
                        extension_path,
                    )
                )
            sources.append(
                ("{}:torch._C".format(runtime_name), extension_path)
            )
            extension_record = file_identity(extension_path)
            extension_record.update(
                {
                    "sonames": [extension_path.name],
                    "sources": [
                        "{}:torch._C:self".format(runtime_name)
                    ],
                }
            )
            libraries_by_path[str(extension_path)] = extension_record
        loaded_paths = torch_record.get("loaded_native_libraries", [])
        if loaded_paths is None:
            loaded_paths = []
        if not isinstance(loaded_paths, list) or not all(
            isinstance(value, str) for value in loaded_paths
        ):
            raise RuntimeReceiptError(
                "{} probe emitted invalid loaded native libraries".format(
                    runtime_name
                )
            )
        for loaded_value in loaded_paths:
            loaded_path = _absolute_path(loaded_value)
            if not _is_critical_library(loaded_path.name):
                continue
            if not loaded_path.is_file():
                raise RuntimeReceiptError(
                    "{} loaded native library is missing: {}".format(
                        runtime_name,
                        loaded_path,
                    )
                )
            loaded_record = libraries_by_path.get(str(loaded_path))
            if loaded_record is None:
                loaded_record = file_identity(loaded_path)
                loaded_record.update(
                    {
                        "sonames": [loaded_path.name],
                        "sources": [],
                    }
                )
                libraries_by_path[str(loaded_path)] = loaded_record
            source = "{}:loaded".format(runtime_name)
            if source not in loaded_record["sources"]:
                loaded_record["sources"].append(source)

    missing: List[Dict[str, str]] = []
    for source_name, binary in sources:
        output = _run_text(
            ["ldd", str(binary)],
            runner=runner,
            environ=environ,
        )
        for soname, target in _parse_ldd(output):
            if not _is_critical_library(soname):
                continue
            if target is None:
                missing.append({"soname": soname, "source": source_name})
                continue
            if not target.startswith("/"):
                continue
            target_path = _absolute_path(target)
            record = libraries_by_path.get(str(target_path))
            if record is None:
                record = file_identity(target_path)
                record.update({"sonames": [], "sources": []})
                libraries_by_path[str(target_path)] = record
            if soname not in record["sonames"]:
                record["sonames"].append(soname)
            if source_name not in record["sources"]:
                record["sources"].append(source_name)

    for explicit in explicit_paths:
        target_path = _absolute_path(explicit)
        record = libraries_by_path.get(str(target_path))
        if record is None:
            record = file_identity(target_path)
            record.update({"sonames": [], "sources": []})
            libraries_by_path[str(target_path)] = record
        soname = target_path.name
        if soname not in record["sonames"]:
            record["sonames"].append(soname)
        if "explicit" not in record["sources"]:
            record["sources"].append("explicit")

    libraries = list(libraries_by_path.values())
    for record in libraries:
        record["sonames"].sort()
        record["sources"].sort()
    libraries.sort(key=lambda item: item["path"])
    missing.sort(key=lambda item: (item["soname"], item["source"]))
    return {"libraries": libraries, "missing": missing}


def collect_gpu_runtime(
    *,
    runner: Runner,
    environ: Mapping[str, str],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,uuid",
            "--format=csv,noheader",
        ],
        runner=runner,
        environ=environ,
    )
    observed: List[Dict[str, str]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            raise RuntimeReceiptError(
                "unexpected nvidia-smi row: {!r}".format(row)
            )
        observed.append(
            {
                "driver_version": row[2].strip(),
                "index": row[0].strip(),
                "model": row[1].strip(),
                "uuid": row[3].strip(),
            }
        )
    observed.sort(key=lambda item: (item["index"], item["model"]))
    model_counts: Dict[str, int] = {}
    for gpu in observed:
        model_counts[gpu["model"]] = model_counts.get(gpu["model"], 0) + 1
    canonical = {
        "driver_versions": sorted(
            {gpu["driver_version"] for gpu in observed}
        ),
        "gpu_count": len(observed),
        "models": [
            {"count": model_counts[name], "name": name}
            for name in sorted(model_counts)
        ],
    }
    return canonical, observed


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def collect_environment(
    environ: Mapping[str, str],
) -> Tuple[Dict[str, Optional[str]], Dict[str, str]]:
    stable: Dict[str, Optional[str]] = {
        key: environ.get(key) for key in STABLE_ENV_KEYS
    }
    for key in sorted(environ):
        if _is_secret_key(key):
            continue
        if any(key.startswith(prefix) for prefix in STABLE_ENV_PREFIXES):
            stable[key] = environ[key]

    volatile: Dict[str, str] = {}
    for key in sorted(environ):
        if _is_secret_key(key):
            continue
        if key in VOLATILE_ENV_KEYS or any(
            key.startswith(prefix) for prefix in VOLATILE_ENV_PREFIXES
        ):
            volatile[key] = environ[key]
    return stable, volatile


def _option_name(argument: str) -> Optional[str]:
    if not argument.startswith("--"):
        return None
    return argument[2:].split("=", 1)[0].lower()


def sanitize_argv(arguments: Sequence[str]) -> List[str]:
    sanitized: List[str] = []
    index = 0
    while index < len(arguments):
        argument = str(arguments[index])
        name = _option_name(argument)
        if name in SECRET_ARG_NAMES or (
            name is not None and _is_secret_key(name)
        ):
            if "=" in argument:
                sanitized.append("--{}=<redacted>".format(name))
            else:
                sanitized.append(argument)
                if index + 1 < len(arguments):
                    sanitized.append("<redacted>")
                    index += 1
            index += 1
            continue
        if "=" in argument:
            key, value = argument.split("=", 1)
            if _is_secret_key(key):
                sanitized.append("{}=<redacted>".format(key))
                index += 1
                continue
            if _is_secret_key(value.split("=", 1)[0]):
                sanitized.append("{}=<redacted>".format(key))
                index += 1
                continue
        if "=" in argument and _is_secret_key(argument.split("=", 1)[0]):
            sanitized.append(
                "{}=<redacted>".format(argument.split("=", 1)[0])
            )
        else:
            sanitized.append(argument)
        index += 1
    return sanitized


def normalize_orchestrator_argv(
    arguments: Sequence[str],
    *,
    volatile_values: Iterable[str],
) -> List[str]:
    sanitized = sanitize_argv(arguments)
    replacements = sorted(
        {
            value
            for value in volatile_values
            if isinstance(value, str) and len(value) >= 6
        },
        key=len,
        reverse=True,
    )
    normalized: List[str] = []
    index = 0
    while index < len(sanitized):
        argument = sanitized[index]
        name = _option_name(argument)
        if name in VOLATILE_ARG_NAMES:
            normalized.append("--{}=<volatile>".format(name))
            if (
                "=" not in argument
                and index + 1 < len(sanitized)
                and not sanitized[index + 1].startswith("--")
            ):
                index += 1
            index += 1
            continue
        for value in replacements:
            argument = argument.replace(value, "<volatile>")
        normalized.append(argument)
        index += 1
    return normalized


def image_identity(image_ref: str) -> Dict[str, Any]:
    if not image_ref.strip():
        raise RuntimeReceiptError("image_ref must not be empty")
    match = OCI_DIGEST_RE.search(image_ref)
    digest = (
        "sha256:{}".format(match.group(1).lower()) if match else None
    )
    return {
        "image_ref": image_ref,
        "registry_digest": digest,
        "registry_digest_available": digest is not None,
    }


def _timestamp(now_fn: NowFn) -> str:
    value = now_fn()
    if isinstance(value, str):
        if not value:
            raise RuntimeReceiptError("now_fn returned an empty timestamp")
        return value
    if not isinstance(value, dt.datetime):
        raise RuntimeReceiptError(
            "now_fn must return datetime or str, got {}".format(
                type(value).__name__
            )
        )
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


def compute_fingerprint_sha256(receipt: Mapping[str, Any]) -> str:
    fingerprint = receipt.get("canonical_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ReceiptValidationError(
            "receipt has no canonical_fingerprint object"
        )
    return sha256_bytes(canonical_json_bytes(fingerprint))


def compute_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(receipt))
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        raise ReceiptValidationError("receipt has no integrity object")
    integrity.pop("receipt_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def build_runtime_receipt(
    config: CollectorConfig,
    *,
    runner: Runner = subprocess.run,
    environ: Optional[Mapping[str, str]] = None,
    hostname_fn: Callable[[], str] = socket.gethostname,
    now_fn: NowFn = lambda: dt.datetime.now(dt.timezone.utc),
) -> Dict[str, Any]:
    contract_sha256 = config.contract_sha256.lower()
    if not SHA256_RE.fullmatch(contract_sha256):
        raise RuntimeReceiptError(
            "contract_sha256 must be exactly 64 hexadecimal characters"
        )
    if not config.segment_id:
        raise RuntimeReceiptError("segment_id must not be empty")
    if not config.package_sources:
        raise RuntimeReceiptError(
            "at least one allow-listed package source is required"
        )
    environment = dict(os.environ if environ is None else environ)

    runtimes = {
        "py310": collect_python_runtime(
            name="py310",
            executable=config.py310_executable,
            wrapper=config.py310_wrapper,
            runner=runner,
            environ=environment,
        ),
        "py38": collect_python_runtime(
            name="py38",
            executable=config.py38_executable,
            wrapper=None,
            runner=runner,
            environ=environment,
        ),
    }
    native_libraries = collect_native_libraries(
        runtimes=runtimes,
        explicit_paths=config.native_libraries,
        runner=runner,
        environ=environment,
    )
    package_sources = [
        source_identity(label, config.package_sources[label])
        for label in sorted(config.package_sources)
    ]
    gpu_canonical, gpu_observed = collect_gpu_runtime(
        runner=runner,
        environ=environment,
    )
    stable_environment, volatile_environment = collect_environment(
        environment
    )
    hostname = str(hostname_fn())
    captured_at = _timestamp(now_fn)
    volatile_values = [
        config.segment_id,
        hostname,
        captured_at,
    ]
    volatile_values.extend(volatile_environment.values())
    volatile_values.extend(gpu["uuid"] for gpu in gpu_observed)
    argv_observed = sanitize_argv(config.orchestrator_argv)
    argv_canonical = normalize_orchestrator_argv(
        config.orchestrator_argv,
        volatile_values=volatile_values,
    )

    canonical_fingerprint = {
        "accelerator": gpu_canonical,
        "collector": file_identity(__file__),
        "environment": stable_environment,
        "image": image_identity(config.image_ref),
        "native_libraries": native_libraries,
        "orchestrator_argv": argv_canonical,
        "package_sources": package_sources,
        "runtimes": runtimes,
        "schema": FINGERPRINT_SCHEMA,
    }
    receipt: Dict[str, Any] = {
        "binding": {
            "contract_sha256": contract_sha256,
        },
        "canonical_fingerprint": canonical_fingerprint,
        "fingerprint_sha256": sha256_bytes(
            canonical_json_bytes(canonical_fingerprint)
        ),
        "integrity": {
            "hash_algorithm": "sha256",
        },
        "schema": RECEIPT_SCHEMA,
        "volatile": {
            "captured_at_utc": captured_at,
            "collection_stage": config.collection_stage,
            "environment": volatile_environment,
            "gpus": gpu_observed,
            "hostname": hostname,
            "orchestrator_argv_observed": argv_observed,
            "segment_id": config.segment_id,
        },
    }
    receipt["integrity"]["receipt_sha256"] = compute_receipt_sha256(
        receipt
    )
    return receipt


def write_immutable_receipt(
    output_value: Union[str, os.PathLike],
    receipt: Mapping[str, Any],
) -> bool:
    """Write once without clobbering.

    Returns ``True`` when a new file is installed and ``False`` when the exact
    same bytes already exist.  A different existing receipt is always refused.
    """

    output = _absolute_path(output_value)
    encoded = rendered_json_bytes(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() == encoded:
            return False
        raise FileExistsError(
            "refusing to overwrite different runtime receipt: {}".format(
                output
            )
        )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(output.parent),
        prefix=".{}.".format(output.name),
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o444)
        try:
            os.link(str(temporary), str(output))
        except FileExistsError:
            if output.read_bytes() == encoded:
                return False
            raise FileExistsError(
                "refusing to overwrite different runtime receipt: {}".format(
                    output
                )
            )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return True


def collect_runtime_receipt(
    config: CollectorConfig,
    *,
    runner: Runner = subprocess.run,
    environ: Optional[Mapping[str, str]] = None,
    hostname_fn: Callable[[], str] = socket.gethostname,
    now_fn: NowFn = lambda: dt.datetime.now(dt.timezone.utc),
) -> Dict[str, Any]:
    receipt = build_runtime_receipt(
        config,
        runner=runner,
        environ=environ,
        hostname_fn=hostname_fn,
        now_fn=now_fn,
    )
    write_immutable_receipt(config.output, receipt)
    return receipt


def load_receipt(value: ReceiptInput) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = _absolute_path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptValidationError(
            "cannot load receipt {}: {}".format(path, type(exc).__name__)
        ) from exc
    if not isinstance(payload, dict):
        raise ReceiptValidationError(
            "receipt {} is not a JSON object".format(path)
        )
    return payload


def verify_receipt(value: ReceiptInput) -> Dict[str, Any]:
    receipt = load_receipt(value)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ReceiptValidationError(
            "unexpected receipt schema: {!r}".format(receipt.get("schema"))
        )
    fingerprint = receipt.get("fingerprint_sha256")
    expected_fingerprint = compute_fingerprint_sha256(receipt)
    if fingerprint != expected_fingerprint:
        raise ReceiptValidationError(
            "fingerprint rehash mismatch: recorded={!r}, actual={!r}".format(
                fingerprint,
                expected_fingerprint,
            )
        )
    integrity = receipt.get("integrity")
    recorded_receipt = (
        integrity.get("receipt_sha256")
        if isinstance(integrity, dict)
        else None
    )
    expected_receipt = compute_receipt_sha256(receipt)
    if recorded_receipt != expected_receipt:
        raise ReceiptValidationError(
            "receipt rehash mismatch: recorded={!r}, actual={!r}".format(
                recorded_receipt,
                expected_receipt,
            )
        )
    return {
        "fingerprint_sha256": expected_fingerprint,
        "receipt_sha256": expected_receipt,
        "valid": True,
    }


def compare_receipts(values: Sequence[ReceiptInput]) -> Dict[str, Any]:
    if len(values) < 2:
        raise ReceiptMismatchError(
            "compare requires at least two runtime receipts"
        )
    fingerprints: List[str] = []
    receipt_hashes: List[str] = []
    for value in values:
        report = verify_receipt(value)
        fingerprints.append(report["fingerprint_sha256"])
        receipt_hashes.append(report["receipt_sha256"])
    unique = sorted(set(fingerprints))
    if len(unique) != 1:
        raise ReceiptMismatchError(
            "canonical runtime fingerprint mismatch: {}".format(
                ", ".join(unique)
            )
        )
    return {
        "fingerprint_sha256": unique[0],
        "receipt_count": len(values),
        "receipt_sha256s": receipt_hashes,
        "valid": True,
    }


def _parse_package_sources(values: Sequence[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeReceiptError(
                "--package-source must use LABEL=PATH: {!r}".format(value)
            )
        label, path_value = value.split("=", 1)
        if not label or not path_value:
            raise RuntimeReceiptError(
                "--package-source must use non-empty LABEL=PATH"
            )
        if label in parsed:
            raise RuntimeReceiptError(
                "duplicate package-source label: {}".format(label)
            )
        parsed[label] = Path(path_value)
    return parsed


def _parse_argv_json(value: str) -> List[str]:
    raw = value
    if value.startswith("@"):
        path = _absolute_path(value[1:])
        raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeReceiptError(
            "--orchestrator-argv-json must be a JSON string list or @file"
        ) from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise RuntimeReceiptError(
            "--orchestrator-argv-json must decode to a list of strings"
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="collect and immutably write one node runtime receipt",
    )
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--image-ref", required=True)
    collect.add_argument("--py310-executable", required=True, type=Path)
    collect.add_argument("--py310-wrapper", type=Path)
    collect.add_argument("--py38-executable", required=True, type=Path)
    collect.add_argument("--contract-sha256", required=True)
    collect.add_argument("--segment-id", required=True)
    collect.add_argument(
        "--collection-stage",
        default="node-preflight",
        help="volatile label such as container-start or segment-start",
    )
    collect.add_argument(
        "--orchestrator-argv-json",
        required=True,
        help="JSON list of strings, or @PATH to a JSON file",
    )
    collect.add_argument(
        "--package-source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="allow-listed source file/tree to hash; repeat as needed",
    )
    collect.add_argument(
        "--native-lib",
        action="append",
        default=[],
        type=Path,
        help="additional critical native library to hash",
    )

    verify = subparsers.add_parser(
        "verify",
        help="rehash and verify one receipt",
    )
    verify.add_argument("receipt", type=Path)

    compare = subparsers.add_parser(
        "compare",
        help="verify receipts and require identical canonical fingerprints",
    )
    compare.add_argument("receipts", nargs="+", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Support the requested ``--compare`` spelling in addition to the clearer
    # subcommand, and keep direct collect flags backwards-friendly.
    if arguments and arguments[0] == "--compare":
        arguments = ["compare"] + arguments[1:]
    elif arguments and arguments[0] == "--verify":
        arguments = ["verify"] + arguments[1:]
    elif arguments and arguments[0] not in {"collect", "compare", "verify"}:
        arguments = ["collect"] + arguments

    parser = _build_parser()
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "verify":
            print(
                json.dumps(
                    verify_receipt(parsed.receipt),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if parsed.command == "compare":
            print(
                json.dumps(
                    compare_receipts(parsed.receipts),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        config = CollectorConfig(
            output=parsed.output,
            image_ref=parsed.image_ref,
            py310_executable=parsed.py310_executable,
            py310_wrapper=parsed.py310_wrapper,
            py38_executable=parsed.py38_executable,
            contract_sha256=parsed.contract_sha256,
            segment_id=parsed.segment_id,
            collection_stage=parsed.collection_stage,
            orchestrator_argv=_parse_argv_json(
                parsed.orchestrator_argv_json
            ),
            package_sources=_parse_package_sources(
                parsed.package_source
            ),
            native_libraries=tuple(parsed.native_lib),
        )
        receipt = collect_runtime_receipt(config)
        print(
            json.dumps(
                {
                    "fingerprint_sha256": receipt["fingerprint_sha256"],
                    "output": str(_absolute_path(config.output)),
                    "receipt_sha256": receipt["integrity"][
                        "receipt_sha256"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        FileExistsError,
        ReceiptMismatchError,
        ReceiptValidationError,
        RuntimeReceiptError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
