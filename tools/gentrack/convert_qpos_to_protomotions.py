#!/usr/bin/env python3
"""Convert the immutable G0 qpos pool with the official ProtoMotions converter.

The wrapper adds the pieces the vendor converter intentionally does not own:

* strict source/output protocol validation;
* deterministic multi-process sharding and resumable logs;
* per-motion SHA-256 and joint-order provenance;
* an atomic completion manifest written only after the full pool validates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motius.models.gentrack.tracker_paths import PROTOMOTIONS_ROOT


DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "outputs/evaluation/gentrack/tracker_training"
    / "g0_coverage_13337_seed0/replay_pool/qpos_npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/evaluation/gentrack/tracker_training"
    / "g0_coverage_13337_seed0/proto_motion_pool"
)
DEFAULT_PROTO_ROOT = PROTOMOTIONS_ROOT
DEFAULT_PROTO_PYTHON = Path(
    os.environ.get(
        "MOTIUS_GENTRACK_PROTO_PYTHON",
        os.environ.get("PHYSFLOW_PROTO_PYTHON", sys.executable),
    )
)
CONVERTER_REL = Path("data/scripts/convert_g1_qpos_npz_to_proto.py")
MJCF_REL = Path("protomotions/data/assets/mjcf/g1_bm_box_feet.xml")

SCHEMA_VERSION = 1
EXPECTED_COUNT = 13_337
INPUT_FPS = 30.0
POLICY_CONTROL_FPS = 50
QPOS_DIM = 36
DOF_DIM = 29
BODY_COUNT = 33


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def add_proto_paths(proto_root: Path) -> None:
    for path in (proto_root, proto_root / "data/scripts"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar, got shape={array.shape}")
    return float(array.reshape(()))


def discover_inputs(
    input_dir: Path, input_manifest: Optional[Path] = None
) -> list[Path]:
    # The immutable G0 replay pool is intentionally flat. Avoid ``rglob`` here:
    # on the large Ceph directory inode, recursive traversal can spend minutes
    # probing nonexistent subtrees before yielding the same direct children.
    if input_manifest is not None:
        payload = json.loads(input_manifest.read_text(encoding="utf-8"))
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError(f"{input_manifest}: missing items list")
        declared = payload.get("count")
        if declared != len(items):
            raise ValueError(
                f"{input_manifest}: count={declared} items={len(items)}"
            )
        files = sorted(Path(str(item["qpos_npz"])).absolute() for item in items)
        outside = [path for path in files if path.parent != input_dir]
        if outside:
            raise ValueError(
                f"{input_manifest}: qpos path is outside input dir: {outside[0]}"
            )
    else:
        files = sorted(path for path in input_dir.glob("*.npz") if path.is_file())
    stems: dict[str, Path] = {}
    for path in files:
        previous = stems.setdefault(path.stem, path)
        if previous != path:
            raise ValueError(
                "official converter flattens output names; duplicate NPZ stem "
                f"{path.stem!r}: {previous} and {path}"
            )
    return files


def validate_source(path: Path, expected_fps: float = INPUT_FPS) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"qpos", "frequency"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        qpos = np.asarray(data["qpos"])
        fps = scalar(data["frequency"])

    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
        raise ValueError(f"{path}: expected qpos [T,{QPOS_DIM}], got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError(f"{path}: expected at least two frames, got {qpos.shape[0]}")
    if not np.issubdtype(qpos.dtype, np.floating):
        raise ValueError(f"{path}: qpos must be floating point, got {qpos.dtype}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{path}: qpos contains non-finite values")
    if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{path}: expected {expected_fps:g} FPS, got {fps:g}")
    quaternion = qpos[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 1e-6):
        raise ValueError(f"{path}: root quaternion contains near-zero norm")
    return {
        "frames": int(qpos.shape[0]),
        "qpos_dim": int(qpos.shape[1]),
        "dtype": str(qpos.dtype),
        "fps": fps,
    }


def extract_joint_order(mjcf_path: Path) -> list[str]:
    root = ET.parse(mjcf_path).getroot()
    names: list[str] = []
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type", "hinge")
        if name and joint_type not in {"free", "ball"}:
            names.append(name)
    if len(names) != DOF_DIM or len(names) != len(set(names)):
        raise ValueError(
            f"{mjcf_path}: expected {DOF_DIM} unique 1-DoF joints, got {len(names)}"
        )
    return names


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected dict payload, got {type(payload).__name__}")
    return payload


def validate_motion(
    path: Path,
    source_frames: int,
    input_fps: float = INPUT_FPS,
    output_fps: int = POLICY_CONTROL_FPS,
) -> dict[str, Any]:
    motion = _torch_load(path)
    required = {
        "fps",
        "dof_pos",
        "dof_vel",
        "rigid_body_pos",
        "rigid_body_rot",
        "rigid_body_contacts",
    }
    missing = required - set(motion)
    if missing:
        raise ValueError(f"{path}: missing motion fields {sorted(missing)}")

    fps = scalar(motion["fps"])
    if not math.isclose(fps, output_fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{path}: expected output FPS {output_fps}, got {fps:g}")

    dof_pos = torch.as_tensor(motion["dof_pos"])
    dof_vel = torch.as_tensor(motion["dof_vel"])
    body_pos = torch.as_tensor(motion["rigid_body_pos"])
    body_rot = torch.as_tensor(motion["rigid_body_rot"])
    contacts = torch.as_tensor(motion["rigid_body_contacts"])
    frames = int(dof_pos.shape[0])
    expected_frames = int(round((source_frames - 1) * output_fps / input_fps)) + 1

    expected_shapes = {
        "dof_pos": (frames, DOF_DIM),
        "dof_vel": (frames, DOF_DIM),
        "rigid_body_pos": (frames, BODY_COUNT, 3),
        "rigid_body_rot": (frames, BODY_COUNT, 4),
        "rigid_body_contacts": (frames, BODY_COUNT),
    }
    tensors = {
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "rigid_body_pos": body_pos,
        "rigid_body_rot": body_rot,
        "rigid_body_contacts": contacts,
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(tensors[name].shape) != expected_shape:
            raise ValueError(
                f"{path}: {name} expected {expected_shape}, got "
                f"{tuple(tensors[name].shape)}"
            )
    if abs(frames - expected_frames) > 1:
        raise ValueError(
            f"{path}: resampled frame count {frames} differs from expected "
            f"{expected_frames} for {source_frames} frames at {input_fps:g}->{output_fps}"
        )
    for name in ("dof_pos", "dof_vel", "rigid_body_pos", "rigid_body_rot"):
        if not torch.isfinite(tensors[name]).all().item():
            raise ValueError(f"{path}: {name} contains non-finite values")
    if contacts.dtype != torch.bool:
        raise ValueError(f"{path}: rigid_body_contacts must be bool, got {contacts.dtype}")
    return {
        "frames": frames,
        "fps": fps,
        "dof_dim": int(dof_pos.shape[1]),
        "body_count": int(body_pos.shape[1]),
    }


def converter_command(
    python: Path,
    converter: Path,
    input_dir: Path,
    output_dir: Path,
    output_fps: int,
    num_ranks: int,
    rank: int,
    force_remake: bool,
    max_files: Optional[int],
    manifest: Optional[Path] = None,
) -> list[str]:
    command = [
        str(python),
        str(converter),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--output-fps",
        str(output_fps),
        "--num-rank",
        str(num_ranks),
        "--slurm-rank",
        str(rank),
    ]
    if force_remake:
        command.append("--force-remake")
    if max_files is not None:
        command.extend(["--max-files", str(max_files)])
    if manifest is not None:
        command.extend(["--manifest", str(manifest)])
    return command


def run_converter_rank(
    rank: int,
    command: list[str],
    proto_root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    log_path = log_dir / f"converter_rank_{rank:03d}.log"
    environment = dict(os.environ)
    python_paths = [str(proto_root), str(proto_root / "data/scripts")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.setdefault("MUJOCO_GL", "disable")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=proto_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(
            f"official converter rank {rank} failed with rc={result.returncode}; "
            f"log={log_path}\n" + "\n".join(lines[-40:])
        )
    return {
        "rank": rank,
        "returncode": result.returncode,
        "command": command,
        "log": str(log_path.resolve()),
    }


def validate_converter_python(python: Path, proto_root: Path) -> None:
    """Fail before the 13k-file preflight when the vendor env is incomplete."""
    environment = dict(os.environ)
    python_paths = [str(proto_root), str(proto_root / "data/scripts")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.setdefault("MUJOCO_GL", "disable")
    result = subprocess.run(
        [str(python), "-c", "import dm_control, torch"],
        cwd=proto_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ProtoMotions converter interpreter is unusable: {python}\n"
            f"{result.stdout[-4000:]}"
        )


def _validated_source_records(
    inputs: Iterable[Path],
    input_dir: Path,
    validate_payloads: bool,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(inputs, start=1):
        metadata = (
            validate_source(path)
            if validate_payloads
            else {"frames": None, "qpos_dim": QPOS_DIM, "dtype": None, "fps": INPUT_FPS}
        )
        records[path.stem] = {
            "path": path,
            "relative_path": path.relative_to(input_dir).as_posix(),
            **metadata,
        }
        if index % 500 == 0:
            print(f"[g0-proto] source preflight {index}", flush=True)
    return records


def finalize_manifest(
    input_dir: Path,
    output_dir: Path,
    selected_inputs: list[Path],
    source_records: dict[str, dict[str, Any]],
    joint_order: list[str],
    converter: Path,
    mjcf: Path,
    converter_runs: list[dict[str, Any]],
    expected_source_count: int,
    output_fps: int,
    manifest_path: Path,
    done_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    outputs = sorted(path for path in output_dir.glob("*.motion") if path.is_file())
    expected_stems = {path.stem for path in selected_inputs}
    actual_stems = {path.stem for path in outputs}
    missing = sorted(expected_stems - actual_stems)
    unexpected = sorted(actual_stems - expected_stems)
    if missing or unexpected:
        raise ValueError(
            f"output inventory mismatch: missing={missing[:5]} ({len(missing)}), "
            f"unexpected={unexpected[:5]} ({len(unexpected)})"
        )

    def finalize_item(source: Path) -> dict[str, Any]:
        source_meta = source_records[source.stem]
        if source_meta["frames"] is None:
            source_meta = {**source_meta, **validate_source(source)}
        output = output_dir / f"{source.stem}.motion"
        motion_meta = validate_motion(
            output,
            source_frames=int(source_meta["frames"]),
            input_fps=float(source_meta["fps"]),
            output_fps=output_fps,
        )
        return {
            "id": source.stem,
            "source_npz": str(source.resolve()),
            "source_relative_path": source_meta["relative_path"],
            "source_sha256": sha256_file(source),
            "source_bytes": source.stat().st_size,
            "source_frames": int(source_meta["frames"]),
            "source_qpos_dim": int(source_meta["qpos_dim"]),
            "source_dtype": source_meta["dtype"],
            "source_fps": float(source_meta["fps"]),
            "motion": str(output.resolve()),
            "motion_relative_path": output.relative_to(output_dir).as_posix(),
            "motion_sha256": sha256_file(output),
            "motion_bytes": output.stat().st_size,
            "motion_frames": int(motion_meta["frames"]),
            "motion_fps": float(motion_meta["fps"]),
            "motion_dof_dim": int(motion_meta["dof_dim"]),
            "motion_body_count": int(motion_meta["body_count"]),
        }

    items: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(
            executor.map(finalize_item, selected_inputs), start=1
        ):
            items.append(item)
            if index % 250 == 0:
                print(
                    f"[g0-proto] finalized {index}/{len(selected_inputs)}",
                    flush=True,
                )

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "g0_qpos30_to_official_protomotions_policy50_v1",
        "created_at_utc": created_at,
        "project_root": str(PROJECT_ROOT.resolve()),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "expected_source_count": expected_source_count,
        "converted_count": len(items),
        "source_contract": {
            "format": "npz",
            "keys": ["qpos", "frequency"],
            "qpos_layout": ["root_xyz", "root_quat_wxyz", "joint_pos"],
            "qpos_dim": QPOS_DIM,
            "input_fps": INPUT_FPS,
            "joint_order": joint_order,
            "joint_order_source": str(mjcf.resolve()),
        },
        "motion_contract": {
            "format": "ProtoMotions .motion torch payload",
            "policy_control_fps": output_fps,
            "dof_dim": DOF_DIM,
            "body_count": BODY_COUNT,
        },
        "provenance": {
            "official_converter": str(converter.resolve()),
            "official_converter_sha256": sha256_file(converter),
            "mjcf": str(mjcf.resolve()),
            "mjcf_sha256": sha256_file(mjcf),
            "converter_runs": converter_runs,
            "taiji_task_flag": os.environ.get("TAIJI_TASK_FLAG"),
            "hostname": os.uname().nodename,
        },
        "items": items,
    }
    atomic_write_json(manifest_path, manifest)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": created_at,
        "count": len(items),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(done_path, completion)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--proto-root", type=Path, default=DEFAULT_PROTO_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PROTO_PYTHON)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--num-ranks",
        type=int,
        help="Global converter shard count. Defaults to --workers for a single-host run.",
    )
    parser.add_argument(
        "--rank-start",
        type=int,
        default=0,
        help="First global converter rank owned by this process.",
    )
    parser.add_argument(
        "--rank-count",
        type=int,
        help="Number of consecutive global ranks owned by this process. Defaults to --workers.",
    )
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--output-fps", type=int, default=POLICY_CONTROL_FPS)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--force-remake", action="store_true")
    parser.add_argument("--skip-conversion", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument(
        "--skip-input-payload-validation",
        action="store_true",
        help="Only check source inventory before conversion; finalization still validates all selected files.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--done-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    num_ranks = args.num_ranks or args.workers
    rank_count = args.rank_count or args.workers
    if num_ranks < 1:
        raise ValueError("--num-ranks must be positive")
    if args.rank_start < 0:
        raise ValueError("--rank-start must be non-negative")
    if rank_count < 1:
        raise ValueError("--rank-count must be positive")
    if args.rank_start + rank_count > num_ranks:
        raise ValueError(
            "owned rank interval exceeds --num-ranks: "
            f"[{args.rank_start}, {args.rank_start + rank_count}) vs {num_ranks}"
        )
    if args.expected_count < 1:
        raise ValueError("--expected-count must be positive")
    if args.output_fps != POLICY_CONTROL_FPS:
        raise ValueError(
            f"this protocol requires policy control rate {POLICY_CONTROL_FPS} FPS"
        )
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")
    if args.max_files is not None and args.workers != 1:
        raise ValueError("--max-files smoke conversion requires --workers 1")

    input_dir = args.input_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    proto_root = args.proto_root.resolve(strict=True)
    converter = proto_root / CONVERTER_REL
    mjcf = proto_root / MJCF_REL
    # Keep the virtual-environment entrypoint intact. Resolving its ``python``
    # symlink to the base interpreter drops the venv's site-packages (including
    # dm_control) even though the requested path itself is valid.
    python = args.python.expanduser().absolute()
    if not python.exists():
        raise FileNotFoundError(python)
    manifest_path = (args.manifest or output_dir / "manifest.json").resolve()
    done_path = (args.done_file or output_dir / "_CONVERSION_DONE.json").resolve()
    if not args.skip_finalize:
        done_path.unlink(missing_ok=True)
    add_proto_paths(proto_root)
    for required in (converter, mjcf):
        if not required.is_file():
            raise FileNotFoundError(required)
    validate_converter_python(python, proto_root)

    input_manifest = (
        args.input_manifest.expanduser().resolve(strict=True)
        if args.input_manifest is not None
        else None
    )
    all_inputs = discover_inputs(input_dir, input_manifest=input_manifest)
    if len(all_inputs) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} source NPZ files, found {len(all_inputs)}"
        )
    selected_inputs = (
        all_inputs[: args.max_files] if args.max_files is not None else all_inputs
    )
    print(
        f"[g0-proto] source_count={len(all_inputs)} selected={len(selected_inputs)} "
        f"input_fps={INPUT_FPS:g} policy_fps={args.output_fps}",
        flush=True,
    )
    source_records = _validated_source_records(
        selected_inputs,
        input_dir,
        validate_payloads=not args.skip_input_payload_validation,
    )
    joint_order = extract_joint_order(mjcf)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    converter_runs: list[dict[str, Any]] = []
    if not args.skip_conversion:
        owned_ranks = range(args.rank_start, args.rank_start + rank_count)
        owned_rank_set = set(owned_ranks)
        rank_inputs: dict[int, list[str]] = {rank: [] for rank in owned_ranks}
        for path in selected_inputs:
            relative = path.relative_to(input_dir)
            file_hash = int(
                hashlib.sha256(str(relative).encode("utf-8")).hexdigest(), 16
            )
            rank = file_hash % num_ranks
            if rank in owned_rank_set:
                rank_inputs[rank].append(relative.as_posix())
        default_rank_manifest_dir = (
            Path("/tmp/physflow_proto_rank_manifests")
            / hashlib.sha1(str(output_dir).encode("utf-8")).hexdigest()[:12]
        )
        rank_manifest_dir = Path(
            os.environ.get(
                "PHYSFLOW_PROTO_RANK_MANIFEST_DIR", str(default_rank_manifest_dir)
            )
        )
        rank_manifest_dir.mkdir(parents=True, exist_ok=True)
        rank_manifests: dict[int, Path] = {}
        for rank, entries in rank_inputs.items():
            rank_manifest = rank_manifest_dir / f"input_rank_{rank:03d}.json"
            atomic_write_json(rank_manifest, entries)
            rank_manifests[rank] = rank_manifest
        commands = [
            converter_command(
                python=python,
                converter=converter,
                input_dir=input_dir,
                output_dir=output_dir,
                output_fps=args.output_fps,
                num_ranks=num_ranks,
                rank=rank,
                force_remake=args.force_remake,
                max_files=args.max_files,
                manifest=rank_manifests[rank],
            )
            for rank in owned_ranks
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_converter_rank, rank, command, proto_root, log_dir
                ): rank
                for rank, command in zip(owned_ranks, commands)
            }
            for future in concurrent.futures.as_completed(futures):
                converter_runs.append(future.result())
        converter_runs.sort(key=lambda item: int(item["rank"]))

    if args.skip_finalize:
        print(
            json.dumps(
                {
                    "status": "conversion_finished_without_manifest",
                    "selected": len(selected_inputs),
                    "output_dir": str(output_dir),
                    "runs": converter_runs,
                },
                sort_keys=True,
            )
        )
        return

    manifest = finalize_manifest(
        input_dir=input_dir,
        output_dir=output_dir,
        selected_inputs=selected_inputs,
        source_records=source_records,
        joint_order=joint_order,
        converter=converter,
        mjcf=mjcf,
        converter_runs=converter_runs,
        expected_source_count=args.expected_count,
        output_fps=args.output_fps,
        manifest_path=manifest_path,
        done_path=done_path,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "count": manifest["converted_count"],
                "manifest": str(manifest_path),
                "done": str(done_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
