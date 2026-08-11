#!/usr/bin/env python3
"""Verify motion-tracking checkpoints, rollouts, trainers, and viewer provenance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class MethodSpec:
    artifact_required: bool
    trainer: str
    output_shapes: Mapping[str, tuple[int, ...]]


METHODS = {
    "any2track": MethodSpec(
        artifact_required=True,
        trainer="not-released-upstream",
        output_shapes={"continuous_actions": (1, 29)},
    ),
    "protomotions": MethodSpec(
        artifact_required=True,
        trainer="required",
        output_shapes={
            "actions": (1, 29),
            "joint_pos_targets": (1, 29),
            "stiffness_targets": (1, 29),
            "damping_targets": (1, 29),
        },
    ),
    "humanoid_gpt": MethodSpec(
        artifact_required=True,
        trainer="not-released-upstream",
        output_shapes={
            "continuous_actions": (1, 29),
            "motor_targets": (1, 29),
        },
    ),
    "sonic": MethodSpec(
        artifact_required=True,
        trainer="required",
        output_shapes={"action": (1, 29)},
    ),
    "beyondmimic": MethodSpec(
        artifact_required=False,
        trainer="required",
        output_shapes={"actions": (1, 29)},
    ),
}

TRAINER_STAGES = (
    "parameter_update",
    "checkpoint_save",
    "resume",
    "export",
    "pipeline_reload",
    "physical_rollout",
)

REPLAY_METRIC_RELATIVE_TOLERANCES = {
    # Foot slip is gated by a hard 5 cm contact-height threshold. A one-ULP
    # trajectory divergence can therefore change which frames enter the mean.
    "foot_slip_m_s": 0.20,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _version_tuple(value: str) -> tuple[int, ...]:
    result = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        result.append(int(digits))
    return tuple(result)


def verify_runtime() -> dict[str, Any]:
    import onnxruntime

    version = str(onnxruntime.__version__)
    errors = []
    if sys.version_info < (3, 10):
        errors.append("motion tracking requires Python 3.10 or newer")
    if _version_tuple(version) < (1, 18):
        errors.append(
            f"onnxruntime {version} cannot load all released IR-10 policies; "
            "install onnxruntime>=1.18"
        )
    return {
        "status": "pass" if not errors else "fail",
        "python": sys.version.split()[0],
        "onnxruntime": version,
        "errors": errors,
    }


def verify_inventory(artifact: Path) -> dict[str, Any]:
    inventory_path = artifact / "artifact_inventory.json"
    errors = []
    checked = 0
    if not inventory_path.is_file():
        return {
            "status": "fail",
            "checked": 0,
            "errors": ["missing artifact_inventory.json"],
        }
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for relative, metadata in inventory.get("files", {}).items():
        path = artifact / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        checked += 1
        if path.stat().st_size != int(metadata["bytes"]):
            errors.append(f"size mismatch: {relative}")
        if _sha256(path) != metadata["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")
    return {
        "status": "pass" if not errors else "fail",
        "checked": checked,
        "errors": errors,
    }


def _identity_quaternion(array: np.ndarray) -> None:
    if array.shape[-1] == 4:
        array[..., 3] = 1.0


def _protomotions_inputs(pipeline: Any) -> dict[str, np.ndarray]:
    session = pipeline.bundle._sessions["policy"].session
    inputs = {}
    for value in session.get_inputs():
        shape = [
            dimension if isinstance(dimension, int) and dimension > 0 else 1
            for dimension in value.shape
        ]
        array = np.zeros(shape, dtype=np.float32)
        if "rot" in value.name:
            _identity_quaternion(array)
        inputs[value.name] = array
    return inputs


def _forward(pipeline: Any, method: str) -> Mapping[str, Any]:
    if method == "any2track":
        return pipeline.infer_motion_tracking(
            np.zeros((1, 156), dtype=np.float32)
        )
    if method == "protomotions":
        return pipeline.infer_motion_tracking(_protomotions_inputs(pipeline))
    if method == "humanoid_gpt":
        return pipeline.infer_motion_tracking(
            np.zeros((1, 136), dtype=np.float32)
        )
    if method == "sonic":
        return pipeline.infer_motion_tracking(
            np.zeros((1, pipeline.bundle.encoder_dim), dtype=np.float32),
            np.zeros((1, pipeline.bundle.decoder_state_dim), dtype=np.float32),
        )
    if method == "beyondmimic":
        return pipeline.infer_motion_tracking(
            np.zeros((1, 154), dtype=np.float32),
            time_step=0,
        )
    raise KeyError(method)


def verify_artifact(method: str, artifact: Path) -> dict[str, Any]:
    from motius import Pipeline

    errors = []
    inventory = verify_inventory(artifact)
    try:
        pipeline = Pipeline.from_pretrained(artifact)
        output = _forward(pipeline, method)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        output = {}
    tensors = {}
    for name, shape in METHODS[method].output_shapes.items():
        if name not in output:
            errors.append(f"missing output {name}")
            continue
        value = np.asarray(output[name])
        tensors[name] = {
            "shape": list(value.shape),
            "finite": bool(np.isfinite(value).all()),
        }
        if tuple(value.shape) != shape:
            errors.append(f"{name} shape is {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            errors.append(f"{name} contains non-finite values")
    if inventory["status"] != "pass":
        errors.append("artifact inventory failed")
    return {
        "status": "pass" if not errors else "fail",
        "artifact": str(artifact.resolve()),
        "inventory": inventory,
        "outputs": tensors,
        "errors": errors,
    }


def _resolved_rollout(case: Mapping[str, Any], result_path: Path) -> Path:
    declared = Path(str(case["artifact"]))
    if declared.is_file():
        return declared
    candidate = result_path.parent / "rollouts" / declared.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(declared)


def verify_rollouts(method: str, result_path: Path) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    errors = []
    cases = payload.get("cases", [])
    digest = hashlib.sha256()
    case_ids = []
    for case in cases:
        case_ids.append(str(case["id"]))
        try:
            path = _resolved_rollout(case, result_path)
            with np.load(path, allow_pickle=False) as archive:
                qpos = np.asarray(archive["qpos"])
                reference = np.asarray(archive["reference_qpos"])
                actions = np.asarray(archive["actions"])
                if qpos.ndim != 2 or qpos.shape[1] != 36:
                    errors.append(f"{case['id']}: qpos shape {qpos.shape}")
                if reference.shape != qpos.shape:
                    errors.append(f"{case['id']}: reference shape {reference.shape}")
                if actions.ndim != 2 or actions.shape[1] != 29:
                    errors.append(f"{case['id']}: action shape {actions.shape}")
                for name, value in (
                    ("qpos", qpos),
                    ("reference_qpos", reference),
                    ("actions", actions),
                ):
                    if not np.isfinite(value).all():
                        errors.append(f"{case['id']}: non-finite {name}")
            digest.update(path.name.encode("utf-8"))
            digest.update(bytes.fromhex(_sha256(path)))
        except Exception as error:
            errors.append(f"{case.get('id')}: {type(error).__name__}: {error}")
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate rollout case ids")
    if _method_key(payload.get("method", "")) != _method_key(method):
        errors.append(f"method mismatch: {payload.get('method')}")
    return {
        "status": "pass" if not errors else "fail",
        "result": str(result_path.resolve()),
        "protocol_id": payload.get("protocol_id"),
        "cases": len(cases),
        "case_ids": case_ids,
        "rollout_digest": digest.hexdigest(),
        "errors": errors,
    }


def _archive_text(archive: Any, name: str) -> str:
    return str(np.asarray(archive[name]).reshape(-1)[0])


def verify_replay(
    method: str,
    replay_path: Path,
    baseline_path: Path,
    *,
    first_step_atol: float = 1e-6,
    metric_relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare one fresh physical replay with its published baseline.

    Closed-loop physics can amplify a one-ULP difference after hundreds of
    steps. We therefore require exact protocol/reference/initial-state parity,
    near-exact first-step policy output, and bounded final metric drift instead
    of requiring the full trajectories to remain bitwise identical.
    """

    errors = []
    arrays = {}
    metric_drift = {}
    first_difference = None
    with (
        np.load(replay_path, allow_pickle=False) as replay,
        np.load(baseline_path, allow_pickle=False) as baseline,
    ):
        for name in ("method", "backend", "protocol_id"):
            actual = _archive_text(replay, name)
            expected = _archive_text(baseline, name)
            if actual != expected:
                errors.append(f"{name} mismatch: {actual!r} != {expected!r}")
        archived_method = _archive_text(replay, "method")
        if _method_key(archived_method) != _method_key(method):
            errors.append(f"method mismatch: {archived_method!r}")

        for name in ("qpos", "reference_qpos", "actions"):
            actual = np.asarray(replay[name])
            expected = np.asarray(baseline[name])
            if actual.shape != expected.shape:
                errors.append(
                    f"{name} shape mismatch: {actual.shape} != {expected.shape}"
                )
                continue
            difference = np.abs(actual - expected)
            arrays[name] = {
                "shape": list(actual.shape),
                "max_abs": float(difference.max(initial=0.0)),
                "bitwise_equal": bool(np.array_equal(actual, expected)),
            }
            if name == "reference_qpos" and not np.array_equal(actual, expected):
                errors.append("reference_qpos is not bitwise identical")
            if name == "qpos" and not np.array_equal(actual[0], expected[0]):
                errors.append("initial qpos is not bitwise identical")
            if name == "actions" and actual.shape[0]:
                first_step_max = float(difference[0].max(initial=0.0))
                arrays[name]["first_step_max_abs"] = first_step_max
                if first_step_max > first_step_atol:
                    errors.append(
                        f"first action drift {first_step_max:.3g} exceeds "
                        f"{first_step_atol:.3g}"
                    )
                per_step = difference.reshape(difference.shape[0], -1).max(axis=1)
                changed = np.flatnonzero(per_step > 0)
                if changed.size:
                    index = int(changed[0])
                    first_difference = {
                        "step": index,
                        "max_abs": float(per_step[index]),
                    }

        replay_metrics = json.loads(_archive_text(replay, "metrics_json"))
        baseline_metrics = json.loads(_archive_text(baseline, "metrics_json"))
        metric_names = sorted(set(replay_metrics) & set(baseline_metrics))
        for name in metric_names:
            actual = replay_metrics[name]
            expected = baseline_metrics[name]
            if not isinstance(actual, (int, float)) or not isinstance(
                expected, (int, float)
            ):
                continue
            absolute = abs(float(actual) - float(expected))
            relative = absolute / max(abs(float(expected)), 1e-12)
            metric_drift[name] = {
                "replay": float(actual),
                "baseline": float(expected),
                "relative": relative,
            }
            if name in {"success_rate", "survival_rate"}:
                if absolute > 1e-12:
                    errors.append(f"{name} changed: {actual} != {expected}")
            else:
                tolerance = REPLAY_METRIC_RELATIVE_TOLERANCES.get(
                    name,
                    metric_relative_tolerance,
                )
                if relative <= tolerance:
                    continue
                errors.append(
                    f"{name} drift {relative:.3%} exceeds "
                    f"{tolerance:.3%}"
                )

    return {
        "status": "pass" if not errors else "fail",
        "replay": str(replay_path.resolve()),
        "baseline": str(baseline_path.resolve()),
        "arrays": arrays,
        "first_bitwise_action_difference": first_difference,
        "metric_drift": metric_drift,
        "errors": errors,
    }


def verify_viewer(
    manifest_path: Path,
    rollouts: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    cases = manifest.get("cases", [])
    manifest_ids = {str(case["case_id"]).split("__f", 1)[0] for case in cases}
    columns = {column["key"] for column in manifest.get("columns", [])}
    coverage = {}
    for method, report in rollouts.items():
        if method not in columns:
            errors.append(f"viewer has no {method} column")
            continue
        expected = {case_id.split("__f", 1)[0] for case_id in report["case_ids"]}
        if expected != manifest_ids:
            errors.append(
                f"{method}: viewer/result case mismatch "
                f"({len(manifest_ids)} vs {len(expected)})"
            )
        available = sum(method in case.get("assets", {}) for case in cases)
        coverage[method] = available
        if available != len(cases):
            errors.append(f"{method}: only {available}/{len(cases)} viewer assets")
    return {
        "status": "pass" if not errors else "fail",
        "manifest": str(manifest_path.resolve()),
        "cases": len(cases),
        "coverage": coverage,
        "errors": errors,
    }


def verify_trainer_evidence(method: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("method") != method:
        errors.append(f"method mismatch: {payload.get('method')}")
    stages = payload.get("stages", {})
    for stage in TRAINER_STAGES:
        result = stages.get(stage)
        if not isinstance(result, Mapping):
            errors.append(f"missing stage {stage}")
        elif result.get("status") != "pass":
            errors.append(f"{stage}: {result.get('status', 'missing')}")
    update = stages.get("parameter_update", {})
    if (
        update.get("before_sha256")
        and update.get("after_sha256")
        and update["before_sha256"] == update["after_sha256"]
    ):
        errors.append("parameter_update did not change checkpoint state")
    return {
        "status": "pass" if not errors else "fail",
        "evidence": str(path.resolve()),
        "stages": stages,
        "errors": errors,
    }


def _assignments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected METHOD=PATH, got {value!r}")
        if key not in METHODS:
            raise ValueError(f"Unknown method {key!r}")
        result[key] = Path(path).expanduser()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--rollouts", action="append", default=[], metavar="METHOD=RESULTS")
    parser.add_argument("--replay", action="append", default=[], metavar="METHOD=NPZ")
    parser.add_argument(
        "--baseline-rollout",
        action="append",
        default=[],
        metavar="METHOD=NPZ",
    )
    parser.add_argument("--trainer-evidence", action="append", default=[], metavar="METHOD=JSON")
    parser.add_argument("--viewer-manifest", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every applicable artifact and trainer gate is supplied and passes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/motion_tracking/reproduction.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_paths = _assignments(args.artifact)
    rollout_paths = _assignments(args.rollouts)
    replay_paths = _assignments(args.replay)
    baseline_paths = _assignments(args.baseline_rollout)
    evidence_paths = _assignments(args.trainer_evidence)
    runtime = verify_runtime()
    artifacts = {
        method: verify_artifact(method, path)
        for method, path in artifact_paths.items()
    }
    rollouts = {
        method: verify_rollouts(method, path)
        for method, path in rollout_paths.items()
    }
    replay_methods = sorted(set(replay_paths) | set(baseline_paths))
    replays = {}
    for method in replay_methods:
        if method not in replay_paths or method not in baseline_paths:
            replays[method] = {
                "status": "fail",
                "errors": ["both --replay and --baseline-rollout are required"],
            }
        else:
            replays[method] = verify_replay(
                method,
                replay_paths[method],
                baseline_paths[method],
            )
    trainer_evidence = {
        method: verify_trainer_evidence(method, path)
        for method, path in evidence_paths.items()
    }
    viewer = (
        verify_viewer(args.viewer_manifest, rollouts)
        if args.viewer_manifest is not None
        else None
    )

    errors = []
    for section_name, section in (
        ("artifact", artifacts),
        ("rollout", rollouts),
        ("replay", replays),
        ("trainer", trainer_evidence),
    ):
        for method, report in section.items():
            if report["status"] != "pass":
                errors.append(f"{section_name}:{method}")
    if runtime["status"] != "pass":
        errors.append("runtime")
    if viewer is not None and viewer["status"] != "pass":
        errors.append("viewer")
    if args.require_complete:
        for method, spec in METHODS.items():
            if spec.artifact_required and method not in artifacts:
                errors.append(f"artifact:{method}:missing")
            if spec.trainer == "required" and method not in trainer_evidence:
                errors.append(f"trainer:{method}:missing")

    payload = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "runtime": runtime,
        "artifacts": artifacts,
        "rollouts": rollouts,
        "replays": replays,
        "trainer_evidence": trainer_evidence,
        "viewer": viewer,
        "errors": errors,
    }
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "errors": errors}, indent=2))
    print(f"Saved {output}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
