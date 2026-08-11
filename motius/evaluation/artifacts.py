"""Canonical paths and metadata checks for Motius evaluation artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_SLUG = re.compile(r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?$")


def _validate_slug(name: str, value: str) -> str:
    value = str(value)
    if not _SLUG.fullmatch(value):
        raise ValueError(
            f"{name} must contain only lowercase letters, digits, '_' or '-': {value!r}"
        )
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class EvaluationArtifactLayout:
    """Address one immutable benchmark protocol under ``outputs/evaluation``."""

    task_id: str
    benchmark_id: str
    protocol_id: str
    root: Path = Path("outputs/evaluation")

    def __post_init__(self) -> None:
        _validate_slug("task_id", self.task_id)
        _validate_slug("benchmark_id", self.benchmark_id)
        _validate_slug("protocol_id", self.protocol_id)

    @property
    def protocol_root(self) -> Path:
        return self.root / self.task_id / self.benchmark_id / self.protocol_id

    @property
    def protocol_dir(self) -> Path:
        return self.protocol_root / "protocol"

    @property
    def protocol_metadata(self) -> Path:
        return self.protocol_dir / "protocol.json"

    @property
    def protocol_manifest(self) -> Path:
        return self.protocol_dir / "manifest.jsonl"

    @property
    def references_dir(self) -> Path:
        return self.protocol_dir / "references"

    @property
    def runs_dir(self) -> Path:
        return self.protocol_root / "runs"

    @property
    def leaderboard_dir(self) -> Path:
        return self.protocol_root / "leaderboard"

    @property
    def leaderboard_results(self) -> Path:
        return self.leaderboard_dir / "results.json"

    def run_root(self, method_id: str, run_id: str) -> Path:
        return (
            self.runs_dir
            / _validate_slug("method_id", method_id)
            / _validate_slug("run_id", run_id)
        )

    def run_metadata(self, method_id: str, run_id: str) -> Path:
        return self.run_root(method_id, run_id) / "run.json"

    def predictions_dir(
        self, method_id: str, run_id: str, representation: Optional[str] = None
    ) -> Path:
        path = self.run_root(method_id, run_id) / "predictions"
        return path / _validate_slug("representation", representation) if representation else path

    def metrics_dir(self, method_id: str, run_id: str) -> Path:
        return self.run_root(method_id, run_id) / "metrics"

    def visualization_dir(self, method_id: str, run_id: str) -> Path:
        return self.run_root(method_id, run_id) / "visualization"

    def logs_dir(self, method_id: str, run_id: str) -> Path:
        return self.run_root(method_id, run_id) / "logs"

    def init_protocol(self, metadata: Optional[Mapping[str, Any]] = None) -> Path:
        value: Dict[str, Any] = {
            "schema_version": 1,
            "task_id": self.task_id,
            "benchmark_id": self.benchmark_id,
            "protocol_id": self.protocol_id,
        }
        if metadata:
            value.update(metadata)
        _write_json(self.protocol_metadata, value)
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_dir.mkdir(parents=True, exist_ok=True)
        return self.protocol_root

    def init_run(
        self,
        method_id: str,
        run_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        method_id = _validate_slug("method_id", method_id)
        run_id = _validate_slug("run_id", run_id)
        value: Dict[str, Any] = {
            "schema_version": 1,
            "task_id": self.task_id,
            "benchmark_id": self.benchmark_id,
            "protocol_id": self.protocol_id,
            "method_id": method_id,
            "run_id": run_id,
        }
        if metadata:
            value.update(metadata)
        _write_json(self.run_metadata(method_id, run_id), value)
        self.predictions_dir(method_id, run_id).mkdir(parents=True, exist_ok=True)
        self.metrics_dir(method_id, run_id).mkdir(parents=True, exist_ok=True)
        self.visualization_dir(method_id, run_id).mkdir(parents=True, exist_ok=True)
        self.logs_dir(method_id, run_id).mkdir(parents=True, exist_ok=True)
        return self.run_root(method_id, run_id)

    def validate(self, require_manifest: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.protocol_metadata.is_file():
            errors.append(f"missing {self.protocol_metadata}")
        else:
            metadata = _read_json(self.protocol_metadata)
            errors.extend(
                _identity_errors(
                    self.protocol_metadata,
                    metadata,
                    {
                        "task_id": self.task_id,
                        "benchmark_id": self.benchmark_id,
                        "protocol_id": self.protocol_id,
                    },
                )
            )
        if require_manifest and not self.protocol_manifest.is_file():
            errors.append(f"missing {self.protocol_manifest}")

        if self.runs_dir.is_dir():
            for run_metadata in sorted(self.runs_dir.glob("*/*/run.json")):
                relative = run_metadata.relative_to(self.runs_dir)
                method_id, run_id = relative.parts[:2]
                metadata = _read_json(run_metadata)
                errors.extend(
                    _identity_errors(
                        run_metadata,
                        metadata,
                        {
                            "task_id": self.task_id,
                            "benchmark_id": self.benchmark_id,
                            "protocol_id": self.protocol_id,
                            "method_id": method_id,
                            "run_id": run_id,
                        },
                    )
                )
                run_root = run_metadata.parent
                for required in ("predictions", "metrics", "visualization", "logs"):
                    if not (run_root / required).is_dir():
                        errors.append(f"missing {run_root / required}")
        return errors


def _identity_errors(
    path: Path,
    metadata: Mapping[str, Any],
    expected: Mapping[str, str],
) -> Iterable[str]:
    for key, value in expected.items():
        if metadata.get(key) != value:
            yield f"{path}: {key}={metadata.get(key)!r}, expected {value!r}"
