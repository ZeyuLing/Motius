"""Stage-by-stage numerical parity artifacts for monocular pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


PARITY_SCHEMA_VERSION = 1


def _require_outputs_path(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if "outputs" not in resolved.parts:
        raise ValueError(
            f"Parity artifacts must be written under outputs/, got {resolved}."
        )


def _as_numpy(value: Any) -> tuple[np.ndarray, str]:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dependency.
        torch = None

    logical_dtype = ""
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        logical_dtype = str(tensor.dtype)
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy(), logical_dtype
        value = tensor.numpy()

    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("Parity traces do not accept object arrays.")
    return np.ascontiguousarray(array), logical_dtype or str(array.dtype)


def _array_sha256(array: np.ndarray, logical_dtype: str) -> str:
    digest = hashlib.sha256()
    digest.update(logical_dtype.encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, tuple[np.ndarray, str]]:
    if isinstance(value, Mapping):
        flattened: dict[str, tuple[np.ndarray, str]] = {}
        for key in sorted(value, key=str):
            key = str(key)
            if not key or "/" in key:
                raise ValueError(f"Parity keys must be non-empty and cannot contain '/': {key!r}")
            child_prefix = f"{prefix}/{key}" if prefix else key
            flattened.update(_flatten(value[key], child_prefix))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, item in enumerate(value):
            child_prefix = f"{prefix}/{index:04d}" if prefix else f"{index:04d}"
            flattened.update(_flatten(item, child_prefix))
        return flattened
    if value is None:
        return {prefix: (np.asarray("__MOTIUS_NONE__", dtype=np.str_), "none")}
    return {prefix: _as_numpy(value)}


@dataclass(frozen=True)
class ParityMismatch:
    """One structural or numerical difference between two traces."""

    stage: str
    field: str
    reason: str
    reference: Optional[object] = None
    candidate: Optional[object] = None


@dataclass(frozen=True)
class ParityReport:
    """Comparison result for two stage traces."""

    reference_name: str
    candidate_name: str
    mismatches: tuple[ParityMismatch, ...]
    compared_fields: int

    @property
    def exact(self) -> bool:
        return not self.mismatches

    def assert_exact(self) -> None:
        if self.exact:
            return
        preview = "; ".join(
            f"{item.stage}/{item.field}: {item.reason}"
            for item in self.mismatches[:10]
        )
        remaining = len(self.mismatches) - 10
        if remaining > 0:
            preview += f"; ... and {remaining} more"
        raise AssertionError(
            f"{self.candidate_name!r} does not exactly match "
            f"{self.reference_name!r}: {preview}"
        )


class MonocularParityTrace:
    """In-memory stage trace with a pickle-free NPZ serialization."""

    def __init__(
        self,
        name: str,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not name:
            raise ValueError("Trace name must be non-empty.")
        self.name = name
        self.metadata = dict(metadata or {})
        self._stages: dict[str, dict[str, tuple[np.ndarray, str]]] = {}

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(self._stages)

    def capture(self, stage: str, values: Mapping[str, Any]) -> None:
        """Capture every leaf under one uniquely named stage."""

        if not stage or "/" in stage:
            raise ValueError("Stage names must be non-empty and cannot contain '/'.")
        if stage in self._stages:
            raise ValueError(f"Stage {stage!r} was already captured.")
        flattened = _flatten(values)
        if not flattened:
            raise ValueError(f"Stage {stage!r} has no values.")
        for field, (array, _logical_dtype) in flattened.items():
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"{stage}/{field} contains non-finite values.")
        self._stages[stage] = flattened

    def save(self, path: str | Path) -> Path:
        """Write the trace below ``outputs/`` as a compressed NPZ artifact."""

        path = Path(path)
        _require_outputs_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        stages = []
        for stage_index, (stage, fields) in enumerate(self._stages.items()):
            field_records = []
            for field_index, (field, (array, logical_dtype)) in enumerate(fields.items()):
                storage_key = f"s{stage_index:04d}_f{field_index:04d}"
                arrays[storage_key] = array
                field_records.append(
                    {
                        "field": field,
                        "storage_key": storage_key,
                        "shape": list(array.shape),
                        "storage_dtype": str(array.dtype),
                        "logical_dtype": logical_dtype,
                        "sha256": _array_sha256(array, logical_dtype),
                    }
                )
            stages.append({"name": stage, "fields": field_records})
        manifest = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "name": self.name,
            "metadata": self.metadata,
            "stages": stages,
        }
        arrays["manifest_json"] = np.asarray(
            json.dumps(manifest, sort_keys=True),
            dtype=np.str_,
        )
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MonocularParityTrace":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest_json"].item()))
            if manifest.get("schema_version") != PARITY_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported parity schema {manifest.get('schema_version')!r}."
                )
            trace = cls(manifest["name"], metadata=manifest.get("metadata", {}))
            for stage_record in manifest["stages"]:
                fields = {}
                for field_record in stage_record["fields"]:
                    array = np.array(
                        archive[field_record["storage_key"]],
                        copy=True,
                        order="C",
                    )
                    logical_dtype = field_record["logical_dtype"]
                    expected_hash = field_record["sha256"]
                    actual_hash = _array_sha256(array, logical_dtype)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"Corrupt parity field "
                            f"{stage_record['name']}/{field_record['field']}."
                        )
                    fields[field_record["field"]] = (array, logical_dtype)
                trace._stages[stage_record["name"]] = fields
        return trace

    def compare(
        self,
        candidate: "MonocularParityTrace",
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
        field_atol: Optional[Mapping[str, float]] = None,
    ) -> ParityReport:
        """Compare a candidate to this reference; defaults to exact equality.

        ``field_atol`` keys use ``"<stage>/<field>"``. This keeps the global
        contract exact while allowing an explicitly audited CUDA operation to
        have its own absolute-error ceiling.
        """

        if rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be non-negative.")
        tolerances = dict(field_atol or {})
        invalid_tolerances = {
            key: value
            for key, value in tolerances.items()
            if not isinstance(key, str) or not key or value < 0
        }
        if invalid_tolerances:
            raise ValueError(
                f"field_atol keys must be non-empty strings and values "
                f"must be non-negative: {invalid_tolerances}"
            )
        mismatches: list[ParityMismatch] = []
        compared_fields = 0
        reference_stages = set(self._stages)
        candidate_stages = set(candidate._stages)
        for stage in sorted(reference_stages - candidate_stages):
            mismatches.append(ParityMismatch(stage, "*", "missing stage"))
        for stage in sorted(candidate_stages - reference_stages):
            mismatches.append(ParityMismatch(stage, "*", "unexpected stage"))

        for stage in sorted(reference_stages & candidate_stages):
            reference_fields = self._stages[stage]
            candidate_fields = candidate._stages[stage]
            for field in sorted(set(reference_fields) - set(candidate_fields)):
                mismatches.append(ParityMismatch(stage, field, "missing field"))
            for field in sorted(set(candidate_fields) - set(reference_fields)):
                mismatches.append(ParityMismatch(stage, field, "unexpected field"))
            for field in sorted(set(reference_fields) & set(candidate_fields)):
                compared_fields += 1
                reference_array, reference_dtype = reference_fields[field]
                candidate_array, candidate_dtype = candidate_fields[field]
                if reference_dtype != candidate_dtype:
                    mismatches.append(
                        ParityMismatch(
                            stage,
                            field,
                            "logical dtype differs",
                            reference_dtype,
                            candidate_dtype,
                        )
                    )
                    continue
                if reference_array.shape != candidate_array.shape:
                    mismatches.append(
                        ParityMismatch(
                            stage,
                            field,
                            "shape differs",
                            list(reference_array.shape),
                            list(candidate_array.shape),
                        )
                    )
                    continue
                local_atol = tolerances.get(f"{stage}/{field}", atol)
                if rtol == 0.0 and local_atol == 0.0:
                    equal = np.array_equal(reference_array, candidate_array)
                elif np.issubdtype(reference_array.dtype, np.number):
                    equal = np.allclose(
                        reference_array,
                        candidate_array,
                        rtol=rtol,
                        atol=local_atol,
                        equal_nan=False,
                    )
                else:
                    equal = np.array_equal(reference_array, candidate_array)
                if not equal:
                    detail = "values differ"
                    if np.issubdtype(reference_array.dtype, np.number):
                        difference = np.abs(
                            reference_array.astype(np.float64)
                            - candidate_array.astype(np.float64)
                        )
                        detail = (
                            f"values differ (max_abs={float(difference.max()):.9g})"
                        )
                    mismatches.append(ParityMismatch(stage, field, detail))

        return ParityReport(
            reference_name=self.name,
            candidate_name=candidate.name,
            mismatches=tuple(mismatches),
            compared_fields=compared_fields,
        )
