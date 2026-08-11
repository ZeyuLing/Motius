from pathlib import Path

import numpy as np
import pytest
import torch

from motius.utils.monocular_parity import MonocularParityTrace


def _trace(name: str) -> MonocularParityTrace:
    trace = MonocularParityTrace(name, metadata={"method": "synthetic"})
    trace.capture(
        "01_preprocess",
        {
            "bbox_xyxy": np.arange(8, dtype=np.float32).reshape(2, 4),
            "camera": {
                "K": torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
                "is_static": True,
            },
        },
    )
    trace.capture(
        "02_decode",
        {
            "joints": np.zeros((2, 22, 3), dtype=np.float32),
            "body_model": "SMPL-H",
            "optional_camera": None,
        },
    )
    return trace


def test_parity_trace_round_trip_and_exact_match(tmp_path: Path):
    reference = _trace("official")
    artifact = reference.save(tmp_path / "outputs" / "official.npz")
    restored = MonocularParityTrace.load(artifact)

    report = restored.compare(_trace("motius"))

    assert report.exact
    assert report.compared_fields == 6
    report.assert_exact()


def test_parity_trace_rejects_one_value_difference():
    reference = _trace("official")
    candidate = _trace("motius")
    candidate._stages["02_decode"]["joints"][0][0, 0, 0] = np.nextafter(
        np.float32(0.0),
        np.float32(1.0),
    )

    report = reference.compare(candidate)

    assert not report.exact
    assert report.mismatches[0].stage == "02_decode"
    assert report.mismatches[0].field == "joints"
    with pytest.raises(AssertionError, match="max_abs"):
        report.assert_exact()


def test_parity_trace_supports_field_scoped_cuda_tolerance():
    reference = _trace("official")
    candidate = _trace("motius")
    candidate._stages["02_decode"]["joints"][0][0, 0, 0] = np.float32(2e-6)

    accepted = reference.compare(
        candidate,
        field_atol={"02_decode/joints": 3e-6},
    )
    rejected_other_field = reference.compare(
        candidate,
        field_atol={"01_preprocess/bbox_xyxy": 3e-6},
    )

    assert accepted.exact
    assert not rejected_other_field.exact


def test_parity_trace_rejects_invalid_field_tolerance():
    with pytest.raises(ValueError, match="field_atol"):
        _trace("official").compare(_trace("motius"), field_atol={"decode": -1.0})


def test_parity_trace_rejects_missing_stage_and_dtype_change():
    reference = _trace("official")
    candidate = MonocularParityTrace("motius")
    candidate.capture(
        "01_preprocess",
        {
            "bbox_xyxy": np.arange(8, dtype=np.float64).reshape(2, 4),
            "camera": {
                "K": torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
                "is_static": True,
            },
        },
    )

    report = reference.compare(candidate)
    reasons = {(item.stage, item.field, item.reason) for item in report.mismatches}

    assert ("02_decode", "*", "missing stage") in reasons
    assert (
        "01_preprocess",
        "bbox_xyxy",
        "logical dtype differs",
    ) in reasons


def test_parity_trace_only_writes_under_outputs(tmp_path: Path):
    with pytest.raises(ValueError, match="under outputs"):
        _trace("official").save(tmp_path / "official.npz")


def test_parity_trace_rejects_nonfinite_values():
    trace = MonocularParityTrace("official")
    with pytest.raises(ValueError, match="non-finite"):
        trace.capture("decode", {"joints": np.asarray([np.nan], dtype=np.float32)})
