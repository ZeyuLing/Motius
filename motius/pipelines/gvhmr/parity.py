"""GVHMR stage boundaries used for strict migration verification."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from motius.utils.monocular_parity import MonocularParityTrace


def capture_gvhmr_trace(
    output_path: str | Path,
    *,
    name: str,
    bbox: Mapping[str, object],
    data: Mapping[str, object],
    prediction: Mapping[str, object],
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Serialize the documented GVHMR inference boundaries."""

    trace = MonocularParityTrace(name, metadata=metadata)
    trace.capture(
        "01_tracking",
        {
            "bbx_xyxy": bbox["bbx_xyxy"],
            "bbx_xys": bbox["bbx_xys"],
        },
    )
    trace.capture(
        "02_camera",
        {
            "K_fullimg": data["K_fullimg"],
            "cam_angvel": data["cam_angvel"],
        },
    )
    trace.capture(
        "03_visual_features",
        {
            "kp2d": data["kp2d"],
            "f_imgseq": data["f_imgseq"],
        },
    )
    trace.capture("04_model_input", data)
    trace.capture("05_model_output", prediction)
    return trace.save(output_path)


__all__ = ["capture_gvhmr_trace"]
