from __future__ import annotations

import json

from tools.export_t2m_leaderboard_results import OUTPUT_PATH, export_payload


def test_t2m_json_matches_published_html_rows() -> None:
    assert json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) == export_payload()


def test_only_normalized_utmr_fid_is_public() -> None:
    payload = export_payload()
    assert payload["metric_protocol"]["motius_fid_field"] == "utmrFIDNorm"
    evaluated_rows = [
        row
        for row in payload["semantic_rows"]
        if row.get("utmrN") is not None
    ]
    assert evaluated_rows
    assert all(row.get("utmrFIDNorm") is not None for row in evaluated_rows)
