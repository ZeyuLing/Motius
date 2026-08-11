import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "docs/leaderboards/hf_space_temporal_condition"


def test_tp2m_rows_live_in_machine_readable_results() -> None:
    payload = json.loads(
        (SPACE / "temporal_control_results.json").read_text(encoding="utf-8")
    )
    rows = payload["tp2m_rows"]

    assert len(rows) == 15
    assert len({(row["method"], row["settingId"]) for row in rows}) == 15
    assert {row["settingId"] for row in rows} == {"c1", "c5", "c9"}
    assert {row["method"] for row in rows} == {
        "GT",
        "FlowMDM",
        "MotionStreamer",
        "KIMODO",
        "PRISM-KT",
    }
    assert payload["tp2m_protocol"]["fid_space"] == (
        "MotionStreamer evaluator embedding space"
    )


def test_temporal_frontend_reads_tp2m_rows_from_json() -> None:
    script = (SPACE / "leaderboard.js").read_text(encoding="utf-8")

    assert "const TP2M_ROWS" not in script
    assert "state.rowsByProtocol.tp2m = Array.from(data.tp2m_rows)" in script
