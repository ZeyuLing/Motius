#!/usr/bin/env python3
"""Audit leaderboard publication, metrics, visualization, and cross-links."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs/leaderboards/catalog.json"
TAXONOMY_PATH = ROOT / "docs/tasks/taxonomy.json"
GITHUB_DOCS_BASE = "https://github.com/ZeyuLing/Motius/blob/main/docs/tasks/"

NAV_START = "<!-- motius-benchmark-nav:start -->"
NAV_END = "<!-- motius-benchmark-nav:end -->"
SETTINGS_START = "<!-- motius-benchmark-settings:start -->"
SETTINGS_END = "<!-- motius-benchmark-settings:end -->"
PRODUCT_SHELL_BENCHMARKS = {
    "part_level_motion_control_humanml3d",
    "motion_repair_brokenamass",
    "motion_reconstruction_humanml3d",
    "monocular_motion_capture_3dpw_test",
    "motion_tracking_mujoco_lafan1_g1",
    "motion_tracking_isaaclab_g1",
}
PRODUCT_SHELL_TOKENS = {
    'class="score-strip"': "summary score strip",
    'class="panel comparison-studio"': "comparison studio",
    'id="bar-chart"': "bar chart",
    'id="radar-chart"': "radar chart",
    'id="table-body"': "results table",
    'class="panel protocol-details"': "protocol details",
    'class="site-footer"': "site footer",
}
CASE_EXPLORER_BENCHMARKS = {
    "part_level_motion_control_humanml3d",
    "motion_repair_brokenamass",
    "motion_reconstruction_humanml3d",
    "motion_tracking_mujoco_lafan1_g1",
    "motion_tracking_isaaclab_g1",
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def _public_target(target: str) -> str:
    if target.startswith(("https://", "http://")):
        return target
    return urljoin(GITHUB_DOCS_BASE, target)


def _navigation_target(target: str) -> tuple[str, str]:
    marker = "https://huggingface.co/spaces/"
    public = _public_target(target)
    if public.startswith(marker):
        repo_id = public[len(marker):].strip("/")
        static_host = repo_id.replace("/", "-").lower()
        return f"https://{static_host}.static.hf.space/", "_self"
    return public, "_blank"


def _leaderboard_id(benchmark: dict) -> str:
    return benchmark.get("leaderboard", {}).get("id", benchmark["id"])


def _navigation_targets(benchmarks: list[dict]) -> list[tuple[str, str]]:
    targets = []
    seen = set()
    for benchmark in benchmarks:
        leaderboard = benchmark.get("leaderboard")
        entry_id = _leaderboard_id(benchmark)
        if entry_id in seen:
            continue
        seen.add(entry_id)
        targets.append(
            _navigation_target(
                leaderboard["target"] if leaderboard else benchmark["target"]
            )
        )
    return targets


def _load_registry() -> tuple[list[dict], dict[str, dict], dict]:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_by_id = {item["id"]: item for item in catalog["benchmarks"]}
    benchmarks = []
    for item in taxonomy["benchmarks"]:
        merged = dict(item)
        merged.update(catalog_by_id.get(item["id"], {}))
        benchmarks.append(merged)
    return benchmarks, catalog_by_id, catalog


def _source_file(source: str) -> Path:
    return ROOT / source.split("#", 1)[0]


def _load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _numeric_fields(
    audit: Audit,
    row: dict,
    fields: tuple[str, ...],
    context: str,
) -> None:
    for field in fields:
        value = row.get(field)
        audit.require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{context}: missing numeric metric {field}",
        )


def _audit_t2m(audit: Audit, data: dict) -> None:
    required = (
        "utmrN",
        "utmrR1",
        "utmrR2",
        "utmrR3",
        "utmrFIDNorm",
        "utmrMM",
        "utmrDiv",
    )
    rows = data["semantic_rows"]
    audit.require(
        bool(rows) and rows[0].get("isReference") is True,
        "T2M: GT reference must be the first row",
    )
    generated = [row for row in rows if not row.get("isReference")]
    for row in generated:
        _numeric_fields(
            audit,
            row,
            required,
            f"T2M {row['method']} · {row.get('version', '')}",
        )
    audit.require(
        all(row.get("utmrFIDNorm") is not None for row in generated),
        "T2M: every generated row must publish normalized uTMR FID",
    )
    audit.require(
        data["metric_protocol"].get("motius_fid_field") == "utmrFIDNorm",
        "T2M: the public Motius FID field must be utmrFIDNorm",
    )
    expected = {
        ("HYMotion", "1.0B · 360f"): {
            "msR3": 0.9516,
            "msFID": 13.5194,
            "utmrR3": 0.9122,
            "utmrFIDNorm": 0.0147197,
        },
        ("HYMotion", "0.46B · 360f"): {
            "msR3": 0.9487,
            "msFID": 10.2711,
            "utmrR3": 0.9143,
            "utmrFIDNorm": 0.0156875,
        },
        ("MotionCanvas", "0.46B · 360f"): {
            "msR3": 0.9521,
            "msFID": 8.2765,
            "utmrR3": 0.9103,
            "utmrFIDNorm": 0.0107806,
        },
    }
    indexed = {(row["method"], row.get("version")): row for row in generated}
    for identity, metrics in expected.items():
        row = indexed.get(identity)
        audit.require(row is not None, f"T2M: missing fixed-360f row {identity}")
        if row is None:
            continue
        for field, value in metrics.items():
            audit.require(
                abs(float(row[field]) - value) < 5e-5,
                f"T2M {identity}: {field} is not the verified fixed-360f value",
            )


def _audit_motion_repair(audit: Audit, data: dict) -> None:
    rows = data["rows"]
    required = (
        "samples",
        "utmr_r1",
        "utmr_r3",
        "utmr_m2m",
        "mpjpe_cm",
        "gb_mpjpe_cm",
        "rte_m",
        "accel_error",
        "jitter",
        "skating_ratio",
        "penetration_ratio",
    )
    audit.require(
        data.get("population") == 299
        and data.get("clip_frames") == 100
        and data.get("fps") == 20,
        "Motion Repair: expected the 299-case, 100-frame, 20-fps protocol",
    )
    audit.require(
        [row.get("method") for row in rows[:2]]
        == ["Clean GT", "Corrupted input"],
        "Motion Repair: Clean GT and corrupted input must be first",
    )
    generated = [
        row for row in rows if not row.get("is_reference") and not row.get("is_input")
    ]
    audit.require(
        {row["method"] for row in generated}
        == {"MoGenDiT", "StableMotion", "MotionCanvas"},
        "Motion Repair: expected MoGenDiT, StableMotion, and MotionCanvas",
    )
    expected_support = {
        "MoGenDiT": ("method-native adaptive", "predicted_support"),
        "StableMotion": ("oracle v6", "oracle_support"),
        "MotionCanvas": ("oracle v6", "oracle_support"),
    }
    for row in rows:
        _numeric_fields(audit, row, required, f"Motion Repair {row['method']}")
        audit.require(
            row["samples"] == 299,
            f"Motion Repair {row['method']}: expected all 299 cases",
        )
    for row in generated:
        audit.require(
            (row.get("support"), row.get("track"))
            == expected_support[row["method"]],
            f"Motion Repair {row['method']}: support track is not explicit",
        )
    audit.require(
        data.get("primary_sort", {}).get("metric") == "utmr_r3",
        "Motion Repair: primary ordering must use uTMR R@3",
    )


def _audit_g1(audit: Audit, data: dict) -> None:
    rows = data["comparison_snapshot"]["rows"]
    audit.require(
        bool(rows) and rows[0].get("kind") == "reference",
        "G1: GT calibration reference must be the first row",
    )
    generated = [row for row in rows if row["kind"] == "generated"]
    hymotion = [row for row in generated if row["method"] == "HY-Motion G1"]
    audit.require(
        len(generated) == 2,
        "G1: expected exactly two released generated-method rows",
    )
    audit.require(
        len(hymotion) == 1 and hymotion[0].get("variant") == "released",
        "G1: HY-Motion must expose only the final released checkpoint",
    )
    for row in rows:
        _numeric_fields(
            audit,
            row,
            ("samples", "r1", "r2", "r3", "fid", "mm_dist", "diversity"),
            f"G1 {row['method']} · {row['variant']}",
        )


def _audit_m2t(audit: Audit, data: dict) -> None:
    required = (
        "bleu1",
        "bleu4",
        "rougeL",
        "cider",
        "bertRaw",
        "bertF1",
        "r1",
        "r2",
        "r3",
        "matching",
    )
    methods = data["methods"]
    audit.require(
        bool(methods) and methods[0].get("kind") == "reference",
        "M2T: GT captions must be the first row",
    )
    for method in methods:
        _numeric_fields(
            audit,
            method["metrics"],
            required,
            f"M2T {method['name']}",
        )


def _audit_babel(audit: Audit, data: dict) -> None:
    required = (
        "samples",
        "r1",
        "r2",
        "r3",
        "fid",
        "mmDist",
        "diversity",
        "transitionFid",
        "transitionDiversity",
        "peakJerk",
        "aujGap",
    )
    rows = data["rows"]
    audit.require(
        bool(rows) and rows[0].get("isReference") is True,
        "BABEL: GT reference must be the first row",
    )
    for row in rows:
        _numeric_fields(audit, row, required, f"BABEL {row['method']}")
    audit.require(
        data.get("fid_space")
        == "per-sample L2-normalized uTMR embedding space",
        "BABEL: FID must be declared in normalized uTMR space",
    )


def _audit_temporal(audit: Audit, data: dict) -> None:
    required = (
        "r_precision_top1",
        "r_precision_top2",
        "r_precision_top3",
        "fid",
        "mm_dist",
        "constraint_error_cm",
        "fail_20",
        "fail_50",
        "foot_skating",
        "diversity",
    )
    audit.require(len(data["settings"]) == 8, "Temporal: expected eight settings")
    for setting in data["settings"]:
        audit.require(
            setting.get("review_status") == "complete",
            f"Temporal {setting['id']}: review is not complete",
        )
        audit.require(
            bool(setting["methods"])
            and setting["methods"][0].get("is_reference") is True,
            f"Temporal {setting['id']}: GT reference must be first",
        )
        for method in setting["methods"]:
            _numeric_fields(
                audit,
                method["metrics"],
                required,
                f"Temporal {setting['id']} · {method['method']}",
            )


def _audit_body_part(audit: Audit, data: dict) -> None:
    required = (
        "r_precision_top1",
        "r_precision_top2",
        "r_precision_top3",
        "fid",
        "mm_dist",
        "control_error",
        "foot_skating",
        "jitter",
        "diversity",
    )
    audit.require(len(data["settings"]) == 84, "Body-part: expected 84 settings")
    for setting in data["settings"]:
        audit.require(
            bool(setting["methods"])
            and setting["methods"][0].get("is_reference") is True,
            f"Body-part {setting['id']}: GT reference must be first",
        )
        for method in setting["methods"]:
            audit.require(
                method["protocol_status"] != "unsupported",
                f"Body-part {setting['id']} · {method['method']}: "
                "unsupported rows must not be published",
            )
            fields = tuple(
                field
                for field in required
                if not (method.get("is_reference") and field == "jitter")
            )
            _numeric_fields(
                audit,
                method["metrics"],
                fields,
                f"Body-part {setting['id']} · {method['method']}",
            )
            audit.require(
                method["artifacts"].get("count", 0) > 0,
                f"Body-part {setting['id']} · {method['method']}: no artifacts",
            )


def _audit_motion_edit(audit: Audit, data: dict) -> None:
    required = {item["key"] for item in data["metrics"]}
    for track in data["tracks"]:
        audit.require(
            bool(track["rows"])
            and track["rows"][0].get("reference_kind") == "gt",
            f"Motion editing {track['id']}: GT target must be first",
        )
        for row in track["rows"]:
            missing = required - set(row["metrics"])
            audit.require(
                not missing,
                f"Motion editing {track['id']} · {row['method']}: "
                f"missing {sorted(missing)}",
            )
            if row.get("reference_kind") != "gt":
                audit.require(
                    all(value is not None for value in row["metrics"].values()),
                    f"Motion editing {track['id']} · {row['method']}: null metric",
                )


def _audit_instruction_edit(audit: Audit, data: dict) -> None:
    required = tuple(data["metrics"])
    rows = data["rows"]
    audit.require(
        bool(rows) and rows[0].get("method") == "GT target",
        "MotionFix: GT target must be the first row",
    )
    motioncanvas = [row for row in rows if row.get("method") == "MotionCanvas"]
    audit.require(
        len(motioncanvas) == 1
        and motioncanvas[0].get("ranked") is True
        and motioncanvas[0].get("rank") == 1,
        "MotionFix: MotionCanvas must be a formal rank-1 measured row",
    )
    for row in rows:
        _numeric_fields(
            audit,
            row["metrics"],
            required,
            f"MotionFix {row['method']}",
        )


def _audit_music_to_dance(audit: Audit, data: dict) -> None:
    required = (
        "fid_k",
        "fid_g",
        "diversity_k",
        "diversity_g",
        "beat_align_30fps",
        "fid_utmr",
        "jitter",
        "float",
        "slide",
    )
    rows = data["rows"]
    audit.require(
        bool(rows) and rows[0].get("method") == "GT",
        "Music-to-Dance: GT must be the first row",
    )
    for row in rows:
        _numeric_fields(audit, row, required, f"Music-to-Dance {row['method']}")


def _audit_dance_to_music(audit: Audit, data: dict) -> None:
    rows = data["rows"]
    audit.require(
        [row.get("method") for row in rows] == ["UniMuMo"],
        "Dance-to-Music: only the completed UniMuMo reproduction may be listed",
    )
    audit.require(
        data.get("results_scope") == "motius_measured_only"
        and all(
            row.get("source") == "motius" and not row.get("reference", False)
            for row in rows
        ),
        "Dance-to-Music: public rows must contain only Motius-measured results",
    )
    audit.require(
        "paper_rows" not in data,
        "Dance-to-Music: paper-reported result rows must be omitted",
    )
    for row in rows:
        _numeric_fields(
            audit,
            row,
            ("beat_count_ratio", "beat_hit_rate"),
            f"Dance-to-Music {row['method']}",
        )


def _audit_speech(audit: Audit, data: dict) -> None:
    required = (
        "fgd",
        "bc",
        "diversity",
        "rotation_geodesic",
        "expression_l2",
        "translation_l2",
        "utmr_fid",
        "utmr_paired_distance",
        "utmr_diversity",
    )
    rows = data["rows"]
    audit.require(
        bool(rows) and rows[0].get("kind") == "reference",
        "Speech-to-Gesture: GT must be the first row",
    )
    for row in rows:
        _numeric_fields(audit, row, required, f"Speech-to-Gesture {row['method']}")


def _audit_reconstruction(audit: Audit, data: dict) -> None:
    rows = data["rows"]
    audit.require(
        bool(rows) and rows[0].get("is_reference") is True,
        "Motion Reconstruction: GT must be the first row",
    )
    required = (
        "rfid",
        "embedding_l2",
        "mpjpe_mm",
        "pa_mpjpe_mm",
        "mpjre_deg",
        "slide",
        "float",
        "penetration",
        "jitter",
    )
    for row in rows:
        _numeric_fields(
            audit,
            row,
            required,
            f"Motion Reconstruction {row['method']}",
        )


def _audit_monocular(audit: Audit, data: dict) -> None:
    rows = data["rows"]
    audit.require(
        bool(rows) and rows[0].get("reference") is True,
        "Monocular Capture: GT must be the first row",
    )
    audit.require(len(data["methods"]) == 4, "Monocular Capture: expected four methods")
    for method in data["methods"]:
        demo = method.get("demo", {})
        audit.require(
            str(demo.get("video", "")).startswith("https://"),
            f"Monocular Capture {method['method']}: missing public video demo",
        )
    for row in rows:
        _numeric_fields(
            audit,
            row["metrics"],
            ("pa_mpjpe_mm", "mpjpe_mm", "accel_mps2"),
            f"Monocular Capture {row['method']}",
        )


def _audit_motion_tracking(audit: Audit, data: dict) -> None:
    rows = data.get("rows", [])
    audit.require(
        data.get("schema_version") == 2 and data.get("status") == "complete",
        "Motion Tracking: expected complete schema v2 result pack",
    )
    audit.require(
        bool(rows)
        and rows[0].get("kind") == "reference"
        and rows[0].get("rankable") is False,
        "Motion Tracking: GT calibration reference must be first",
    )
    generated = [row for row in rows if row.get("kind") != "reference"]
    expected_methods = (
        3
        if data.get("benchmark_id") == "motion_tracking_mujoco_lafan1_g1"
        else 2
    )
    audit.require(
        len(generated) == expected_methods,
        f"Motion Tracking: expected {expected_methods} measured method rows",
    )
    metric_keys = tuple(item["key"] for item in data.get("metric_definitions", []))
    split_populations = {
        item["id"]: item["population"] for item in data.get("splits", [])
    }
    audit.require(bool(metric_keys), "Motion Tracking: missing metric definitions")
    audit.require(bool(split_populations), "Motion Tracking: missing split registry")
    for row in rows:
        for split, population in split_populations.items():
            measured = row.get("splits", {}).get(split)
            audit.require(
                isinstance(measured, dict),
                f"Motion Tracking {row['method']}: missing split {split}",
            )
            if not isinstance(measured, dict):
                continue
            coverage = measured.get("coverage", {})
            evaluated = coverage.get("evaluated")
            audit.require(
                isinstance(evaluated, int)
                and 0 < evaluated <= population
                and coverage.get("population") == population,
                f"Motion Tracking {row['method']} · {split}: invalid coverage",
            )
            _numeric_fields(
                audit,
                measured.get("metrics", {}),
                metric_keys,
                f"Motion Tracking {row['method']} · {split}",
            )
        if row.get("kind") == "reference":
            audit.require(
                all(
                    row["splits"][split]["coverage"]["evaluated"] == population
                    for split, population in split_populations.items()
                ),
                "Motion Tracking: GT must cover every registered reference",
            )

    if data.get("benchmark_id") == "motion_tracking_mujoco_lafan1_g1":
        audit.require(
            split_populations == {"lafan1_g1": 40}
            and all(row.get("rankable") for row in generated)
            and all(
                row["splits"]["lafan1_g1"]["coverage"]["evaluated"] == 40
                for row in generated
            ),
            "MuJoCo Motion Tracking: all controllers must cover all 40 cases",
        )
    if data.get("benchmark_id") == "motion_tracking_isaaclab_g1":
        by_method = {row["method"]: row for row in generated}
        sonic = by_method.get("SONIC", {})
        beyond = by_method.get("BeyondMimic", {})
        audit.require(
            split_populations == {"lafan1_g1": 40, "amass_test_g1": 138},
            "Isaac Lab Motion Tracking: expected LAFAN1 40 + AMASS-test 138",
        )
        audit.require(
            sonic.get("rankable") is True
            and sonic.get("splits", {}).get("lafan1_g1", {}).get("coverage", {}).get("evaluated") == 40
            and sonic.get("splits", {}).get("amass_test_g1", {}).get("coverage", {}).get("evaluated") == 138,
            "Isaac Lab Motion Tracking: SONIC must cover all 178 references",
        )
        audit.require(
            beyond.get("rankable") is False
            and beyond.get("splits", {}).get("lafan1_g1", {}).get("coverage", {}).get("evaluated") == 40
            and beyond.get("splits", {}).get("amass_test_g1", {}).get("coverage", {}).get("evaluated") == 100,
            "Isaac Lab Motion Tracking: BeyondMimic upper-bound coverage must be 40 + 100",
        )


METRIC_AUDITORS = {
    "text_to_motion_humanml3d": _audit_t2m,
    "text_to_motion_unitree_g1": _audit_g1,
    "motion_to_text_humanml3d": _audit_m2t,
    "sequential_text_to_motion_babel": _audit_babel,
    "temporal_motion_completion_humanml3d": _audit_temporal,
    "part_level_motion_control_humanml3d": _audit_body_part,
    "motion_editing_style_content": _audit_motion_edit,
    "motion_editing_motionfix": _audit_instruction_edit,
    "motion_repair_brokenamass": _audit_motion_repair,
    "music_to_dance_aistpp": _audit_music_to_dance,
    "dance_to_music_aistpp": _audit_dance_to_music,
    "speech_to_gesture_beat2": _audit_speech,
    "motion_reconstruction_humanml3d": _audit_reconstruction,
    "monocular_motion_capture_3dpw_test": _audit_monocular,
    "motion_tracking_mujoco_lafan1_g1": _audit_motion_tracking,
    "motion_tracking_isaaclab_g1": _audit_motion_tracking,
}


def _manifest_population(data: dict) -> int | None:
    population = data.get("population")
    if isinstance(population, int):
        return population
    cases = data.get("cases")
    if isinstance(cases, list):
        return len(cases)
    if isinstance(data.get("tracks"), dict):
        return sum(
            len(track.get("cases", [])) for track in data["tracks"].values()
        )
    methods = data.get("methods")
    if isinstance(methods, list):
        return len(methods)
    return None


def _audit_visualization(audit: Audit, benchmark: dict) -> None:
    visual = benchmark["visualization"]
    if visual["status"] not in {"complete", "partial"}:
        audit.note(
            f"{benchmark['label']}: visualization {visual['status']}"
            + (f" ({visual['reason']})" if visual.get("reason") else "")
        )
        return
    if visual["status"] == "partial":
        audit.note(
            f"{benchmark['label']}: visualization partial"
            + (f" ({visual['reason']})" if visual.get("reason") else "")
        )
    entry = ROOT / visual["entry"]
    manifest = ROOT / visual["manifest"]
    audit.require(entry.is_file(), f"{benchmark['label']}: missing viewer {entry}")
    audit.require(
        manifest.is_file(),
        f"{benchmark['label']}: missing visualization manifest {manifest}",
    )
    if not manifest.is_file():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if visual.get("external_assets"):
        asset_base = data.get("asset_base_url")
        audit.require(
            (
                isinstance(asset_base, str)
                and asset_base.startswith(("https://", "http://"))
            )
            or "https://" in json.dumps(data),
            f"{benchmark['label']}: external visualization assets have no "
            "absolute public URL",
        )
    population = _manifest_population(data)
    audit.require(
        population == visual["population"],
        f"{benchmark['label']}: visualization population "
        f"{population} != {visual['population']}",
    )
    if visual.get("audio"):
        serialized = json.dumps(data).lower()
        audit.require(
            "audio" in serialized,
            f"{benchmark['label']}: audio visualization has no audio source",
        )
    if benchmark["id"] == "part_level_motion_control_humanml3d":
        protocol = data.get("protocol", {})
        setting_id = protocol.get("setting_id")
        results = _load_json(benchmark["results"])
        settings = {
            setting["id"]: setting for setting in results.get("settings", [])
        }
        audit.require(
            setting_id in settings,
            f"{benchmark['label']}: viewer setting {setting_id!r} has no metrics",
        )
        if setting_id in settings:
            metric_methods = [
                method["method"] for method in settings[setting_id]["methods"]
            ]
            visual_methods = [
                method["label"] for method in data.get("motion_methods", [])
            ]
            audit.require(
                visual_methods == metric_methods,
                f"{benchmark['label']}: visual methods {visual_methods} do not "
                f"match metric methods {metric_methods}",
            )
    if benchmark["id"] == "motion_reconstruction_humanml3d":
        results = _load_json(benchmark["results"])
        metric_methods = [row["method"] for row in results.get("rows", [])]
        visual_methods = [
            method["label"] for method in data.get("motion_methods", [])
        ]
        audit.require(
            visual_methods == metric_methods,
            f"{benchmark['label']}: visual methods {visual_methods} do not "
            f"match metric methods {metric_methods}",
        )
    if benchmark["id"] == "motion_repair_brokenamass":
        results = _load_json(benchmark["results"])
        metric_methods = [row["method"] for row in results.get("rows", [])]
        visual_methods = [
            method["label"] for method in data.get("motion_methods", [])
        ]
        audit.require(
            visual_methods[:2] == ["Clean GT", "Corrupted input"]
            and set(visual_methods) == set(metric_methods),
            f"{benchmark['label']}: visual methods {visual_methods} do not "
            f"match metric methods {metric_methods}",
        )
        audit.require(
            data.get("population") == 299
            and len(data.get("cases", [])) == 299,
            "Motion Repair: viewer must expose all 299 paired cases",
        )
    if benchmark["id"].startswith("motion_tracking_"):
        results = _load_json(benchmark["results"])
        metric_methods = [row["method"] for row in results.get("rows", [])]
        visual_methods = [column["label"] for column in data.get("columns", [])]
        audit.require(
            visual_methods == metric_methods,
            f"{benchmark['label']}: visual methods {visual_methods} do not "
            f"match metric methods {metric_methods}",
        )
        cases = data.get("cases", [])
        split_counts = {
            split["id"]: sum(case.get("split") == split["id"] for case in cases)
            for split in results.get("splits", [])
        }
        audit.require(
            all(
                split_counts.get(split["id"]) == split["population"]
                for split in results.get("splits", [])
            ),
            f"{benchmark['label']}: viewer split population mismatch",
        )
        audit.require(
            all("gt" in case.get("assets", {}) for case in cases),
            f"{benchmark['label']}: every case must retain a GT rollout",
        )
        if benchmark["id"] == "motion_tracking_isaaclab_g1":
            audit.require(
                sum("sonic" in case.get("assets", {}) for case in cases) == 178
                and sum(
                    "beyondmimic" in case.get("assets", {}) for case in cases
                )
                == 140,
                "Isaac Lab Motion Tracking: viewer coverage must be SONIC 178 "
                "+ BeyondMimic 140",
            )


def _audit_page(
    audit: Audit,
    benchmark: dict,
    benchmarks: list[dict],
    hub_target: str,
) -> None:
    source = _source_file(benchmark["source"])
    audit.require(source.exists(), f"{benchmark['label']}: missing source {source}")
    if not source.is_dir():
        return
    page = source / "index.html"
    readme = source / "README.md"
    audit.require(page.is_file(), f"{benchmark['label']}: missing index.html")
    audit.require(readme.is_file(), f"{benchmark['label']}: missing README.md")
    if readme.is_file():
        audit.require(
            benchmark["label"] in readme.read_text(encoding="utf-8"),
            f"{benchmark['label']}: README does not use the canonical title",
        )
    if not page.is_file():
        return
    text = page.read_text(encoding="utf-8")
    audit.require(NAV_START in text and NAV_END in text, f"{benchmark['label']}: unsynchronized navigation")
    audit.require(
        f'data-benchmark-id="{benchmark["id"]}"' in text,
        f"{benchmark['label']}: missing benchmark identity",
    )
    audit.require(
        benchmark["label"] in text,
        f"{benchmark['label']}: canonical title is absent",
    )
    audit.require(
        hub_target in text,
        f"{benchmark['label']}: Benchmark Hub link is absent",
    )
    if benchmark["id"] in PRODUCT_SHELL_BENCHMARKS:
        for token, feature in PRODUCT_SHELL_TOKENS.items():
            audit.require(
                token in text,
                f"{benchmark['label']}: product shell is missing {feature}",
            )
        style = source / "leaderboard.css"
        audit.require(
            style.is_file(),
            f"{benchmark['label']}: shared leaderboard stylesheet is absent",
        )
    if benchmark["id"] in CASE_EXPLORER_BENCHMARKS:
        audit.require(
            'class="panel case-explorer"' in text
            and 'src="cases/index.html"' in text,
            f"{benchmark['label']}: embedded all-case comparison is absent",
        )
    preferred_sort_tokens = {
        "text_to_motion_humanml3d": 'sortKey: "utmrR3"',
        "sequential_text_to_motion_babel": 'sortKey:"r3"',
        "motion_repair_brokenamass": 'activeMetric = "utmr_r3"',
        "music_to_dance_aistpp": "ranks.fid_utmr.get(row.method)",
        "speech_to_gesture_beat2": "ranks.utmr_fid.get(row.method)",
    }
    token = preferred_sort_tokens.get(benchmark["id"])
    if token:
        sort_source = text
        script = source / "leaderboard.js"
        if script.is_file():
            sort_source += script.read_text(encoding="utf-8")
        audit.require(
            token in sort_source,
            f"{benchmark['label']}: default ranking is not uTMR-first",
        )
    for target, target_mode in _navigation_targets(benchmarks):
        audit.require(
            f'href="{target}" target="{target_mode}"' in text,
            f"{benchmark['label']}: missing sandbox-safe navigation to {target}",
        )
    leaderboard = benchmark.get("leaderboard")
    if leaderboard:
        audit.require(
            SETTINGS_START in text and SETTINGS_END in text,
            f"{benchmark['label']}: setting navigation is absent",
        )
        settings = [
            item
            for item in benchmarks
            if _leaderboard_id(item) == leaderboard["id"]
        ]
        for setting in settings:
            target, target_mode = _navigation_target(setting["target"])
            audit.require(
                (
                    f'href="{target}" target="{target_mode}"' in text
                    and setting["setting"]["label"] in text
                ),
                f"{benchmark['label']}: missing setting "
                f"{setting['setting']['label']}",
            )


def _audit_online(audit: Audit, benchmarks: list[dict]) -> None:
    for benchmark in benchmarks:
        target = benchmark["target"]
        marker = "https://huggingface.co/spaces/"
        if marker not in target:
            continue
        repo_id = target.split(marker, 1)[1]
        api_url = f"https://huggingface.co/api/spaces/{repo_id}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(api_url, timeout=30) as response:
                    metadata = json.load(response)
                stage = (metadata.get("runtime") or {}).get("stage")
                audit.require(
                    stage == "RUNNING",
                    f"{benchmark['label']}: live Space stage is {stage}",
                )
                static_url = (
                    f"https://{repo_id.replace('/', '-').lower()}.static.hf.space/"
                )
                with urllib.request.urlopen(static_url, timeout=30) as response:
                    live_page = response.read().decode("utf-8", errors="replace")
                audit.require(
                    NAV_START in live_page,
                    f"{benchmark['label']}: live Space has stale navigation",
                )
                visual = benchmark["visualization"]
                if visual["status"] in {"complete", "partial"}:
                    source_dir = Path(benchmark["source"])
                    entry_path = Path(visual["entry"]).relative_to(source_dir)
                    manifest_path = Path(visual["manifest"]).relative_to(
                        source_dir
                    )
                    with urllib.request.urlopen(
                        urljoin(static_url, entry_path.as_posix()),
                        timeout=30,
                    ) as response:
                        response.read(1)
                    with urllib.request.urlopen(
                        urljoin(static_url, manifest_path.as_posix()),
                        timeout=30,
                    ) as response:
                        json.load(response)
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(attempt + 1)
        if last_error is not None:
            audit.errors.append(
                f"{benchmark['label']}: online check failed after 3 attempts: "
                f"{last_error}"
            )


def _find_unsupported_entries(value: object, path: str = "$") -> list[str]:
    entries: list[str] = []
    if isinstance(value, dict):
        for key in ("status", "protocol_status"):
            status = value.get(key)
            if isinstance(status, str) and status.lower().replace(" ", "_") in {
                "unsupported",
                "not_supported",
            }:
                entries.append(f"{path}.{key}")
        for key, child in value.items():
            entries.extend(_find_unsupported_entries(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.extend(_find_unsupported_entries(child, f"{path}[{index}]"))
    return entries


def run(*, online: bool = False) -> Audit:
    audit = Audit()
    benchmarks, catalog_by_id, catalog = _load_registry()
    taxonomy_ids = {item["id"] for item in benchmarks}
    catalog_ids = set(catalog_by_id)
    audit.require(
        taxonomy_ids == catalog_ids,
        "Catalog and task taxonomy benchmark IDs differ: "
        f"missing={sorted(taxonomy_ids - catalog_ids)}, "
        f"extra={sorted(catalog_ids - taxonomy_ids)}",
    )
    for benchmark in benchmarks:
        expected_root = (
            "outputs/evaluation/"
            f"{benchmark['task']}/{benchmark['id']}/{benchmark['protocol_id']}"
        )
        audit.require(
            benchmark["artifact_root"] == expected_root,
            f"{benchmark['label']}: artifact root is not canonical",
        )
        status = benchmark["status"]
        metrics = benchmark["metrics"]
        visual = benchmark["visualization"]
        if status == "complete":
            audit.require(
                metrics["status"] == "complete"
                and visual["status"] == "complete",
                f"{benchmark['label']}: complete status overstates coverage",
            )
        elif status == "metrics_complete":
            audit.require(
                metrics["status"] == "complete"
                and visual["status"] in {"partial", "pending"},
                f"{benchmark['label']}: metrics_complete status is inconsistent",
            )
        elif status == "protocol_only":
            audit.require(
                metrics["status"] == "pending",
                f"{benchmark['label']}: protocol-only metrics must be pending",
            )
        elif status == "protocol_ready":
            audit.require(
                metrics["status"] == "pending"
                and visual["status"] == "pending",
                f"{benchmark['label']}: protocol_ready status is inconsistent",
            )
        elif status == "paused":
            audit.require(
                metrics["status"] == "paused",
                f"{benchmark['label']}: paused metrics must be paused",
            )
        else:
            audit.errors.append(f"{benchmark['label']}: unknown status {status}")

        _audit_page(
            audit,
            benchmark,
            benchmarks,
            catalog["navigation_target"],
        )
        _audit_visualization(audit, benchmark)
        result_path = benchmark.get("results")
        if result_path:
            result = ROOT / result_path
            audit.require(
                result.is_file(),
                f"{benchmark['label']}: missing result snapshot {result}",
            )
            if result.is_file():
                snapshot = _load_json(result_path)
                unsupported = _find_unsupported_entries(snapshot)
                audit.require(
                    not unsupported,
                    f"{benchmark['label']}: unsupported entries must be omitted "
                    f"from public results ({unsupported[:5]})",
                )
                if benchmark["id"] in METRIC_AUDITORS:
                    METRIC_AUDITORS[benchmark["id"]](audit, snapshot)

    if online:
        _audit_online(audit, benchmarks)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Also verify that every public Hugging Face Space is running and synchronized.",
    )
    args = parser.parse_args()
    audit = run(online=args.online)
    if audit.notes:
        print("Coverage notes:")
        for note in audit.notes:
            print(f"  - {note}")
    if audit.errors:
        print(f"Leaderboard audit failed with {len(audit.errors)} error(s):")
        for error in audit.errors:
            print(f"  - {error}")
        return 1
    print("Leaderboard audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
