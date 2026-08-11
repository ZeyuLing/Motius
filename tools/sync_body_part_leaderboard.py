#!/usr/bin/env python3
"""Refresh the public part-control snapshot from completed metric artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT
    / "docs/leaderboards/hf_space_body_part_condition_humanml3d/"
    "body_part_condition_results.json"
)
DEFAULT_PAGE = (
    ROOT / "docs/leaderboards/hf_space_body_part_condition_humanml3d/index.html"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/outputs/"
    "evaluation/body_part/humanml3d_official_test_4012"
)
METHOD_DIRECTORIES = {
    "MotionCanvas": "ours",
    "OmniControl": "omnicontrol",
    "CondMDI": "condmdi",
    "MaskControl": "maskcontrol",
    "MotionLab": "motionlab",
}
GT_REFERENCE = {
    "method": "GT",
    "method_id": "gt",
    "protocol_status": "reference",
    "is_reference": True,
    "rank_excluded": True,
    "metrics": {
        "r_precision_top1": 0.6873,
        "r_precision_top2": 0.8485,
        "r_precision_top3": 0.9058,
        "fid": 0.0,
        "mm_dist": 27.6009,
        "control_error": 0.0,
        "foot_skating": 0.08914819392293807,
        "jitter": None,
        "diversity": 53.941,
    },
    "metric_sources": {
        "semantic": "HumanML3D selected-caption GT reference",
        "control": "analytic identity",
    },
    "artifacts": {"count": 4012},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def refresh(data: dict, artifact_root: Path) -> dict:
    for setting in data["settings"]:
        rows = {
            row["method"]: row
            for row in setting["methods"]
            if row.get("method_id") != "gt"
            and row.get("protocol_status") == "native"
            and row.get("method") in METHOD_DIRECTORIES
        }
        for method, directory in METHOD_DIRECTORIES.items():
            summary = artifact_root / setting["id"] / directory / "metrics_summary.json"
            if not summary.is_file():
                continue
            row = rows.get(method)
            if row is None:
                row = {
                    "method": method,
                    "method_id": directory,
                    "protocol_status": "native",
                    "metrics": {},
                    "metric_sources": {},
                    "artifacts": {},
                }
                rows[method] = row
            row["protocol_status"] = "native"
            row["metrics"] = json.loads(summary.read_text(encoding="utf-8"))
            relative_root = (
                "outputs/evaluation/body_part/humanml3d_official_test_4012/"
                f"{setting['id']}/{directory}"
            )
            row["metric_sources"] = {"summary": f"{relative_root}/metrics_summary.json"}
            row["artifacts"] = {"root": relative_root, "count": data["num_cases"]}

        original_order = [
            row["method"]
            for row in setting["methods"]
            if row.get("method_id") != "gt"
            and row.get("protocol_status") == "native"
            and row.get("method") in METHOD_DIRECTORIES
        ]
        added = [method for method in METHOD_DIRECTORIES if method not in original_order]
        setting["methods"] = [dict(GT_REFERENCE)] + [
            rows[method] for method in original_order + added if method in rows
        ]
    data["updated"] = date.today().isoformat()
    return data


def embed_snapshot(page: str, data: dict) -> str:
    public = json.loads(json.dumps(data))
    for setting in public["settings"]:
        setting.pop("canonical_root", None)
        for row in setting["methods"]:
            row["metric_sources"] = {}
            row["artifacts"].pop("root", None)
    payload = json.dumps(public, ensure_ascii=False, separators=(",", ":"))
    pattern = re.compile(r"const DATA=\{.*?\};const E=", re.DOTALL)
    replacement = f"const DATA={payload};const E="
    updated, count = pattern.subn(replacement, page, count=1)
    if count == 0 and 'src="leaderboard.js"' in page:
        return page
    if count != 1:
        raise RuntimeError("could not locate embedded body-part snapshot")
    return updated


def main() -> None:
    args = parse_args()
    data = refresh(
        json.loads(args.results.read_text(encoding="utf-8")),
        args.artifact_root,
    )
    args.results.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = embed_snapshot(args.page.read_text(encoding="utf-8"), data)
    args.page.write_text(page, encoding="utf-8")
    completed = sum(
        1
        for setting in data["settings"]
        for row in setting["methods"]
        if row.get("method_id") != "gt"
    )
    print(f"refreshed {len(data['settings'])} settings and {completed} measured rows")


if __name__ == "__main__":
    main()
