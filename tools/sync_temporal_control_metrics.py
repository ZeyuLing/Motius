#!/usr/bin/env python3
"""Incrementally publish Temporal Motion Completion evaluation results."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_temporal_condition"
    / "temporal_control_results.json"
)
DEFAULT_METHODS = {
    "maskcontrol": "MaskControl",
    "omnicontrol": "OmniControl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        required=True,
        type=Path,
        help="Root containing temporal_<setting>/<method>/metrics_* directories.",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="ID=LABEL",
        help="Method to synchronize; repeat for multiple methods.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print updates without writing the results file.",
    )
    return parser.parse_args()


def parse_methods(values: list[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_METHODS)
    methods: dict[str, str] = {}
    for value in values:
        method_id, separator, label = value.partition("=")
        if not separator or not method_id.strip() or not label.strip():
            raise ValueError(f"Invalid --method {value!r}; expected ID=LABEL")
        methods[method_id.strip()] = label.strip()
    return methods


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def load_complete_metrics(
    canonical_root: Path,
    setting_id: str,
    method_id: str,
    expected_samples: int,
) -> dict | None:
    method_root = canonical_root / setting_id / method_id
    paths = {
        "semantic": method_root / "metrics_utmr" / "leaderboard.json",
        "condition": method_root / "metrics_condition" / "summary.json",
        "physical": method_root / "metrics_physical" / "summary.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None

    semantic = load_json(paths["semantic"])
    condition = load_json(paths["condition"])
    physical_payload = load_json(paths["physical"])
    if len(physical_payload) != 1:
        raise ValueError(f"Expected one physical result in {paths['physical']}")
    physical = next(iter(physical_payload.values()))

    sample_counts = {
        "semantic": semantic.get("num_samples"),
        "condition": condition.get("num_cases"),
        "physical": physical.get("n"),
    }
    invalid = {key: value for key, value in sample_counts.items() if value != expected_samples}
    if invalid:
        raise ValueError(
            f"Incomplete {method_id}/{setting_id}: expected {expected_samples}, got {invalid}"
        )

    return {
        "r_precision_top1": semantic["R-Precision Top 1"],
        "r_precision_top2": semantic["R-Precision Top 2"],
        "r_precision_top3": semantic["R-Precision Top 3"],
        "fid": semantic["FID"],
        "mm_dist": semantic["MM-Dist"],
        "constraint_error_cm": condition["condition_error_cm"],
        "fail_20": condition["fail_20"],
        "fail_50": condition["fail_50"],
        "foot_skating": physical["foot_skating_ratio"],
        "diversity": semantic["Diversity"],
    }


def upsert_method(setting: dict, method: dict) -> bool:
    methods = setting["methods"]
    for index, current in enumerate(methods):
        if current.get("method_id") == method["method_id"]:
            if current == method:
                return False
            methods[index] = method
            return True
    methods.append(method)
    return True


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    payload = load_json(args.results)
    expected_samples = int(payload["num_cases"])
    changed = False
    complete = 0

    for setting in payload["settings"]:
        setting_id = setting["id"]
        for method_id, label in methods.items():
            metrics = load_complete_metrics(
                args.canonical_root,
                setting_id,
                method_id,
                expected_samples,
            )
            if metrics is None:
                print(f"[skip-incomplete] {method_id}:{setting_id}")
                continue
            row = {
                "method": label,
                "method_id": method_id,
                "samples": expected_samples,
                "metrics": metrics,
            }
            row_changed = upsert_method(setting, row)
            changed = changed or row_changed
            complete += 1
            print(f"[{'update' if row_changed else 'unchanged'}] {method_id}:{setting_id}")

    if changed:
        payload["updated"] = date.today().isoformat()
    if not args.check and changed:
        args.results.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"complete={complete} changed={changed} "
        f"mode={'check' if args.check else 'write'} results={args.results}"
    )


if __name__ == "__main__":
    main()
