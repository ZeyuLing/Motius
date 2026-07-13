#!/usr/bin/env python3
"""Render the public T2M leaderboard from the Model Zoo release manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs/model_zoo/release_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/leaderboards/t2m_humanml3d.md"
EVALUATORS = {
    "HumanML3D Official": "../evaluator_zoo/humanml3d_official.md",
    "MotionStreamer Evaluator": "../evaluator_zoo/motionstreamer.md",
    "Motius Joint-Position Evaluator": "../evaluator_zoo/motius_joint_position.md",
}
METRICS = ("R1", "R2", "R3", "FID", "MM-Dist", "Diversity")


def _format_value(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(manifest: dict[str, object]) -> str:
    lines = [
        "# T2M HumanML3D Leaderboard",
        "",
        "[Back to the Motius Model Zoo](../../README.md#model-zoo)",
        "",
        "This leaderboard is generated from the measured rows in",
        "[`release_manifest.json`](../model_zoo/release_manifest.json). It reports",
        "the three evaluator views required by every public Motius T2M model card.",
        "Compare methods only within the same evaluator table; each evaluator has",
        "its own motion representation, embedding space, and protocol.",
        "",
        "R@1, R@2, R@3, and Diversity are higher-is-better. FID and MM-Dist are",
        "lower-is-better.",
        "",
    ]

    models = manifest.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("release manifest field 'models' must be an object")

    for evaluator, card_path in EVALUATORS.items():
        lines.extend(
            [
                f"## [{evaluator}]({card_path})",
                "",
                "| Method | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |",
                "| ------ | ------- | ------: | --: | --: | --: | --: | ------: | --------: |",
            ]
        )
        found = False
        for slug, model in models.items():
            if not isinstance(model, dict):
                continue
            for row in model.get("metrics", []):
                if not isinstance(row, dict) or row.get("evaluator") != evaluator:
                    continue
                found = True
                method = model.get("method", slug)
                values = " | ".join(_format_value(row[name]) for name in METRICS)
                lines.append(
                    f"| [{method}](../model_zoo/{slug}.md) | {row['variant']} | "
                    f"{_format_value(row['samples'])} | {values} |"
                )
        if not found:
            raise ValueError(f"release manifest has no rows for {evaluator}")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the tracked page is stale.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    markdown = render(manifest)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != markdown:
            print(f"stale leaderboard: {output}", file=sys.stderr)
            return 1
        print(f"leaderboard is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
