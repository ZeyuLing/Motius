#!/usr/bin/env python3
"""Audit Model Zoo cards for release completeness.

The audit is intentionally simple and conservative. A model is release-complete
only when its card has a visible demo reference, no pending evaluator rows, and
the README checkpoint cell points at reachable Hugging Face artifacts or another
explicitly published URL.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
MODEL_CARD_RE = re.compile(r"\[Model Card\]\((docs/model_zoo/[^)]+)\)")
HF_RE = re.compile(r"https://huggingface\.co/([^)\s|]+)")


@dataclass
class ModelRow:
    method: str
    checkpoint_cell: str
    card_path: Path


def _split_table_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _read_model_rows() -> list[ModelRow]:
    rows: list[ModelRow] = []
    in_table = False
    for line in README.read_text().splitlines():
        if line.startswith("| Method | Task | Motion Rep. | Checkpoint | Card | References |"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|"):
            continue
        if line.startswith("| ------"):
            continue
        cells = _split_table_row(line)
        if len(cells) < 5:
            continue
        match = MODEL_CARD_RE.search(cells[4])
        if not match:
            continue
        rows.append(ModelRow(cells[0], cells[3], REPO_ROOT / match.group(1)))
    return rows


def _hf_repo_exists(repo_id: str, timeout: int = 20) -> bool:
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def _checkpoint_status(cell: str, check_hf: bool) -> tuple[str, str]:
    if "Not released yet" in cell or "Pending" in cell:
        return "missing", "not released"
    repos = HF_RE.findall(cell)
    if not repos:
        return "missing", "no HF link"
    if not check_hf:
        return "present", ", ".join(repos)
    missing = [repo for repo in repos if not _hf_repo_exists(repo)]
    if missing:
        return "missing", "missing HF: " + ", ".join(missing)
    return "present", ", ".join(repos)


def _demo_status(card_text: str) -> tuple[str, str]:
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", card_text)
    refs.extend(re.findall(r"<img\s+[^>]*src=[\"']([^\"']+)[\"']", card_text))
    media = [ref for ref in refs if ref.lower().endswith((".gif", ".mp4", ".png", ".jpg", ".jpeg", ".webp"))]
    if len(media) >= 3:
        return "present", f"{len(media)} media refs"
    if media:
        return "missing", f"only {len(media)} media refs"
    if "Validated" in card_text and "will be added" in card_text:
        return "missing", "validated demo missing"
    return "missing", "no demo media reference"


def _metric_status(card_text: str) -> tuple[str, str]:
    section = card_text.split("## Evaluation Results", 1)
    if len(section) == 1:
        return "missing", "no Evaluation Results section"
    tail = section[1].split("\n## ", 1)[0]
    required = ["HumanML3D Official", "MotionStreamer Evaluator", "Motius Joint-Position Evaluator"]
    missing_required = [name for name in required if name not in tail]
    if missing_required:
        return "missing", "missing rows: " + ", ".join(missing_required)
    if "Pending" in tail:
        return "incomplete", "contains Pending rows"
    return "complete", "all required rows measured"


def _format_markdown(rows: Iterable[dict[str, str]]) -> str:
    out = [
        "# Model Zoo Release Audit",
        "",
        "| Method | Checkpoint | Demo | Metrics | Notes |",
        "| ------ | ---------- | ---- | ------- | ----- |",
    ]
    for row in rows:
        out.append(
            f"| {row['method']} | {row['checkpoint']} | {row['demo']} | "
            f"{row['metrics']} | {row['notes']} |"
        )
    out.append("")
    return "\n".join(out)


def run(check_hf: bool) -> str:
    audit_rows = []
    for row in _read_model_rows():
        text = row.card_path.read_text() if row.card_path.exists() else ""
        checkpoint, checkpoint_note = _checkpoint_status(row.checkpoint_cell, check_hf)
        demo, demo_note = _demo_status(text)
        metrics, metric_note = _metric_status(text)
        notes = "; ".join(note for note in [checkpoint_note, demo_note, metric_note] if note)
        audit_rows.append(
            {
                "method": row.method,
                "checkpoint": checkpoint,
                "demo": demo,
                "metrics": metrics,
                "notes": notes,
            }
        )
    return _format_markdown(audit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-hf", action="store_true", help="Verify Hugging Face model URLs over the network.")
    parser.add_argument("--output", type=Path, help="Optional markdown output path. Use outputs/ for generated audits.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    markdown = run(check_hf=args.check_hf)
    if args.output:
        out = args.output
        if not out.is_absolute():
            out = REPO_ROOT / out
        try:
            out.relative_to(REPO_ROOT / "outputs")
        except ValueError:
            print("error: generated audit outputs must be written under outputs/", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
