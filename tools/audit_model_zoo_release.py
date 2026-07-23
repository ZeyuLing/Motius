#!/usr/bin/env python3
"""Audit Model Zoo cards for release completeness.

The audit is intentionally simple and conservative. A model is release-complete
only when its card has a visible demo reference, no pending evaluator rows, and
the Model Zoo entry points at reachable Hugging Face artifacts or another
explicitly published checkpoint source.
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
README = REPO_ROOT / "docs/model_zoo/README.md"
MODEL_ZOO_DIR = README.parent
MODEL_ENTRY_RE = re.compile(
    r"^- \*\*\[([^\]]+)\]\(([^)]+\.md)\)\*\* - (.*?)(?=^- \*\*\[|\n## |\Z)",
    re.MULTILINE | re.DOTALL,
)
HF_RE = re.compile(r"https://huggingface\.co/([^)\s|]+)")
CARD_TASK_RE = re.compile(r"^\| Tasks \| ([^|]+?) \|$", re.MULTILINE)
CARD_TASK_INLINE_RE = re.compile(r"^\*\*Tasks:\*\*\s*(.+?)\.?$", re.MULTILINE)
TASK_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)$")
TASK_LABELS = {
    "Text-to-Motion",
    "Motion-to-Text",
    "Temporal Condition",
    "Body-Part Condition",
    "Sequential Generation",
    "Motion Editing",
    "Music-to-Dance",
    "Dance-to-Music",
    "Speech-to-Gesture",
    "Kinematic Control",
    "Two-Person Text-to-Motion",
    "Robot Motion Control",
}
TASK_LEADERBOARDS = {
    "Text-to-Motion": "https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard",
    "Motion-to-Text": "https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard",
    "Temporal Condition": "https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard",
    "Body-Part Condition": "https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard",
    "Sequential Generation": "https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard",
    "Motion Editing": "https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard",
    "Music-to-Dance": "https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard",
    "Dance-to-Music": "https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard",
    "Speech-to-Gesture": "https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard",
    "Kinematic Control": "../tasks/README.md#kinematic-control",
    "Two-Person Text-to-Motion": "../tasks/README.md#two-person-text-to-motion",
    "Robot Motion Control": "../tasks/README.md#robot-motion-control",
}
TASK_METRIC_MARKERS = {
    "Text-to-Motion": (
        "HumanML3D Official",
        "MotionStreamer Evaluator",
        "Motius Joint-Position Evaluator",
    ),
    "Motion-to-Text": ("HumanML3D Motion-to-Text",),
    "Music-to-Dance": ("AIST++ Music-to-Dance",),
}


@dataclass
class ModelRow:
    method: str
    task_cell: str
    checkpoint_cell: str
    card_path: Path


def _read_model_rows() -> list[ModelRow]:
    rows = []
    for match in MODEL_ENTRY_RE.finditer(README.read_text()):
        method, relative_card, body = match.groups()
        task_cell, separator, resources = body.partition("·")
        if not separator:
            continue
        task_cell = " ".join(task_cell.split())
        rows.append(
            ModelRow(
                method,
                task_cell,
                resources,
                MODEL_ZOO_DIR / relative_card,
            )
        )
    return rows


def _parse_task_entries(cell: str) -> list[tuple[str, str | None]]:
    entries = []
    for value in cell.split(","):
        value = value.strip()
        if not value:
            continue
        match = TASK_LINK_RE.fullmatch(value)
        entries.append((match.group(1), match.group(2)) if match else (value, None))
    return entries


def _task_status(readme_cell: str, card_text: str) -> tuple[str, str]:
    readme_entries = _parse_task_entries(readme_cell)
    readme_labels = [label for label, _ in readme_entries]
    if not readme_labels:
        return "invalid", "README task field is empty"
    invalid = [label for label in readme_labels if label not in TASK_LABELS]
    if invalid:
        return "invalid", "unknown README tasks: " + ", ".join(invalid)
    for label, target in readme_entries:
        expected = TASK_LEADERBOARDS.get(label)
        if expected and target != expected:
            return "invalid", f"{label} must link to {expected}"
        if not expected and target:
            return "invalid", f"{label} links to an unregistered leaderboard"
        if (
            expected
            and not expected.startswith(("http://", "https://"))
            and not (README.parent / expected.split("#", 1)[0]).resolve().is_file()
        ):
            return "invalid", f"missing leaderboard target: {expected}"

    card_match = CARD_TASK_RE.search(card_text)
    inline_match = CARD_TASK_INLINE_RE.search(card_text)
    if not card_match and not inline_match:
        return "invalid", "model card has no Task/Tasks row"
    card_value = card_match.group(1) if card_match else inline_match.group(1)
    card_entries = _parse_task_entries(card_value)
    card_labels = [label for label, _ in card_entries]
    invalid = [label for label in card_labels if label not in TASK_LABELS]
    if invalid:
        return "invalid", "unknown model-card tasks: " + ", ".join(invalid)
    if any(target for _, target in card_entries):
        return "invalid", "model-card tasks must use portable plain-text labels"
    if card_labels != readme_labels:
        return "invalid", "README/model-card task mismatch"
    return "valid", ""


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
    interactive = re.findall(r"https?://[^)\s]+/cases/index\.html", card_text)
    if interactive:
        return "present", f"{len(set(interactive))} interactive all-case viewer(s)"
    if media:
        return "missing", f"only {len(media)} media refs"
    if "Validated" in card_text and "will be added" in card_text:
        return "missing", "validated demo missing"
    return "missing", "no demo media reference"


def _metric_status(card_text: str, task_cell: str) -> tuple[str, str]:
    section = card_text.split("## Evaluation Results", 1)
    if len(section) == 1:
        return "missing", "no Evaluation Results section"
    tail = section[1].split("\n## ", 1)[0]
    task_labels = [label for label, _ in _parse_task_entries(task_cell)]
    required = list(
        dict.fromkeys(
            marker
            for task in task_labels
            for marker in TASK_METRIC_MARKERS.get(task, ())
        )
    )
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
        "| Method | Tasks | Checkpoint | Demo | Metrics | Notes |",
        "| ------ | ----- | ---------- | ---- | ------- | ----- |",
    ]
    for row in rows:
        out.append(
            f"| {row['method']} | {row['tasks']} | {row['checkpoint']} | {row['demo']} | "
            f"{row['metrics']} | {row['notes']} |"
        )
    out.append("")
    return "\n".join(out)


def run(check_hf: bool) -> str:
    audit_rows = []
    for row in _read_model_rows():
        text = row.card_path.read_text() if row.card_path.exists() else ""
        tasks, task_note = _task_status(row.task_cell, text)
        checkpoint, checkpoint_note = _checkpoint_status(row.checkpoint_cell, check_hf)
        demo, demo_note = _demo_status(text)
        metrics, metric_note = _metric_status(text, row.task_cell)
        notes = "; ".join(
            note for note in [task_note, checkpoint_note, demo_note, metric_note] if note
        )
        audit_rows.append(
            {
                "method": row.method,
                "tasks": tasks,
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
