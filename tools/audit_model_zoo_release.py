#!/usr/bin/env python3
"""Audit Model Zoo cards for release completeness.

The audit is intentionally simple and conservative. A model is release-complete
only when its card has a visible demo reference, no pending evaluator rows, and
the Model Zoo entry points at reachable Hugging Face artifacts or another
explicitly published checkpoint source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "docs/model_zoo/README.md"
MODEL_ZOO_DIR = README.parent
TASK_REGISTRY_PATH = REPO_ROOT / "docs/tasks/taxonomy.json"
TASK_REGISTRY = json.loads(TASK_REGISTRY_PATH.read_text(encoding="utf-8"))
MODEL_TABLE_METHOD_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+\.md)\)$")
HF_RE = re.compile(r"https://huggingface\.co/([^)\s|]+)")
CARD_TASK_RE = re.compile(r"^\| Tasks \| ([^|]+?) \|$", re.MULTILINE)
CARD_TASK_INLINE_RE = re.compile(r"^\*\*Tasks:\*\*\s*(.+?)\.?$", re.MULTILINE)
CARD_TASK_STATUS_RE = re.compile(r"^\| Task status \| ([^|]+?) \|$", re.MULTILINE)
TASK_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)$")
DEMO_START = "<!-- MOTIUS_TASK_DEMOS:START -->"
DEMO_END = "<!-- MOTIUS_TASK_DEMOS:END -->"
METRICS_START = "<!-- MOTIUS_CANONICAL_METRICS:START -->"
METRICS_END = "<!-- MOTIUS_CANONICAL_METRICS:END -->"
UNREGISTERED_TASK_CELL = "**Not registered**"
RESTRICTED_TASK_CELL = "**Restricted upstream runtime**"
TASK_LABELS = {task["label"] for task in TASK_REGISTRY["tasks"]}
TASK_LEADERBOARDS = {
    task["label"]: task["model_zoo_target"] for task in TASK_REGISTRY["tasks"]
}
RUNTIME_ONLY_TASK_LABELS: set[str] = set()
USER_CHECKPOINT_METHODS = {"BeyondMimic"}
TASK_METRIC_MARKERS = {
    "Text-to-Motion": (
        "HumanML3D Official",
        "MotionStreamer Evaluator",
        "Motius Joint-Position Evaluator",
    ),
    "Motion-to-Text": ("Canonical Motion-to-Text Snapshot",),
    "Music-to-Dance": ("Canonical Music-to-Dance Snapshot",),
    "Dance-to-Music": ("Canonical Dance-to-Music Snapshot",),
    "Temporal Motion Completion": (
        "Canonical Temporal-Completion Snapshot",
    ),
    "Motion Repair": ("Canonical Motion-Repair Snapshot",),
    "Monocular Motion Capture": (
        "Canonical Monocular-Capture Snapshot",
    ),
    "Motion Tracking": (
        "Measured physical rollout",
        "Success ↑",
        "Completion ↑",
        "Local MPJPE ↓",
        "Joint MAE ↓",
    ),
}


@dataclass
class ModelRow:
    method: str
    task_cell: str
    checkpoint_cell: str
    card_path: Path


def _read_model_rows() -> list[ModelRow]:
    rows = []
    catalog = README.read_text(encoding="utf-8").split("## Method Catalog", 1)[1]
    catalog = catalog.split("\n## ", 1)[0]
    for line in catalog.splitlines():
        if not line.startswith("| ["):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        method_match = MODEL_TABLE_METHOD_RE.fullmatch(cells[0])
        if not method_match:
            continue
        method, relative_card = method_match.groups()
        rows.append(
            ModelRow(
                method,
                cells[1],
                cells[3],
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
    if readme_cell == UNREGISTERED_TASK_CELL:
        status_match = CARD_TASK_STATUS_RE.search(card_text)
        if not status_match or status_match.group(1) != "Not registered":
            return "invalid", "model card must declare unregistered task status"
        return "unregistered", "no canonical Motius task contract"
    if readme_cell == RESTRICTED_TASK_CELL:
        status_match = CARD_TASK_STATUS_RE.search(card_text)
        if not status_match or status_match.group(1) != "Restricted upstream runtime":
            return "invalid", "model card must declare restricted upstream runtime"
        return "restricted", "upstream license prevents a self-contained release"

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
            publication_withheld = (
                target is None
                and "not listed on the public" in card_text.lower()
                and "leaderboard yet" in card_text.lower()
            )
            if not publication_withheld:
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


def _demo_status(
    card_text: str,
    task_cell: str,
) -> tuple[str, str]:
    match = re.search(
        rf"{re.escape(DEMO_START)}(.*?){re.escape(DEMO_END)}",
        card_text,
        flags=re.DOTALL,
    )
    if not match:
        return "missing", "no generated task-demo block"
    block = match.group(1)
    if "Demo not published" in block:
        return "missing", "contains an unpublished task demo"

    rows = []
    for line in block.splitlines():
        if not line.startswith("| ") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] != "Task":
            rows.append(cells)
    expected = (
        ["No registered public task"]
        if task_cell == UNREGISTERED_TASK_CELL
        else [label for label, _ in _parse_task_entries(task_cell)]
    )
    labels = [row[0] for row in rows]
    if labels != expected:
        return "missing", (
            "task-demo rows do not match public tasks: "
            f"expected {expected}, found {labels}"
        )
    invalid_media = []
    for row in rows:
        if row[0] == "No registered public task":
            continue
        if len(row) < 4:
            invalid_media.append(f"{row[0]} (malformed row)")
            continue
        match = re.search(
            r'<video\s+[^>]*src="'
            r"https://github\.com/user-attachments/assets/"
            r'[0-9a-fA-F-]{36}"[^>]*controls[^>]*>',
            row[2],
        )
        if match is None:
            invalid_media.append(f"{row[0]} (no GitHub video player)")
    if invalid_media:
        if (
            set(expected) <= RUNTIME_ONLY_TASK_LABELS
            and "pending a shared simulator protocol" in block
        ):
            return "not applicable", "shared physical protocol not registered"
        return "missing", "invalid task media: " + ", ".join(invalid_media)
    return "present", f"{len(rows)} task-specific demo row(s)"


def _metric_status(card_text: str, task_cell: str) -> tuple[str, str]:
    section = card_text.split("## Evaluation Results", 1)
    if len(section) == 1:
        return "missing", "no Evaluation Results section"
    tail = section[1].split("\n## ", 1)[0]
    if METRICS_START not in tail or METRICS_END not in tail:
        return "missing", "no canonical metric block"
    task_labels = [label for label, _ in _parse_task_entries(task_cell)]
    if task_labels and set(task_labels) <= RUNTIME_ONLY_TASK_LABELS:
        if "Runtime verification" not in tail:
            return "missing", "runtime verification disclosure is missing"
        return "not applicable", "shared physical protocol not registered"
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
    if re.search(r"\b(?:Pending|Not measured|not measured)\b", tail):
        return "incomplete", "contains explicitly unmeasured rows"
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
        text = (
            row.card_path.read_text(encoding="utf-8")
            if row.card_path.exists()
            else ""
        )
        tasks, task_note = _task_status(row.task_cell, text)
        if tasks == "restricted":
            checkpoint, checkpoint_note = (
                "restricted",
                "official distribution only",
            )
            demo, demo_note = "not applicable", ""
        elif row.method in USER_CHECKPOINT_METHODS:
            checkpoint, checkpoint_note = (
                "user-supplied",
                "official exporter; no upstream pretrained policy",
            )
            demo, demo_note = _demo_status(text, row.task_cell)
        else:
            checkpoint, checkpoint_note = _checkpoint_status(
                row.checkpoint_cell,
                check_hf,
            )
            demo, demo_note = _demo_status(
                text,
                row.task_cell,
            )
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
        out.write_text(markdown, encoding="utf-8", newline="\n")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
