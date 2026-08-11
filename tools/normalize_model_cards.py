#!/usr/bin/env python3
"""Normalize public Model Zoo cards to the Motius v2 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from tools.hf_checkpoint_specs import CHECKPOINT_SPECS
except ModuleNotFoundError:
    from hf_checkpoint_specs import CHECKPOINT_SPECS


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZOO_DIR = REPO_ROOT / "docs" / "model_zoo"
MODEL_ZOO_README = MODEL_ZOO_DIR / "README.md"
TASK_REGISTRY = json.loads(
    (REPO_ROOT / "docs" / "tasks" / "taxonomy.json").read_text(
        encoding="utf-8"
    )
)

HEADING_ALIASES = {
    "Preview": "Visual Results",
    "Demo": "Visual Results",
    "Release Snapshot": "Model Overview",
    "Supported Tasks": "Model Overview",
    "Integration Status": "Model Overview",
    "Native Control Contract": "Model Overview",
    "Usage": "Quick Start",
    "Evaluation": "Evaluation Results",
    "M2T Evaluation": "Evaluation Results",
    "Representation Notes": "Motion Representation",
    "Motius Components": "Implementation Notes",
    "Citation": "Citation and License",
    "License": "Citation and License",
    "License And Attribution": "Citation and License",
    "Attribution And License": "Citation and License",
    "Provenance": "Citation and License",
}

REQUIRED_SECTIONS = (
    "Visual Results",
    "Model Overview",
    "Quick Start",
    "Evaluation Results",
    "Motion Representation",
    "Citation and License",
)

NESTED_SECTION_PARENTS = {
    "Capability Boundary": "Model Overview",
    "Checkpoint": "Model Overview",
    "Checkpoints": "Model Overview",
    "Control Contract": "Model Overview",
    "Implementation Notes": "Model Overview",
    "Validation Status": "Model Overview",
    "Installation": "Quick Start",
    "Exact Parity": "Evaluation Results",
    "Reproduction": "Evaluation Results",
    "Reproduction Audit": "Evaluation Results",
    "Reproduction Check": "Evaluation Results",
    "Stage Parity": "Evaluation Results",
    "Verification": "Evaluation Results",
}

TASK_LABEL_OVERRIDES = {
    "g1_qpos_generation": "G1 Qpos Generation",
    "g1_realtime_navigation": "G1 Realtime Navigation",
}

TASK_TARGET_OVERRIDES = {
    ("beyondmimic", "motion_tracking"): (
        "https://huggingface.co/spaces/ZeyuLing/"
        "motion-tracking-isaaclab-leaderboard"
    ),
    ("sonic", "motion_tracking"): (
        "https://huggingface.co/spaces/ZeyuLing/"
        "motion-tracking-isaaclab-leaderboard"
    ),
}

NAV_START = "<!-- MOTIUS_MODEL_CARD_NAV:START -->"
NAV_END = "<!-- MOTIUS_MODEL_CARD_NAV:END -->"
TASKS_START = "<!-- MOTIUS_MODEL_CARD_TASKS:START -->"
TASKS_END = "<!-- MOTIUS_MODEL_CARD_TASKS:END -->"
FOOTER_START = "<!-- MOTIUS_MODEL_CARD_FOOTER:START -->"
FOOTER_END = "<!-- MOTIUS_MODEL_CARD_FOOTER:END -->"

NAV_BLOCK = f"""\
{NAV_START}
<p align="center">
  <a href="#visual-results">Visual Results</a> ·
  <a href="#model-overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#motion-representation">Motion Representation</a>
</p>
{NAV_END}
"""

FOOTER_BLOCK = f"""\
{FOOTER_START}
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
{FOOTER_END}
"""


def _catalog_rows() -> dict[str, tuple[Path, str]]:
    catalog = MODEL_ZOO_README.read_text(encoding="utf-8").split(
        "## Method Catalog",
        1,
    )[1]
    catalog = catalog.split("\n## ", 1)[0]
    rows = {}
    for line in catalog.splitlines():
        match = re.match(
            r"^\| \[([^\]]+)\]\(([^)]+\.md)\) \| (.*?) \|",
            line,
        )
        if match:
            _, relative_path, task_cell = match.groups()
            rows[Path(relative_path).stem] = (
                MODEL_ZOO_DIR / relative_path,
                task_cell,
            )
    return rows


def _catalog_cards() -> dict[str, Path]:
    return {
        package: path
        for package, (path, _) in _catalog_rows().items()
    }


def _catalog_task_contracts() -> dict[str, list[str]]:
    label_to_id = {
        task["label"]: task["id"] for task in TASK_REGISTRY["tasks"]
    }
    contracts: dict[str, list[str]] = {}
    for package, (_, task_cell) in _catalog_rows().items():
        if task_cell.startswith("**"):
            contracts[package] = []
            continue
        tasks = []
        for value in task_cell.split(","):
            value = value.strip()
            match = re.fullmatch(r"\[([^\]]+)\]\([^)]+\)", value)
            label = match.group(1) if match else value
            if not label:
                continue
            task = label_to_id.get(label)
            if task is None:
                raise ValueError(
                    f"Unknown Model Zoo task {label!r} for {package}"
                )
            tasks.append(task)
        contracts[package] = tasks
    return contracts


def _task_contracts() -> dict[str, list[str]]:
    """Return the public task contract shown in the Model Zoo.

    Artifact metadata may expose auxiliary APIs that are intentionally not
    registered as public benchmark tasks. Model Cards must follow the public
    catalog so each advertised task can be paired with task-specific metrics
    and visual evidence.
    """
    return _catalog_task_contracts()


def _task_metadata() -> dict[str, tuple[str, str]]:
    return {
        task["id"]: (task["label"], task["model_zoo_target"])
        for task in TASK_REGISTRY["tasks"]
    }


def _task_table(package: str, tasks: list[str]) -> str:
    metadata = _task_metadata()
    rows = [
        TASKS_START,
        "",
        "### Task APIs",
        "",
        "| Task | Pipeline API | Evaluation and examples |",
        "| --- | --- | --- |",
    ]
    if not tasks:
        rows.append(
            "| Not registered | No canonical task API | "
            "See the capability boundary below |"
        )
    for task in tasks:
        label, target = metadata.get(
            task,
            (
                TASK_LABEL_OVERRIDES.get(
                    task,
                    task.replace("_", " ").title(),
                ),
                "",
            ),
        )
        target = TASK_TARGET_OVERRIDES.get((package, task), target)
        if package == "prompthmr":
            evidence = "Restricted official-runtime protocol"
        elif task == "motion_tracking":
            evidence = f"[Task and runtime contract]({target})"
        elif target:
            evidence = f"[Benchmark and examples]({target})"
        else:
            evidence = "No public benchmark registered"
        rows.append(f"| {label} | `infer_{task}` | {evidence} |")
    rows.extend(["", TASKS_END])
    return "\n".join(rows)


def _replace_marked_block(
    text: str,
    start: str,
    end: str,
    replacement: str,
) -> str:
    pattern = re.compile(
        rf"{re.escape(start)}.*?{re.escape(end)}\n?",
        re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(replacement.rstrip() + "\n", text, count=1)
    return text


def _normalize_sections(text: str, path: Path) -> str:
    """Keep six stable card sections and nest method-specific material."""
    matches = list(re.finditer(r"^## (.+)\n", text, flags=re.MULTILINE))
    if not matches:
        raise ValueError(f"{path} has no level-two section")

    intro = text[: matches[0].start()].rstrip()
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        sections.append(
            (
                match.group(1),
                text[match.end() : end].strip(),
            )
        )

    present = {title for title, _ in sections}
    missing = set(REQUIRED_SECTIONS) - present
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required section(s): {joined}")

    grouped: dict[str, list[str]] = {title: [] for title in REQUIRED_SECTIONS}
    current: str | None = None
    for title, body in sections:
        if title in grouped:
            current = title
            if body:
                grouped[current].append(body)
            continue
        if current is None:
            raise ValueError(
                f"{path} has '{title}' before a required section"
            )
        nested = f"### {title}"
        if body:
            nested = f"{nested}\n\n{body}"
        grouped[current].append(nested)

    bodies = {
        title: "\n\n".join(grouped[title]).strip()
        for title in REQUIRED_SECTIONS
    }
    moved: dict[str, list[str]] = {title: [] for title in REQUIRED_SECTIONS}
    for source in REQUIRED_SECTIONS:
        body = bodies[source]
        subsections = list(
            re.finditer(r"^### (.+)\n", body, flags=re.MULTILINE)
        )
        removals: list[tuple[int, int]] = []
        for index, match in enumerate(subsections):
            title = match.group(1)
            target = NESTED_SECTION_PARENTS.get(title)
            if target is None or target == source:
                continue
            end = (
                subsections[index + 1].start()
                if index + 1 < len(subsections)
                else len(body)
            )
            moved[target].append(body[match.start() : end].strip())
            removals.append((match.start(), end))
        for start, end in reversed(removals):
            body = body[:start].rstrip() + "\n\n" + body[end:].lstrip()
        bodies[source] = body.strip()

    for target, blocks_to_move in moved.items():
        if not blocks_to_move:
            continue
        addition = "\n\n".join(blocks_to_move)
        if target == "Quick Start" and addition.startswith("### Installation"):
            bodies[target] = f"{addition}\n\n{bodies[target]}".strip()
        else:
            bodies[target] = f"{bodies[target]}\n\n{addition}".strip()

    blocks = [intro]
    for title in REQUIRED_SECTIONS:
        body = bodies[title]
        block = f"## {title}"
        if body:
            block += f"\n\n{body}"
        blocks.append(block)
    return "\n\n".join(blocks).rstrip() + "\n"


def normalize_card(path: Path, package: str, tasks: list[str]) -> str:
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        match = re.fullmatch(r"(##) (.+)", line)
        if match:
            heading = HEADING_ALIASES.get(match.group(2), match.group(2))
            line = f"## {heading}"
        if line in {
            "## BABEL Sequential Results",
            "## TP2M Results",
            "## Temporal Motion Completion · HumanML3D",
            "## Part-Level Motion Control · HumanML3D",
        }:
            line = f"#{line}"
        lines.append(line)
    text = "\n".join(lines).rstrip() + "\n"

    text = _replace_marked_block(text, NAV_START, NAV_END, NAV_BLOCK)
    if NAV_START not in text:
        first_section = re.search(r"^## ", text, flags=re.MULTILINE)
        if first_section is None:
            raise ValueError(f"{path} has no level-two section")
        text = (
            text[: first_section.start()].rstrip()
            + "\n\n"
            + NAV_BLOCK
            + "\n"
            + text[first_section.start() :]
        )
    intro, remainder = text.split(NAV_START, 1)
    text = intro.replace("</a> |", "</a> ·") + NAV_START + remainder

    task_block = _task_table(package, tasks)
    text = _replace_marked_block(text, TASKS_START, TASKS_END, task_block)
    if TASKS_START not in text:
        heading = "## Model Overview\n"
        if heading not in text:
            raise ValueError(f"{path} has no Model Overview section")
        text = text.replace(
            heading,
            heading + "\n" + task_block + "\n\n",
            1,
        )

    text = _replace_marked_block(
        text,
        FOOTER_START,
        FOOTER_END,
        FOOTER_BLOCK,
    )
    if FOOTER_START not in text:
        text = text.rstrip() + "\n\n" + FOOTER_BLOCK
    return _normalize_sections(text, path)


def normalize_all(write: bool) -> list[Path]:
    cards = _catalog_cards()
    contracts = _task_contracts()
    changed = []
    for package, path in sorted(cards.items()):
        tasks = contracts.get(package)
        if tasks is None:
            raise ValueError(f"No task contract registered for {package}")
        normalized = normalize_card(path, package, tasks)
        if normalized != path.read_text(encoding="utf-8"):
            changed.append(path)
            if write:
                path.write_text(normalized, encoding="utf-8")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite cards. Without this flag, only report drift.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = normalize_all(write=args.write)
    for path in changed:
        print(path.relative_to(REPO_ROOT))
    if changed and not args.write:
        print(f"{len(changed)} card(s) require normalization")
        return 1
    print(f"{len(changed)} card(s) normalized" if args.write else "All cards normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
