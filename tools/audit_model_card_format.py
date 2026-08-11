#!/usr/bin/env python3
"""Audit the public Model Zoo card information contract."""

from __future__ import annotations

from pathlib import Path
import re

try:
    from tools.hf_checkpoint_specs import CHECKPOINT_SPECS
    from tools.normalize_model_cards import (
        FOOTER_END,
        FOOTER_START,
        MODEL_ZOO_README,
        NAV_END,
        NAV_START,
        REQUIRED_SECTIONS,
        TASKS_END,
        TASKS_START,
        _catalog_cards,
        _task_contracts,
        normalize_all,
    )
except ModuleNotFoundError:
    from hf_checkpoint_specs import CHECKPOINT_SPECS
    from normalize_model_cards import (
        FOOTER_END,
        FOOTER_START,
        MODEL_ZOO_README,
        NAV_END,
        NAV_START,
        REQUIRED_SECTIONS,
        TASKS_END,
        TASKS_START,
        _catalog_cards,
        _task_contracts,
        normalize_all,
    )


MARKERS = (
    NAV_START,
    NAV_END,
    TASKS_START,
    TASKS_END,
    FOOTER_START,
    FOOTER_END,
)

USER_CHECKPOINT_PACKAGES = {"beyondmimic"}


def _checkpoint_contracts() -> dict[str, list]:
    contracts: dict[str, list] = {}
    for spec in CHECKPOINT_SPECS:
        package = spec.pipeline_class.split(".")[-2]
        contracts.setdefault(package, []).append(spec)
    return contracts


def _relative_link_errors(path: Path, text: str) -> list[str]:
    errors = []
    targets = re.findall(
        r"\[[^\]]*\]\(([^)]+)\)|<a href=\"([^\"]+)\"",
        text,
    )
    for matches in targets:
        target = next(item for item in matches if item)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        relative = target.split("#", 1)[0]
        if relative and not (path.parent / relative).resolve().exists():
            errors.append(f"{path.name}: broken relative link {target}")
    return errors


def audit_model_cards() -> list[str]:
    """Return actionable format errors for every catalog card."""
    errors: list[str] = []
    cards = _catalog_cards()
    tasks = _task_contracts()
    checkpoints = _checkpoint_contracts()
    drift = set(normalize_all(write=False))

    for package, path in sorted(cards.items()):
        text = path.read_text(encoding="utf-8")
        label = path.name

        if path in drift:
            errors.append(f"{label}: requires normalization")

        h1s = re.findall(
            r'^<h1 align="center">[^<]+ Model Card</h1>$',
            text,
            flags=re.MULTILINE,
        )
        if len(h1s) != 1:
            errors.append(f"{label}: expected one centered Model Card title")

        intro = text.split(NAV_START, 1)[0]
        if not re.search(
            r'<p align="center">\s*<strong>.+?</strong>\s*</p>',
            intro,
            flags=re.DOTALL,
        ):
            errors.append(f"{label}: missing centered capability summary")
        if len(re.findall(r'<p align="center">', intro)) < 2:
            errors.append(f"{label}: missing centered resource link bar")

        headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
        if tuple(headings) != REQUIRED_SECTIONS:
            errors.append(
                f"{label}: level-two sections do not match the canonical order"
            )

        for marker in MARKERS:
            if text.count(marker) != 1:
                errors.append(f"{label}: marker {marker} must appear once")

        for section in REQUIRED_SECTIONS:
            match = re.search(
                rf"^## {re.escape(section)}\n(.*?)(?=^## |\Z)",
                text,
                flags=re.MULTILINE | re.DOTALL,
            )
            if match is None or len(match.group(1).strip()) < 20:
                errors.append(f"{label}: {section} is empty")

        for task in tasks[package]:
            method = f"infer_{task}"
            if method not in text:
                errors.append(f"{label}: missing task API {method}")

        if package == "prompthmr":
            if "Restricted upstream runtime" not in text:
                errors.append(f"{label}: missing restricted-runtime disclosure")
            if "Pipeline.from_pretrained" not in text:
                errors.append(f"{label}: missing loader availability disclosure")
        elif package in USER_CHECKPOINT_PACKAGES:
            if "from motius import Pipeline" not in text:
                errors.append(f"{label}: missing unified loader example")
            if "Public pretrained checkpoint | Not released upstream" not in text:
                errors.append(
                    f"{label}: missing public-checkpoint availability disclosure"
                )
            if "tools/export_motion_tracking_hf.py" not in text:
                errors.append(f"{label}: missing local artifact export instructions")
        else:
            if "from motius import Pipeline" not in text:
                errors.append(f"{label}: missing unified loader example")
            if package not in checkpoints:
                errors.append(f"{label}: no registered checkpoint artifact")
            for spec in checkpoints.get(package, []):
                if spec.target_repo not in text:
                    errors.append(
                        f"{label}: missing checkpoint {spec.target_repo}"
                    )

        if package == "vermo":
            if "No external paper or original repository" not in text:
                errors.append(f"{label}: missing Motius-native provenance")
        else:
            if "Paper" not in intro:
                errors.append(f"{label}: missing paper link")
            if "github.com/" not in intro:
                errors.append(f"{label}: missing upstream source link")
        errors.extend(_relative_link_errors(path, text))

    errors.extend(
        _relative_link_errors(
            MODEL_ZOO_README,
            MODEL_ZOO_README.read_text(encoding="utf-8"),
        )
    )
    return errors


def main() -> int:
    errors = audit_model_cards()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"{len(errors)} Model Card format error(s)")
        return 1
    print(f"{len(_catalog_cards())} Model Cards satisfy the v2 contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
