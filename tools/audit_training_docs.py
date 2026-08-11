#!/usr/bin/env python3
"""Audit Motius training support, output paths, and public documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_HUB = ROOT / "docs" / "training" / "README.md"

RELEASES = {
    "gentrack": {
        "label": "GenTrack",
        "config": "configs/gentrack/train_gentrack_g1.py",
        "additional_configs": (
            "configs/gentrack/train_gentrack_offline_g1.py",
        ),
        "trainer": "motius/trainers/gentrack/__init__.py",
        "trainer_class": "GenTrackFlowGRPOTrainer",
        "card": "docs/training/gentrack.md",
        "output": "outputs/training/gentrack_g1",
    },
    "hymotion_t2m": {
        "label": "HY-Motion T2M",
        "config": "configs/hymotion_t2m/train_hymotion_t2m.py",
        "additional_configs": (
            "configs/hymotion_t2m/train_hymotion_g1.py",
        ),
        "trainer": "motius/trainers/hymotion_t2m/hymotion_t2m_trainer.py",
        "trainer_class": "HyMotionT2MTrainer",
        "card": "docs/model_zoo/hymotion_t2m.md",
        "output": "outputs/training/hymotion_t2m",
    },
    "prism": {
        "label": "PRISM",
        "config": "configs/prism/train_prism.py",
        "trainer": "motius/trainers/prism/prism_trainer.py",
        "trainer_class": "PrismTrainer",
        "card": "docs/model_zoo/prism.md",
        "output": "outputs/training/prism",
    },
    "motioncanvas": {
        "label": "MotionCanvas",
        "config": "configs/motioncanvas/train_motioncanvas_0p46b.py",
        "trainer": "motius/trainers/motioncanvas/sparse_rollout_join_trainer.py",
        "trainer_class": "MotionCanvasSparseRolloutJoinTrainer",
        "card": "docs/model_zoo/motioncanvas.md",
        "output": "outputs/training/motioncanvas_0p46b",
    },
    "tmr": {
        "label": "Motius Joint-Position Evaluator",
        "config": "configs/tmr/train_tmr_smpl22.py",
        "trainer": "motius/trainers/tmr/tmr_trainer.py",
        "trainer_class": "TMRTrainer",
        "card": "docs/evaluator_zoo/motius_joint_position.md",
        "output": "outputs/training/tmr_smpl22",
    },
    "protomotions": {
        "label": "ProtoMotions",
        "config": "configs/motion_tracking/protomotions_g1_bones_seed.yaml",
        "trainer": "motius/trainers/protomotions/train.py",
        "trainer_class": "ProtoMotionsTrainer",
        "card": "docs/model_zoo/protomotions.md",
        "output": "outputs/training/protomotions",
        "card_requirements": (
            "Precision",
            "Objective",
            "Checkpoints",
            "outputs/training/",
        ),
    },
    "sonic": {
        "label": "SONIC",
        "config": "configs/motion_tracking/sonic_g1_bones_seed.yaml",
        "trainer": "motius/trainers/sonic/train.py",
        "trainer_class": "SonicTrainer",
        "card": "docs/model_zoo/sonic.md",
        "output": "outputs/training/sonic",
        "card_requirements": (
            "Precision",
            "Objective",
            "Checkpoints",
            "outputs/training/",
        ),
    },
    "beyondmimic": {
        "label": "BeyondMimic",
        "config": "configs/motion_tracking/beyondmimic_g1_lafan1.yaml",
        "trainer": "motius/trainers/beyondmimic/train.py",
        "trainer_class": "BeyondMimicTrainer",
        "card": "docs/model_zoo/beyondmimic.md",
        "output": "outputs/training/beyondmimic",
        "card_requirements": (
            "Precision",
            "Objective",
            "Checkpoints",
            "--auto-resume",
            "outputs/training/",
        ),
    },
}

CARD_REQUIREMENTS = (
    "--auto-resume",
    "--load-scope full",
    "Precision",
    "Objective",
    "Checkpoints",
    "outputs/training/",
)


def audit() -> list[str]:
    errors: list[str] = []
    hub = TRAINING_HUB.read_text(encoding="utf-8")

    expected_packages = set(RELEASES)
    actual_packages = {
        path.name
        for path in (ROOT / "motius" / "trainers").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    if actual_packages != expected_packages:
        errors.append(
            "Trainer packages and documented releases differ: "
            f"actual={sorted(actual_packages)}, "
            f"documented={sorted(expected_packages)}"
        )

    expected_configs = {item["config"] for item in RELEASES.values()}
    for item in RELEASES.values():
        expected_configs.update(item.get("additional_configs", ()))
    actual_configs = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "configs").glob("*/train*.py")
    }
    actual_configs.update(
        str(path.relative_to(ROOT))
        for path in (ROOT / "configs" / "motion_tracking").glob("*.yaml")
    )
    if actual_configs != expected_configs:
        errors.append(
            "Training configs and documented releases differ: "
            f"actual={sorted(actual_configs)}, "
            f"documented={sorted(expected_configs)}"
        )

    for package, item in RELEASES.items():
        config_path = ROOT / item["config"]
        trainer_path = ROOT / item["trainer"]
        card_path = ROOT / item["card"]
        for path in (config_path, trainer_path, card_path):
            if not path.is_file():
                errors.append(f"{package}: missing {path.relative_to(ROOT)}")

        if not all(path.is_file() for path in (config_path, trainer_path, card_path)):
            continue

        config_text = config_path.read_text(encoding="utf-8")
        output_literals = {
            item["output"],
            f'"{item["output"]}"',
            f"'{item['output']}'",
        }
        if not any(literal in config_text for literal in output_literals):
            errors.append(
                f"{package}: config does not declare work_dir {item['output']!r}"
            )

        trainer_text = trainer_path.read_text(encoding="utf-8")
        if not re.search(
            rf"class\s+{re.escape(item['trainer_class'])}\s*(?:\(|:)",
            trainer_text,
        ):
            errors.append(
                f"{package}: trainer class {item['trainer_class']} not found"
            )

        card = card_path.read_text(encoding="utf-8")
        if not re.search(r"^#{2,3} Training$", card, flags=re.MULTILINE):
            errors.append(f"{package}: card missing a Training section")
        for requirement in item.get("card_requirements", CARD_REQUIREMENTS):
            if requirement not in card:
                errors.append(f"{package}: card missing {requirement!r}")
        if item["config"] not in card:
            errors.append(f"{package}: card does not link {item['config']}")
        for additional in item.get("additional_configs", ()):
            if additional not in card:
                errors.append(f"{package}: card does not link {additional}")
        if item["output"] not in card:
            errors.append(f"{package}: card does not declare {item['output']}")

        for value in (
            item["label"],
            item["config"],
            item["trainer_class"],
            item["output"],
            *item.get("additional_configs", ()),
        ):
            if value not in hub:
                errors.append(f"{package}: Training Hub missing {value!r}")

    public_runtime_files = [
        ROOT / "configs" / "_base_" / "default_runtime.py",
        ROOT / "tools" / "train.py",
        *(ROOT / item["config"] for item in RELEASES.values()),
        *(
            ROOT / config
            for item in RELEASES.values()
            for config in item.get("additional_configs", ())
        ),
    ]
    for path in public_runtime_files:
        text = path.read_text(encoding="utf-8")
        if "work_dirs/" in text:
            errors.append(
                f"{path.relative_to(ROOT)}: public training output uses work_dirs/"
            )

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_zoo = (ROOT / "docs" / "model_zoo" / "README.md").read_text(
        encoding="utf-8"
    )
    if "docs/training/README.md" not in root_readme:
        errors.append("Root README does not link the Training Hub")
    if "../training/README.md" not in model_zoo:
        errors.append("Model Zoo does not link the Training Hub")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = audit()
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Training documentation audit failed with {len(errors)} error(s).")
        return 1
    print(f"Training documentation audit passed ({len(RELEASES)} releases).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
