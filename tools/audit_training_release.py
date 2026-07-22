#!/usr/bin/env python3
"""Fail when released training code contains private infrastructure details."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATHS = (
    ROOT / "configs/prism/train_prism.py",
    ROOT / "configs/tmr/train_tmr_smpl22.py",
    ROOT / "configs/hymotion_t2m/train_hymotion_t2m.py",
    ROOT / "motius/datasets/text_motion.py",
    ROOT / "motius/datasets/motion/tmr_text_motion_dataset.py",
    ROOT / "motius/models/prism/bundle.py",
    ROOT / "motius/models/tmr/bundle.py",
    ROOT / "motius/trainers/prism/prism_trainer.py",
    ROOT / "motius/trainers/tmr/tmr_trainer.py",
    ROOT / "motius/trainers/hymotion_t2m/hymotion_t2m_trainer.py",
    ROOT / "docs/training/prism_tmr_hymotion_t2m.md",
    ROOT / "tools/train.py",
)

FORBIDDEN = {
    "private filesystem root": re.compile(
        r"/(?:apdcephfs|cephfs|workspace|home)/[^\s'\"]+", re.IGNORECASE
    ),
    "cluster or job identifier": re.compile(
        r"(?:train_keyframe|taiji[_-]?exec|share_\d{5,}|8x8-\d{6,})",
        re.IGNORECASE,
    ),
    "private source label": re.compile(
        r"(?:motionhub|data_src|trigger_sources|Taobao)", re.IGNORECASE
    ),
    "remote data URI": re.compile(r"(?:s3|cos|hdfs|ceph)://", re.IGNORECASE),
    "credential": re.compile(
        r"(?:access[_-]?token|secret[_-]?key|private[_-]?token)\s*[:=]",
        re.IGNORECASE,
    ),
}


def audit() -> list[str]:
    findings: list[str] = []
    for path in RELEASE_PATHS:
        if not path.is_file():
            findings.append(f"missing release file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    return findings


def main() -> int:
    findings = audit()
    if findings:
        print("Training release privacy audit failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"Training release privacy audit passed ({len(RELEASE_PATHS)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
