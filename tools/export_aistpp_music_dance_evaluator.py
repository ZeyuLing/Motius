#!/usr/bin/env python3
"""Package the AIST++ music-to-dance metric protocol for Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.evaluation.music_to_dance import AISTPPMusicDanceEvaluator  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.pkl")):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--joint-reference-embeddings", type=Path)
    parser.add_argument("--smpl-skeleton", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motions-zip", type=Path)
    source.add_argument("--motions-root", type=Path)
    parser.add_argument("--ignore-list", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        default="637d0aadf69e3e926ba70bfee9ff89571fd18813",
        help="Revision of the public AIST++ archive mirror used for the pool.",
    )
    parser.add_argument(
        "--motions-sha256",
        help="Previously verified source archive/tree hash; skips rehashing.",
    )
    parser.add_argument("--card", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = AISTPPMusicDanceEvaluator(
        physical=False,
        reference_feature_path=args.reference_features,
        joint_reference_embeddings=(
            np.load(args.joint_reference_embeddings)
            if args.joint_reference_embeddings is not None
            else None
        ),
    )
    evaluator.save_pretrained(args.output)
    shutil.copy2(args.smpl_skeleton, args.output / "aistpp_smpl24_skeleton.npz")
    shutil.copy2(args.calibration_report, args.output / "aistpp_smpl24_skeleton.json")
    model_dir = Path(__file__).resolve().parents[1] / "motius/models/bailando"
    for name in ("LICENSE", "ATTRIBUTIONS.md"):
        shutil.copy2(model_dir / name, args.output / name)
    if args.card is not None:
        shutil.copy2(args.card, args.output / "README.md")

    config_path = args.output / "evaluator_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    motion_source = args.motions_zip or args.motions_root
    motion_hash = args.motions_sha256 or (
        _sha256(args.motions_zip)
        if args.motions_zip is not None
        else _sha256_tree(args.motions_root)
    )
    config.update(
        {
            "reference_pool": "all 1,320 valid motion PKLs in AIST++ v1",
            "reference_pool_frames": "full sequence",
            "reference_pool_revision": args.source_revision,
            "reference_pool_mirror": "https://huggingface.co/datasets/yeok/danceba",
            "official_dataset": "https://google.github.io/aistplusplus_dataset/",
            "smpl24_skeleton": "aistpp_smpl24_skeleton.npz",
            "calibration_report": "aistpp_smpl24_skeleton.json",
            "source_artifacts": {
                str(motion_source.name): motion_hash,
                "ignore_list.txt": _sha256(args.ignore_list),
                "reference_features": _sha256(args.reference_features),
                **(
                    {
                        "joint_reference_embeddings": _sha256(
                            args.joint_reference_embeddings
                        )
                    }
                    if args.joint_reference_embeddings is not None
                    else {}
                ),
            },
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = {
        "output": str(args.output.resolve()),
        "num_reference_samples": config["num_reference_samples"],
        "source_artifacts": config["source_artifacts"],
    }
    (args.output / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
