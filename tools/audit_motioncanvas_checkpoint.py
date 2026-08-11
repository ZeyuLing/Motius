#!/usr/bin/env python3
"""Verify bit-exact MotionCanvas source-to-Hugging-Face conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


LEARNED_TENSORS = (
    "null_vtxt_feat",
    "null_ctxt_input",
    "special_game_vtxt_feat",
    "special_game_ctxt_feat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--bone-offsets", type=Path, required=True)
    return parser.parse_args()


def _assert_equal(name: str, source: torch.Tensor, target: torch.Tensor) -> None:
    source = source.detach().cpu()
    target = target.detach().cpu()
    if source.dtype != target.dtype:
        raise ValueError(
            f"{name}: dtype differs ({source.dtype} != {target.dtype})"
        )
    if source.shape != target.shape:
        raise ValueError(
            f"{name}: shape differs ({source.shape} != {target.shape})"
        )
    if not torch.equal(source, target):
        difference = (source.float() - target.float()).abs().max().item()
        raise ValueError(f"{name}: values differ (max abs {difference})")


def audit(args: argparse.Namespace) -> dict:
    source_weights = args.source / "model.safetensors"
    source_custom = args.source / "custom_checkpoint_0.pkl"
    artifact_weights = args.artifact / "motion_transformer.safetensors"
    for path in (source_weights, source_custom, artifact_weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    with safe_open(str(source_weights), framework="pt", device="cpu") as src:
        source_keys = set(src.keys())
        with safe_open(
            str(artifact_weights),
            framework="pt",
            device="cpu",
        ) as dst:
            artifact_keys = set(dst.keys())
            expected = {
                f"motion_transformer.{key}" for key in source_keys
            } | set(LEARNED_TENSORS)
            if artifact_keys != expected:
                missing = sorted(expected - artifact_keys)
                unexpected = sorted(artifact_keys - expected)
                raise ValueError(
                    "artifact key set differs: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for key in sorted(source_keys):
                _assert_equal(
                    key,
                    src.get_tensor(key),
                    dst.get_tensor(f"motion_transformer.{key}"),
                )

            learned = torch.load(
                source_custom,
                map_location="cpu",
                weights_only=True,
            )
            for key in LEARNED_TENSORS:
                if key not in learned:
                    raise ValueError(f"{source_custom}: missing {key}")
                _assert_equal(key, learned[key], dst.get_tensor(key))

    for name in ("Mean.npy", "Std.npy"):
        source = np.load(args.stats_dir / name)
        target = np.load(args.artifact / name)
        if not np.array_equal(source, target):
            raise ValueError(f"{name}: artifact statistics differ")

    source_offsets = torch.load(
        args.bone_offsets,
        map_location="cpu",
        weights_only=True,
    )
    target_offsets = torch.load(
        args.artifact / "bone_offsets_22.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_equal("bone_offsets_22", source_offsets, target_offsets)

    return {
        "status": "exact",
        "source_checkpoint": str(args.source),
        "artifact": str(args.artifact),
        "transformer_tensors": len(source_keys),
        "learned_condition_tensors": len(LEARNED_TENSORS),
        "max_abs_difference": 0.0,
        "statistics": "exact",
        "bone_offsets": "exact",
    }


def main() -> int:
    result = audit(parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
