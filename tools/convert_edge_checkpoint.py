#!/usr/bin/env python3
"""Convert the official EDGE pickle into a self-contained Motius artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.models.edge.bundle import (  # noqa: E402
    DEFAULT_EDGE_CONFIG,
    EDGE_OFFICIAL_CHECKPOINT_SHA256,
    EDGEBundle,
)
from motius.models.edge.network import DanceDecoder  # noqa: E402


class _LegacyMinMaxScaler:
    pass


class _LegacyNormalizer:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_checkpoint(path: Path):
    aliases = {
        "dataset": types.ModuleType("dataset"),
        "dataset.preprocess": types.ModuleType("dataset.preprocess"),
        "dataset.scaler": types.ModuleType("dataset.scaler"),
    }
    aliases["dataset.preprocess"].Normalizer = _LegacyNormalizer
    aliases["dataset.scaler"].MinMaxScaler = _LegacyMinMaxScaler
    aliases["dataset"].preprocess = aliases["dataset.preprocess"]
    aliases["dataset"].scaler = aliases["dataset.scaler"]
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    digest = sha256(checkpoint)
    if digest != EDGE_OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(
            f"EDGE checkpoint SHA256 mismatch: expected {EDGE_OFFICIAL_CHECKPOINT_SHA256}, got {digest}"
        )
    payload = load_official_checkpoint(checkpoint)
    state = payload["ema_state_dict"]
    normalizer = payload["normalizer"].scaler
    network = DanceDecoder(**DEFAULT_EDGE_CONFIG["network"])
    incompatible = network.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    bundle = EDGEBundle(
        config=DEFAULT_EDGE_CONFIG,
        network=network,
        normalizer_scale=normalizer.scale_,
        normalizer_min=normalizer.min_,
        provenance={
            "official_checkpoint_sha256": digest,
            "state": "ema_state_dict",
            "normalizer": "checkpoint.normalizer.scaler",
        },
    )
    bundle.save_pretrained(str(args.output))
    report = {
        "source": str(checkpoint),
        "source_sha256": digest,
        "output": str(args.output.resolve()),
        "parameters": sum(value.numel() for value in state.values()),
        "state_keys": len(state),
    }
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
