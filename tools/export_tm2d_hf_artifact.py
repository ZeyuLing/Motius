#!/usr/bin/env python3
"""Convert the authors' TM2D checkpoints into one Motius HF artifact."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from motius.models.tm2d.bundle import (
    DEFAULT_TM2D_CONFIG,
    TM2D_SOURCE_REPOSITORY,
    TM2D_SOURCE_REVISION,
    TM2DBundle,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--vq-checkpoint", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def load_vocabulary(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as handle:
        return pickle.load(handle)


def main():
    args = parse_args()
    joint = load_payload(args.joint_checkpoint)
    vq = load_payload(args.vq_checkpoint)
    provenance = {
        "source_repository": TM2D_SOURCE_REPOSITORY,
        "source_revision": TM2D_SOURCE_REVISION,
        "joint_checkpoint": {
            "name": args.joint_checkpoint.name,
            "epoch": int(joint.get("ep", 20)),
            "total_iterations": int(joint.get("total_it", 7380)),
        },
        "vq_checkpoint": {
            "name": args.vq_checkpoint.name,
            "epoch": int(vq.get("ep", 190)),
            "total_iterations": int(vq.get("total_it", 2523200)),
        },
    }
    bundle = TM2DBundle(
        DEFAULT_TM2D_CONFIG,
        vocabulary=load_vocabulary(args.vocabulary),
        mean=np.load(args.mean),
        std=np.load(args.std),
        provenance=provenance,
    )
    states = {
        "vq_encoder": vq["vq_encoder"],
        "quantizer": vq["quantizer"],
        "vq_decoder": vq["vq_decoder"],
        "audio_transformer": joint["a2d_transformer"],
        "text_transformer": joint["t2m_transformer"],
    }
    for name, state in states.items():
        getattr(bundle, name).load_state_dict(state, strict=True)
    bundle.save_pretrained(args.output)

    restored = TM2DBundle.from_pretrained(args.output, local_files_only=True)
    for name in states:
        expected = getattr(bundle, name).state_dict()
        actual = getattr(restored, name).state_dict()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    print(json.dumps({"artifact": str(args.output), "status": "verified"}, indent=2))


if __name__ == "__main__":
    main()
