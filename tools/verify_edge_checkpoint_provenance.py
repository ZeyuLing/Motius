"""Verify all native EDGE weights/scaler against the author's original file."""

from pathlib import Path
import argparse
import hashlib
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "artifact", "upstream", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    import numpy as np
    from safetensors.torch import load_file

    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert sha == "28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7"
    sys.path.insert(0, str(args.upstream.resolve()))
    original = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    source = {
        key.removeprefix("module."): value
        for key, value in original["ema_state_dict"].items()
    }
    native = load_file(str(args.artifact / "model.safetensors"))
    assert source.keys() == native.keys()
    assert all(
        source[key].dtype == native[key].dtype and torch.equal(source[key], native[key])
        for key in source
    )
    with np.load(args.artifact / "normalizer.npz", allow_pickle=False) as stats:
        for key, name in [("scale", "scale_"), ("min", "min_")]:
            converted = torch.from_numpy(stats[key])
            value = getattr(original["normalizer"].scaler, name)
            assert converted.dtype == value.dtype and torch.equal(converted, value)
    report = dict(
        complete=True,
        scope="Every native EDGE EMA state and both range-scaler tensors versus the original author Google Drive checkpoint, including dtype. Does not certify raw audio extraction or the original training run.",
        download_url="https://drive.google.com/uc?id=1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ",
        checkpoint_sha256=sha,
        checkpoint_bytes=args.checkpoint.stat().st_size,
        exact_neural_tensors=len(native),
        exact_scaler_tensors=2,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
