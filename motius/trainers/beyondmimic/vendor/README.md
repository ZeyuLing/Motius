# BeyondMimic Training Source Snapshot

This directory contains the MIT-licensed training source from
[`HybridRobotics/whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking)
at commit `cd65172032893724b445448818c34165846d847d`.

Motius keeps the official Isaac Lab environment, PPO configuration, reward,
termination, observation, checkpoint, and ONNX export implementations intact.
Four integration patches are intentionally small:

1. `assets/__init__.py` accepts `MOTIUS_BEYONDMIMIC_ASSET_ROOT`.
2. `scripts/rsl_rl/train.py` accepts an official-format local motion NPZ in
   addition to the upstream WandB registry route.
3. `scripts/csv_to_npz.py` can write the official converted NPZ locally.
4. `scripts/rsl_rl/play.py` provides export-only mode after writing the
   official metadata-bearing ONNX graph.

The Unitree description is not redistributed here. Use
`tools/download_beyondmimic_assets.py`, which downloads the exact archive
specified by the upstream README and verifies its SHA256 digest.
