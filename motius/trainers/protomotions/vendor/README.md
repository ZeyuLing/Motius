# ProtoMotions training source snapshot

This directory contains the Apache-2.0 ProtoMotions training source from
[`NVlabs/ProtoMotions`](https://github.com/NVlabs/ProtoMotions) at commit
`49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c`.

Motius invokes this snapshot through
`python -m motius.trainers.protomotions.train`. No external source checkout is
imported at runtime. The snapshot includes the G1 assets and experiment code
needed by the public tracker configuration, together with the native
checkpoint-to-ONNX deployment exporter. H1_2, licensed SMPL body-model, and
other asset families that are not used by the public G1 configuration are not
packaged.

The GenTrack release also backports the G1 conversion and PyRoki retarget
utilities from upstream ProtoMotions commit
`a87d2f3a0be91fc1ab3485ef4ace2e92ef40b0fd`. The qpos converter and
`gentrack_g1_xy_offset.py` experiment are Motius GenTrack adapters derived
from that Apache-2.0 snapshot; their SPDX headers preserve the upstream
copyright and license.
