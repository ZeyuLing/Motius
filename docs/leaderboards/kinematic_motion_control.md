# Kinematic Motion Control · Native-Skeleton Protocol

<p align="center">
  <a href="README.md">📊 Benchmark Hub</a> ·
  <a href="../tasks/README.md">🧭 Task Registry</a> ·
  <a href="../motion/README.md">🔄 Representation Contracts</a>
</p>

This leaderboard compares methods that generate motion from explicit geometric
evidence: root paths, joint trajectories, sparse positions or rotations,
full-body keyframes, and end-effector targets.

## Fixed Protocol

| Field | Contract |
| ----- | -------- |
| Task | Kinematic Motion Control |
| Tracks | Root path · waypoint · sparse joint · full-body keyframe · end-effector |
| Evaluation space | Each native skeleton first; bridged comparisons reported separately |
| Constraint metric | Position error in meters · rotation error in degrees |
| Motion quality | FID/diversity where a validated evaluator exists |
| Physical quality | Foot slide · floating · jitter · penetration |

Native-skeleton scores are not pooled across HumanML3D, SMPL-X, SOMA, ARDY-27,
or Unitree G1. A cross-skeleton table requires a validated bridge and reports
its fitting error separately.

## Method Coverage

| Method | Native setting | Supported tracks | Status |
| ------ | -------------- | ---------------- | ------ |
| [ARDY](../model_zoo/ardy.md) | ARDY-27 · Unitree G1 | Root path · waypoint · sparse joints · full-body keyframe · end-effector | Protocol integration |
| [CondMDI](../model_zoo/condmdi.md) | HumanML3D-263 | Sparse temporal and spatial constraints | Protocol integration |
| [DART](../model_zoo/dart.md) | DART276 | Trajectory and keyframe control | Protocol integration |
| [KIMODO](../model_zoo/kimodo.md) | SOMA · Unitree G1 · SMPL-X | Root path · full-body keyframe · end-effector | Protocol integration |
| [MaskControl](../model_zoo/maskcontrol.md) | HumanML3D-263 | Joint and temporal masks | Protocol integration |
| [OmniControl](../model_zoo/omnicontrol.md) | HumanML3D-263 | Spatial joint control | Protocol integration |

## Leaderboard

| Method | Native setting | Track | Samples | Constraint error ↓ | Physical score | Status |
| ------ | -------------- | ----- | ------: | -----------------: | -------------- | ------ |
| ARDY | ARDY-27 | Pending | Pending | Pending | Pending | Pending full protocol |
| KIMODO | Native skeleton | Pending | Pending | Pending | Pending | Pending full protocol |
| CondMDI | HumanML3D-263 | Pending | Pending | Pending | Pending | Pending full protocol |
| DART | DART276 | Pending | Pending | Pending | Pending | Pending full protocol |
| MaskControl | HumanML3D-263 | Pending | Pending | Pending | Pending | Pending full protocol |
| OmniControl | HumanML3D-263 | Pending | Pending | Pending | Pending | Pending full protocol |
