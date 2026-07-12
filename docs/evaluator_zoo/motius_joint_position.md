<h1 align="center">Motius Joint-Position Evaluator Card</h1>

<p align="center">
  <strong>Universal SMPL-H joint-position evaluator for source-fair T2M reporting.</strong>
</p>

The Motius Joint-Position Evaluator is the new universal evaluator trained on
HYMotion Data and MotionHub with unified SMPL-H joint positions. It is designed
to avoid rotation-twist ambiguity in SMPL-style rotation features by scoring the
motion in joint-position space.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | Motius Joint-Position Evaluator |
| Motion representation | SMPL-H joints66, 22 joints in xyz |
| Training data | HYMotion Data + MotionHub |
| Caption protocol | HumanML3D selected-caption protocol for HumanML3D reporting; MotionHub official test split for MotionHub reporting |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint/assets | Pending public artifact |

## Reporting Rule

Every T2M model card should include this row. If the joint-position evaluator
has not been run for a method yet, the row should be marked `Pending`.

## Notes

The evaluator expects a unified SMPL-H skeleton and canonicalized joint
positions. Methods that generate HumanML3D-263, MotionStreamer-272, SMPL,
SMPL-X, or DART-style motion must use the checked conversion path before
reporting this metric.
