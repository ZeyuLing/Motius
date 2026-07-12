<h1 align="center">MotionStreamer Evaluator Card</h1>

<p align="center">
  <strong>MotionStreamer-272 text-motion evaluator for SMPL-aligned T2M results.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2503.15451">Paper</a> |
  <a href="https://zju3dv.github.io/MotionStreamer/">Project Page</a> |
  <a href="https://github.com/zju3dv/MotionStreamer">Original GitHub</a>
</p>

MotionStreamer Evaluator is the second public metric view in Motius model
cards. It evaluates motions through the MotionStreamer-272 representation after
the method output has been converted through the checked SMPL/MotionStreamer
path.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | MotionStreamer Evaluator |
| Motion representation | MotionStreamer-272 |
| Caption protocol | HumanML3D selected-caption protocol unless a card states otherwise |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint/assets | Pending public artifact |

## Reporting Rule

Every T2M model card should include this row. HumanML3D-263, SMPL, SMPL-X, and
DART-style outputs must first go through a checked conversion path before this
metric is reported.

## Notes

This evaluator remains useful as a strong semantic metric, but it is not the
only public metric view. Motius reports it together with HumanML3D Official and
the Motius Joint-Position Evaluator.
