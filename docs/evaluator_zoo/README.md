# Motius Evaluator Zoo

<p align="center">
  <a href="../tasks/README.md">🧭 Task Registry</a> ·
  <a href="../model_zoo/README.md">📦 Model Zoo</a> ·
  <a href="../leaderboards/README.md">📊 Benchmark Hub</a> ·
  <a href="../evaluation/physical_metrics.md">🏃 Physical Metrics</a>
</p>

The Evaluator Zoo packages metric implementations separately from model
identity. Choose an evaluator by its declared motion space and benchmark
contract; converting a prediction into another space must remain explicit.

## Evaluator Matrix 📐

| Evaluator | Native input | Principal metrics | Artifact |
| --------- | ------------ | ----------------- | -------- |
| [HumanML3D Official](humanml3d_official.md) | HumanML3D-263 at 20 fps | R-Precision · normalized FID · MM-Dist · Diversity | [📦 Checkpoint](https://huggingface.co/ZeyuLing/motius-evaluator-humanml3d-official) |
| [MotionStreamer Evaluator](motionstreamer.md) | MotionStreamer-272 at 30 fps | R-Precision · normalized FID · MM-Dist · Diversity | [📦 Checkpoint](https://huggingface.co/ZeyuLing/motius-evaluator-motionstreamer-272) |
| [Motius Joint-Position Evaluator](motius_joint_position.md) | Canonical SMPL-22 joints66 at 30 fps | R-Precision · normalized FID · MM-Dist · Diversity | [📦 Checkpoint](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66) |
| [Motius TMR-G1 Evaluator](g1_tmr.md) | Canonical Unitree G1-38D at 30 fps | R-Precision · normalized FID · MM-Dist · Diversity | [📦 Checkpoint](https://huggingface.co/ZeyuLing/motius-evaluator-g1-38d-tmr) |
| [InterCLIP](interclip.md) | Paired InterHuman-262 | R-Precision · normalized FID · MM-Dist · Diversity | [📦 Checkpoint](https://huggingface.co/ZeyuLing/motius-evaluator-interhuman-interclip) |
| [AIST++ Music-to-Dance](aistpp_music_to_dance.md) | AIST++ SMPL-24 joints and music beats | `FID_k` · `FID_g` · Diversity · BeatAlign · `FID_uTMR` | [📦 Protocol artifact](https://huggingface.co/ZeyuLing/Motius-Evaluator-AISTPP-Music-to-Dance) |

## Selection Rules ✅

| Situation | Required action |
| --------- | --------------- |
| Method and evaluator share a native representation | Evaluate directly and report the exact checkpoint |
| Method and evaluator use different representations | Use a validated bridge and disclose conversion diagnostics |
| Semantic and physical quality are both relevant | Report learned semantic metrics and [checkpoint-free physical metrics](../evaluation/physical_metrics.md) separately |
| A benchmark defines a fixed evaluator | Keep that evaluator and its preprocessing unchanged across methods |
| A GT row is available | Treat it as calibration, never as a generated-method ranking entry |

Machine-readable evaluator release metadata is available in
[`release_manifest.json`](release_manifest.json).
