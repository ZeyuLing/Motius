<!-- MOTIUS_DOCS_NAV:START -->
<p><a href="../../README.md"><strong>Motius</strong></a> / <a href="../README.md">Documentation</a> / <strong>Benchmark Hub</strong></p>
<p>
  <a href="../getting_started.md">Quickstart</a> ·
  <a href="../tasks/README.md">Tasks</a> ·
  <a href="../datasets/README.md">Datasets</a> ·
  <a href="../model_zoo/README.md">Models</a> ·
  <a href="../training/README.md">Training</a> ·
  <a href="../evaluator_zoo/README.md">Evaluators</a> ·
  <strong>Benchmarks</strong> ·
  <a href="../motion/README.md">Motion I/O</a>
</p>
<!-- MOTIUS_DOCS_NAV:END -->

# Motius Benchmark Hub

Motius publishes the evaluated benchmark settings listed below. A leaderboard can expose several
representation-specific settings when their metric spaces must remain
separate. Text-to-Motion, for example, switches between SMPL Skeleton and
Unitree G1 Skeleton without presenting them as unrelated leaderboards. The
directory distinguishes complete publications from metric-only settings. Task
protocols without a verified public result pack remain in the
[Task Registry](../tasks/README.md) and are not listed as leaderboards.

Tasks are not grouped into modality or capability families here. Prediction,
in-betweening, and TP2M remain tracks inside Temporal Motion Completion;
style/content and MotionFix remain separate benchmark settings of Motion
Editing. The canonical vocabulary lives in the
[Task Registry](../tasks/README.md).

## Benchmark directory

| Benchmark | Status | Metrics | Visualization | Resources |
| --------- | ------ | ------- | ------------- | --------- |
| <a id="text-to-motion"></a> **Text-to-Motion** | 2 settings: SMPL complete · G1 metrics ready | [SMPL: 27 methods + GT](hf_space_t2m_humanml3d/t2m_results.json) · [G1: 2 methods + GT](hf_space_t2m_unitree_g1/g1_results.json) | [SMPL: 4,042 cases · all methods](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html) · [G1: 64 cases](https://zeyuling-t2m-unitree-g1-leaderboard.static.hf.space/cases/index.html) | [Open Leaderboard](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) · [SMPL Skeleton source](hf_space_t2m_humanml3d) · [Unitree G1 Skeleton source](hf_space_t2m_unitree_g1) · [G1 protocol](t2m_unitree_g1.md) |
| **Motion-to-Text · HumanML3D** | Complete | [5 methods + GT](hf_space_m2t_humanml3d/m2t_results.json) | [4,400 cases · 6/6 rows](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) · [Dataset](../datasets/README.md#humanml3d) · [Source](hf_space_m2t_humanml3d) · [Protocol](../tasks/m2t.md) |
| **Sequential Text-to-Motion · BABEL** | Complete | [5 methods + GT](hf_space_babel_sequential/babel_results.json) | [1,295 cases · 6/6 rows](https://zeyuling-babel-sequential-generation-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) · [Dataset](../datasets/README.md#babel) · [Source](hf_space_babel_sequential) · [Protocol](../evaluation/babel_sequential.md) |
| **Temporal Motion Completion · HumanML3D** | Complete | [8 settings + TP2M](hf_space_temporal_condition/temporal_control_results.json) | [4,012 cases · 8/8 settings](https://zeyuling-temporal-condition-leaderboard.static.hf.space/cases/start_1f/) | [Results](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) · [Dataset](../datasets/README.md#humanml3d) · [Source](hf_space_temporal_condition) |
| **Part-Level Motion Control · HumanML3D** | Complete | [84 settings](hf_space_body_part_condition_humanml3d/body_part_condition_results.json) | [4,012 cases · GT + 5 native methods](https://zeyuling-body-part-condition-humanml3d-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) · [Dataset](../datasets/README.md#humanml3d) · [Source](hf_space_body_part_condition_humanml3d) |
| **Motion Editing · Style and Content** | Complete | [2 tracks](hf_space_motion_edit/motion_edit_results.json) | [130 cases · all measured rows](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | [Results](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) · [Dataset](../datasets/README.md#permo) · [Source](hf_space_motion_edit) |
| **Motion Editing · MotionFix Instructions** | Complete | [4 methods + GT + source](hf_space_instruction_editing/instruction_editing_results.json) | [1,013 cases · 6/6 rows](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) | [Results](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) · [Dataset](../datasets/README.md#motionfix) · [Source](hf_space_instruction_editing) |
| <a id="motion-repair-fixed-support-protocol"></a> **Motion Repair · BrokenAMASS** | Complete | [MoGenDiT + StableMotion + MotionCanvas + references](hf_space_motion_repair/motion_repair_results.json) | [299 cases · 5 synchronized SMPL views](https://zeyuling-motion-repair-brokenamass-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard) · [Source](hf_space_motion_repair) · [Source-only D3 pipeline](../tasks/motion_repair_source_only.md) · [Physical metrics](../evaluation/physical_metrics.md) |
| <a id="motion-reconstruction-humanml3d"></a> **Motion Reconstruction · HumanML3D** | Complete | [9 autoencoders + GT](hf_space_motion_reconstruction/reconstruction_results.json) | [4,042 cases · 10 synchronized SMPL views](https://zeyuling-motion-reconstruction-humanml3d-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/motion-reconstruction-humanml3d-leaderboard) · [Dataset](../datasets/README.md#humanml3d) · [Source](hf_space_motion_reconstruction) |
| **Music-to-Dance · AIST++** | Complete | [4 methods + GT](hf_space_music_to_dance/music_to_dance_results.json) | [40 audio-synchronized cases · 5/5 rows](https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) · [Dataset](../datasets/README.md#aistpp) · [Source](hf_space_music_to_dance) · [Evaluator](../evaluator_zoo/aistpp_music_to_dance.md) |
| **Dance-to-Music · AIST++** | Complete | [1 Motius-measured method: UniMuMo](hf_space_dance_to_music/dance_to_music_results.json) | [86 motion/audio cases · UniMuMo + reference audio](https://zeyuling-dance-to-music-aistpp-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) · [Dataset](../datasets/README.md#aistpp) · [Source](hf_space_dance_to_music) |
| **Speech-to-Gesture · BEAT2** | Complete | [3 methods + GT](hf_space_speech_to_gesture/speech_to_gesture_results.json) | [15 audio-synchronized cases · 4/4 rows](https://zeyuling-speech-to-gesture-beat2-leaderboard.static.hf.space/cases/index.html) | [Results](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) · [Dataset](../datasets/README.md#beat2) · [Source](hf_space_speech_to_gesture) |
| **Monocular Motion Capture · 3DPW Test** | Complete | [4 verified methods + GT](hf_space_monocular_capture/monocular_capture_results.json) | Video demos for every integrated method | [Results and demos](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) · [Dataset](../datasets/README.md#3dpw) · [Source](hf_space_monocular_capture) · [Protocol](../tasks/monocular_motion_capture.md) |
| **Motion Tracking · Unitree G1** | Complete · 2 engine settings | [MuJoCo: GT + 3 controllers](hf_space_motion_tracking_mujoco/motion_tracking_results.json) · [Isaac Lab: GT + SONIC + BeyondMimic upper bound](hf_space_motion_tracking_isaaclab/motion_tracking_results.json) | All-case Three.js comparison: 40 MuJoCo references and 178 Isaac Lab references | [MuJoCo Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) · [Isaac Lab Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard) · [Protocol](../tasks/motion_tracking.md) |

## Result contract

| Result element | Requirement |
| -------------- | ----------- |
| Identity | Record dataset split, condition selection, motion representation, evaluator checkpoint, and sample coverage |
| GT row | Use as a calibration reference, never as a generated-method ranking entry |
| Primary ordering | Prefer a verified uTMR metric when the task has one; keep task-native and physical metrics as selectable secondary views |
| Representation bridge | Validate it and disclose lossy conversion or IK before comparing different spaces |
| Physical diagnostics | Report separately from learned semantic scores |
| Qualitative viewer | Use the persisted predictions scored by the metric job whenever they can be distributed |
| Artifact path | Use the canonical [`task / benchmark / protocol / run` layout](../evaluation/artifact_layout.md); every registered setting publishes its root in `docs/tasks/taxonomy.json` |
| Learned-embedding FID | L2-normalize every reference and generated embedding before estimating means and covariances; raw-space FID is not rankable |
| Publication status | `Complete` requires both metric and qualitative coverage; `Metrics ready` cannot be presented as a complete visual benchmark |

Evaluator details live in the [Evaluator Zoo](../evaluator_zoo/README.md), with
shared diagnostics documented in
[Physical Motion Metrics](../evaluation/physical_metrics.md).

The machine-readable publication inventory is
[`catalog.json`](catalog.json). Run
`python tools/audit_leaderboards.py` before publishing a leaderboard or
`python tools/audit_leaderboards.py --online` after deploying its Space.
The additive, parent-pinned batch workflow is documented in
[Hugging Face Space publishing](HUGGINGFACE_PUBLISHING.md).
