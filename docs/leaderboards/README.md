# Motius Benchmark Hub

<p align="center">
  <a href="../tasks/README.md">🧭 Task Registry</a> ·
  <a href="../model_zoo/README.md">📦 Model Zoo</a> ·
  <a href="../evaluator_zoo/README.md">📐 Evaluator Zoo</a> ·
  <a href="../evaluation/physical_metrics.md">🏃 Physical Metrics</a>
</p>

Motius maintains fifteen benchmark settings. Benchmark titles follow
`Task · Dataset/Protocol`, so the name identifies both the task and the
contract that makes scores comparable.

Tasks are not grouped into modality or capability families here. Prediction,
in-betweening, and TP2M remain tracks inside Temporal Motion Completion;
style/content and MotionFix remain separate benchmark settings of Motion
Editing. The canonical vocabulary lives in the
[Task Registry](../tasks/README.md).

## Benchmark Directory 📊

| Benchmark | Fixed contract | Resources |
| --------- | -------------- | --------- |
| <a id="text-to-motion"></a> **Text-to-Motion · HumanML3D** | Official HumanML3D test split with persisted selected captions. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) · [🧾 Source](hf_space_t2m_humanml3d) · [📐 Evaluator](../evaluator_zoo/humanml3d_official.md) |
| **Text-to-Motion · Unitree G1** | Direct G1-native generation on the fixed robotic test split, normalized to `g1_38` for TMR-G1 evaluation. | [📋 Leaderboard](t2m_unitree_g1.md) · [📐 Evaluator](../evaluator_zoo/g1_tmr.md) |
| **Motion-to-Text · HumanML3D** | Complete input motions, reference captions, language metrics, and semantic motion-text retrieval metrics. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) · [🧾 Source](hf_space_m2t_humanml3d) · [📋 Protocol](../tasks/m2t.md) |
| **Sequential Text-to-Motion · BABEL** | Ordered multi-prompt composition and transition quality under the canonical processed-BABEL protocol. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) · [🧾 Source](hf_space_babel_sequential) · [📋 Protocol](../evaluation/babel_sequential.md) |
| **Text-to-Multi-Person Motion · InterHuman** | Two synchronized actors generated from one interaction caption in a shared world frame. | [📋 Leaderboard](text_to_multi_person_interhuman.md) · [📐 Evaluator](../evaluator_zoo/interclip.md) |
| **Temporal Motion Completion · HumanML3D** | Prediction, in-betweening, sparse-keyframe, and TP2M tracks; text-conditioned and text-free settings use separate result tables. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) · [🧾 Source](hf_space_temporal_condition) |
| **Kinematic Motion Control · Native-Skeleton Protocol** | Root paths, waypoints, joint constraints, full-body keyframes, and end-effectors evaluated first in each native skeleton. | [📋 Leaderboard](kinematic_motion_control.md) |
| **Part-Level Motion Control · HumanML3D** | Semantic body-region prompts and constraints over selected time intervals. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) |
| <a id="motion-reconstruction-humanml3d"></a> **Motion Reconstruction · HumanML3D** | Complete official test split; global and root-aligned joint error, root motion, velocity or drift, reconstruction FID, and physical diagnostics use a declared common skeleton. | [📐 Evaluators](../evaluator_zoo/README.md) |
| **Motion Editing · Style and Content** | Style and content edits measure target compliance and preservation of the complementary source attribute. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) |
| **Motion Editing · MotionFix Instructions** | MotionFix source motions with free-form edit instructions; this remains a Motion Editing benchmark setting. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) |
| <a id="motion-repair-fixed-support-protocol"></a> **Motion Repair · Fixed-Support Protocol** | Separate oracle-mask and method-native-mask tracks; clean target values are never inputs. Geometry, root, skating, jitter, and semantics share one skeleton convention. | [🏃 Physical metrics](../evaluation/physical_metrics.md) |
| **Music-to-Dance · AIST++** | Dance quality, diversity, beat alignment, and physical diagnostics on AIST++. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) · [🧾 Source](hf_space_music_to_dance) · [📐 Evaluator](../evaluator_zoo/aistpp_music_to_dance.md) |
| **Dance-to-Music · AIST++** | Motion-conditioned music generation with synchronized audio and motion beat diagnostics. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) · [🧾 Source](hf_space_dance_to_music) |
| **Speech-to-Gesture · BEAT2** | Speech-conditioned co-speech gesture generation under the fixed BEAT2 protocol. | [↗ Results](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) |

## Result Contract ✅

| Result element | Requirement |
| -------------- | ----------- |
| Identity | Record dataset split, condition selection, motion representation, evaluator checkpoint, and sample coverage |
| GT row | Use as a calibration reference, never as a generated-method ranking entry |
| Representation bridge | Validate it and disclose lossy conversion or IK before comparing different spaces |
| Physical diagnostics | Report separately from learned semantic scores |
| Qualitative viewer | Use the persisted predictions scored by the metric job whenever they can be distributed |

Evaluator details live in the [Evaluator Zoo](../evaluator_zoo/README.md), with
shared diagnostics documented in
[Physical Motion Metrics](../evaluation/physical_metrics.md).
