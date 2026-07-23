# Motius Benchmark Hub

Motius maintains twelve benchmark suites. Every public title follows
`Task · Dataset/Protocol`, so the name identifies both the capability being
measured and the contract that makes scores comparable.

A benchmark is not a task and a dataset is not a task. Prediction,
in-betweening, and TP2M are tracks inside one benchmark; style editing and
instruction editing are two benchmarks of the same Motion Editing task. The
canonical vocabulary lives in the [Task Registry](../tasks/README.md).

## Language And Motion

### Text-to-Motion · HumanML3D

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) ·
[Repository source](hf_space_t2m_humanml3d) ·
[Evaluator](../evaluator_zoo/humanml3d_official.md)

Official HumanML3D test split with the persisted selected-caption protocol.

### Motion-to-Text · HumanML3D

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) ·
[Repository source](hf_space_m2t_humanml3d) ·
[Protocol](../tasks/m2t.md)

Complete input motions, reference captions, language metrics, and semantic
motion-text retrieval metrics.

### Sequential Text-to-Motion · BABEL

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) ·
[Repository source](hf_space_babel_sequential) ·
[Protocol](../evaluation/babel_sequential.md)

Ordered multi-prompt composition and transition quality under the canonical
processed-BABEL protocol. The obsolete alternate protocol is not maintained.

## Conditioned Motion

### Temporal Motion Completion · HumanML3D

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) ·
[Repository source](hf_space_temporal_condition) ·
[Task contract](../tasks/README.md#temporal-motion-completion)

One benchmark with prediction, in-betweening, sparse-keyframe, and TP2M tracks.
Text-conditioned and text-free settings remain separate result tables within
the same suite.

### Part-Level Motion Control · HumanML3D

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) ·
[Task contract](../tasks/README.md#part-level-motion-control)

Semantic body-region prompts and constraints over selected time intervals.
Numeric joint trajectories belong to Kinematic Motion Control instead.

## Motion Transformation And Restoration

<a id="motion-reconstruction-humanml3d"></a>

### Motion Reconstruction · HumanML3D

Tokenizer, codec, and autoencoder reconstruction on the complete official
HumanML3D test split. Reports include global and root-aligned joint error,
root-motion error, velocity or drift, reconstruction FID, and physical
diagnostics after conversion to a declared common skeleton.

### Motion Editing · Style and Content

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) ·
[Task contract](../tasks/README.md#motion-editing)

Style and content edits evaluated for target compliance and preservation of the
complementary source attribute.

### Motion Editing · MotionFix Instructions

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) ·
[Task contract](../tasks/README.md#motion-editing)

MotionFix source motions paired with free-form edit instructions. This is a
benchmark track of Motion Editing, not a separate "Instruction Editing" task.

<a id="motion-repair-fixed-support-protocol"></a>

### Motion Repair · Fixed-Support Protocol

Fixed-support repair with separate oracle-mask and method-native-mask tracks.
The method receives corrupted motion and an allowed repair support; clean target
values are never inputs. Geometry, root, skating, jitter, and semantic metrics
share one canonical skeleton convention.

## Audio And Motion

### Music-to-Dance · AIST++

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) ·
[Repository source](hf_space_music_to_dance) ·
[Evaluator](../evaluator_zoo/aistpp_music_to_dance.md)

Dance quality, diversity, beat alignment, and physical diagnostics on AIST++.
PRISM-MCM is a method evaluated here, not a task or benchmark family.

### Dance-to-Music · AIST++

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) ·
[Repository source](hf_space_dance_to_music) ·
[Task contract](../tasks/README.md#dance-to-music)

Motion-conditioned music generation with synchronized audio and motion beat
diagnostics.

### Speech-to-Gesture · BEAT2

[Open benchmark](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) ·
[Task contract](../tasks/README.md#speech-to-gesture)

Speech-conditioned co-speech gesture generation on the fixed BEAT2 protocol.
PRISM-MCM remains a method in this task.

## Result Contract

- Every result identifies dataset split, condition selection, motion
  representation, evaluator checkpoint, and sample coverage.
- GT rows are calibration references and never participate in generated-method
  ranking.
- Different representations are compared only after a validated bridge, with
  lossy conversion or IK explicitly disclosed.
- Physical diagnostics remain separate from semantic scores.
- Qualitative viewers use the persisted predictions scored by the metric job
  whenever those artifacts are publicly distributable.

Evaluator details live in the [Evaluator Zoo](../evaluator_zoo), with shared
diagnostics documented in
[Physical Motion Metrics](../evaluation/physical_metrics.md).
