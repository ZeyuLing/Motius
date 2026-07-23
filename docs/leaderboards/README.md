# Motius Leaderboard Hub

Motius maintains twelve top-level benchmark families. A family is defined by a
canonical task and evaluation contract, not by an individual method, dataset
variant, or ablation. Public Spaces expose sortable results and qualitative
case browsers; repository protocols remain the source of truth.

Task names follow the [Motius task definitions](../tasks/README.md).

## Generation And Understanding

### T2M HumanML3D

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) ·
[Repository source](hf_space_t2m_humanml3d) ·
[Evaluator](../evaluator_zoo/humanml3d_official.md)

Text-to-Motion evaluation on the official HumanML3D test split using the fixed
selected-caption protocol.

### M2T HumanML3D

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) ·
[Repository source](hf_space_m2t_humanml3d) ·
[Task protocol](../tasks/m2t.md)

Motion-to-Text evaluation with complete input motions, reference captions, and
semantic retrieval metrics.

### Reconstruction

Tokenizer and autoencoder reconstruction on the complete official HumanML3D
test split. Geometry is compared after conversion to the same SMPL-22 body
skeleton; semantic reconstruction uses a declared persisted evaluator.
Canonical reports include global and root-aligned joint error, root-motion
error, velocity or drift diagnostics, reconstruction FID, and physical metrics.

## Conditional And Sequential Motion

### Temporal Condition

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) ·
[Repository source](hf_space_temporal_condition) ·
[Task definition](../tasks/README.md#temporal-condition)

One parent benchmark for prediction, motion in-betweening, adaptive keyframes,
and TP2M. TP2M is not counted as a separate top-level leaderboard.

### Body-Part Condition

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) ·
[Task definition](../tasks/README.md#body-part-condition)

Sparse and dense position or rotation constraints on selected body regions.

### BABEL Sequential Generation

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) ·
[Repository source](hf_space_babel_sequential) ·
[Protocol](../evaluation/babel_sequential.md)

Ordered multi-prompt motion composition and transition quality under the
canonical processed-BABEL protocol. The older alternate sequential protocol is
not maintained.

## Editing And Repair

### Motion Editing

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) ·
[Task definition](../tasks/README.md#motion-editing)

Style and content editing while preserving the complementary source attribute.

### Instruction Editing

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) ·
[Task definition](../tasks/README.md#motion-editing)

MotionFix source motion plus a free-form edit instruction.

### Motion Repair

Fixed-support repair with separate oracle-mask and method-native-mask tracks. A
method receives corrupted motion and a repair support mask; clean target values
are never inputs. Paired geometry, root, skating, jitter, and semantic metrics
share one canonical skeleton convention.

## Audio-Driven Motion

### Music-to-Dance

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) ·
[Repository source](hf_space_music_to_dance) ·
[Evaluator](../evaluator_zoo/aistpp_music_to_dance.md)

Dance quality, diversity, beat alignment, and physical diagnostics on AIST++.
PRISM-MCM is evaluated as a method in this family rather than as a separate
leaderboard.

### Dance-to-Music

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) ·
[Repository source](hf_space_dance_to_music) ·
[Task definition](../tasks/README.md#dance-to-music)

Motion-conditioned music generation with synchronized beat diagnostics.

### Speech-to-Gesture

[Open leaderboard](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) ·
[Task definition](../tasks/README.md#speech-to-gesture)

Speech-conditioned gesture generation on the fixed BEAT2 protocol. PRISM-MCM
remains a method in this task family.

## Evaluation Rules

- Every result names its dataset split, caption or condition selection, motion
  representation, evaluator checkpoint, and sample coverage.
- GT rows are calibration references and do not participate in generated-method
  ranking.
- Different representations are compared only after a validated bridge, with
  lossy conversion or IK explicitly disclosed.
- Physical metrics remain separate columns rather than being folded into a
  semantic score.
- Qualitative viewers use the same persisted predictions as metric jobs whenever
  those artifacts are publicly distributable.

Evaluator details live in the [Evaluator Zoo](../evaluator_zoo), with shared
diagnostics documented in
[Physical Motion Metrics](../evaluation/physical_metrics.md).
