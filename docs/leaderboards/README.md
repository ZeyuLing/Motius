# Motius Leaderboard Hub

Motius maintains twelve top-level benchmark families. A family is defined by a
task and canonical evaluation contract, not by an individual model, dataset
variant, or ablation. Public Spaces expose sortable tables and qualitative case
browsers; repository protocols remain the source of truth for evaluation.

## Generation and Understanding

| Family | Scope | Public page | Repository source |
| --- | --- | --- | --- |
| T2M HumanML3D | Text-to-motion on the official HumanML3D test split | [Open](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) | [Source](hf_space_t2m_humanml3d) |
| M2T HumanML3D | Motion captioning with full-motion references and semantic retrieval | [Open](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | [Source](hf_space_m2t_humanml3d) |
| Reconstruction HumanML3D | Tokenizer and autoencoder reconstruction on the full official test split | Protocol maintained | Metrics use the shared motion bridge and evaluator interfaces |

### Reconstruction

Reconstruction compares tokenizer or autoencoder families rather than text
generators. Geometry is evaluated after conversion to the same SMPL-22 body
skeleton, while semantic reconstruction uses a declared persisted evaluator.
Canonical reports include global and root-aligned joint error, root-motion
error, velocity or drift diagnostics, reconstruction FID, and physical metrics.

## Conditional and Sequential Motion

| Family | Scope | Public page | Repository source |
| --- | --- | --- | --- |
| Temporal Condition | Prediction, motion in-betweening, adaptive keyframes, and TP2M | [Open](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) | [Source](hf_space_temporal_condition) |
| Body-Part Condition | Sparse and dense position or rotation constraints on selected body regions | [Open](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | Public Space |
| BABEL Sequential Generation | Ordered multi-prompt motion composition and transition quality | [Open](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | [Source](hf_space_babel_sequential) |

TP2M is a Temporal Condition subtask. It evaluates caption-guided continuation
from observed prefixes of 1, 5, or 9 frames and is not counted as a separate
top-level leaderboard. The older alternate BABEL sequential protocol is also
excluded; only the canonical processed-BABEL protocol is maintained.

## Editing and Repair

| Family | Scope | Public page | Repository source |
| --- | --- | --- | --- |
| Motion Editing | Style and content editing while preserving the complementary source attribute | [Open](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | Public Space |
| Instruction Editing | MotionFix source motion plus free-form edit instruction | [Open](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) | Public Space |
| Motion Repair | Fixed-support repair with oracle-mask and method-native-mask tracks | Protocol maintained | Shared representation and physical-metric interfaces |

### Motion Repair

Motion Repair is distinct from unconstrained generation. A method receives a
corrupted motion and a repair support mask; clean target values are never model
inputs. Oracle-mask and method-native detector tracks remain separate, and
paired geometry, root, skating, jitter, and semantic-retrieval metrics are
reported under one canonical skeleton convention.

## Audio-Driven Motion

| Family | Scope | Public page | Repository source |
| --- | --- | --- | --- |
| Music-to-Dance | Dance quality, diversity, beat alignment, and physical diagnostics | [Open](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | [Source](hf_space_music_to_dance) |
| Dance-to-Music | Motion-conditioned music generation and synchronized beat diagnostics | [Open](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) | [Source](hf_space_dance_to_music) |
| Speech-to-Gesture | Speech-conditioned gesture on the fixed BEAT2 protocol | [Open](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) | Public Space |

PRISM-MCM is a method evaluated inside Music-to-Dance and Speech-to-Gesture.
It is not a separate leaderboard family. Dataset-specific tracks may use their
own sample sets and metrics, but they remain nested under the same task family.

## Evaluation Rules

- Every result names its dataset split, caption or condition selection, motion
  representation, evaluator checkpoint, and sample coverage.
- GT rows are calibration references and do not participate in generated-method
  ranking.
- Results from different representations are compared only after a validated
  bridge, with lossy conversion or IK explicitly disclosed.
- Physical metrics remain separate columns rather than being folded into a
  semantic score.
- Qualitative viewers use the same persisted prediction artifacts as the metric
  jobs whenever those artifacts are publicly distributable.

Evaluator details live in [`docs/evaluator_zoo`](../evaluator_zoo), with shared
physical diagnostics documented in [Physical Motion Metrics](../evaluation/physical_metrics.md).
