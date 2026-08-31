<!-- MOTIUS_DOCS_NAV:START -->
<p><a href="../../README.md"><strong>Motius</strong></a> / <a href="../README.md">Documentation</a> / <strong>Task Registry</strong></p>
<p>
  <a href="../getting_started.md">Quickstart</a> ·
  <strong>Tasks</strong> ·
  <a href="../datasets/README.md">Datasets</a> ·
  <a href="../model_zoo/README.md">Models</a> ·
  <a href="../training/README.md">Training</a> ·
  <a href="../evaluator_zoo/README.md">Evaluators</a> ·
  <a href="../leaderboards/README.md">Benchmarks</a> ·
  <a href="../motion/README.md">Motion I/O</a>
</p>
<!-- MOTIUS_DOCS_NAV:END -->

# Motius Task Registry

This registry is the single vocabulary used by pipelines, model cards, the
Model Zoo, and benchmark documentation.

| Entity | Meaning |
| ------ | ------- |
| 🧭 **Task** | An input/output contract |
| 🛤️ **Track** | A task narrowed by a conditioning pattern or operating setting |
| 📊 **Benchmark** | A task bound to a dataset, split, selection rule, evaluator, and metric implementation |
| 📦 **Method** | A model implementation serving one or more tasks |

Dataset names, model architectures, motion representations, and training
objectives are not task names. Tasks are intentionally listed without
top-level families because modality, operation, actor count, and application
domain are overlapping axes. The machine-readable source is
[`taxonomy.json`](taxonomy.json).

## Task matrix

Tasks with a public leaderboard link to it. Benchmark-pending tasks link to
their executable runtime contract.

| Task | Condition → output | Principal scope / tracks | Leaderboard settings |
| ---- | ------------------ | ------------------------ | -------------------- |
| [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) | Text → motion | Offline · streaming · duration control | [SMPL Skeleton](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) · [Unitree G1 Skeleton](https://huggingface.co/spaces/ZeyuLing/t2m-unitree-g1-leaderboard) |
| [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | Motion → caption | Caption generation · API alias `M2T` | [HumanML3D](../datasets/README.md#humanml3d) |
| [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | Ordered prompts → continuous motion | Multi-action composition · transitions | [BABEL](../datasets/README.md#babel) |
| [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | Interaction text → shared-frame actors | Two-person · multi-person | [InterHuman](../datasets/README.md#interhuman) |
| [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) | Observed frames ± text → motion | Prediction · in-betweening · keyframes · TP2M | [HumanML3D](../datasets/README.md#humanml3d) |
| [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | Numeric geometry → motion | Joint · root · trajectory · end-effector | Native-skeleton protocol |
| [Part-Level Motion Control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | Body-region semantics → motion | Spatial and temporal part composition | [HumanML3D](../datasets/README.md#humanml3d) |
| [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | Motion + semantic edit → motion | Style · content · free-form instruction | [PerMo](../datasets/README.md#permo) · [MotionFix](../datasets/README.md#motionfix) |
| [Motion Repair](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard) | Corrupted motion ± support → motion | 299 paired cases · oracle-v6 and method-native support | [BrokenAMASS](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard) |
| [Motion Reconstruction](../leaderboards/README.md#motion-reconstruction-humanml3d) | Motion → reconstructed motion | Tokenizer · codec · autoencoder | [HumanML3D](../datasets/README.md#humanml3d) |
| [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | Music ± text → dance | Beat-aligned dance · API alias `M2D` | [AIST++](../datasets/README.md#aistpp) |
| [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) | Dance → music | Motion-conditioned audio · API alias `D2M` | [AIST++](../datasets/README.md#aistpp) |
| [Speech-to-Gesture](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) | Speech ± caption → gesture | Co-speech gesture · API alias `S2G` | [BEAT2](../datasets/README.md#beat2) |
| [Monocular Motion Capture](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) | Monocular RGB video → body motion | Camera-relative · world-grounded · multi-person tracking | [3DPW Test](../datasets/README.md#3dpw) · [EMDB-1/2](../datasets/README.md#emdb) protocol support |
| [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | Reference motion + robot state → physical trajectory and control action | Policy-step and closed-loop APIs · MuJoCo / Isaac Lab settings | [LAFAN1-G1](../datasets/README.md#lafan1) · [AMASS](../datasets/README.md#amass) |

Robot representations, human-to-robot retargeting, and G1 export are
implementation capabilities rather than standalone tasks. A robotic method can
still implement Text-to-Motion, as in the Unitree G1 T2M setting, or Motion
Tracking when it maps a reference motion and measured robot state to actions.

## Task Definitions

### Text-to-Motion

**Input:** one natural-language motion description, with an optional requested
duration.

**Output:** one motion sequence for a declared body or robot skeleton.

`T2M` is an accepted API alias. Human and robotic skeletons are benchmark
settings of the same task, not different task names. Offline, streaming,
diffusion, flow-matching, autoregressive, masked-token, and latent generators
also remain the same task.

### Motion-to-Text

**Input:** one complete motion sequence.

**Output:** one natural-language motion description.

`M2T` is an accepted API alias. Caption generation is the task; retrieval,
BLEU, ROUGE, CIDEr, or embedding similarity are benchmark metrics.

### Sequential Text-to-Motion

**Input:** an ordered sequence of text prompts, each assigned an interval or
duration.

**Output:** one continuous motion containing the requested actions and their
transitions.

Online prompt updates and multi-prompt composition belong here. "Sequential"
describes the input contract, not a model architecture.

### Text-to-Multi-Person Motion

**Input:** one natural-language interaction description.

**Output:** synchronized motions for two or more actors in one shared world
frame.

Current InterHuman releases use a two-actor track. Actor count is an output
layout property, so `Two-Person T2M`, `Interaction T2M`, and
`Multi-Person Motion` are not additional public task labels.

### Temporal Motion Completion

**Input:** observed motion frames or frame-level constraints, optionally paired
with text.

**Output:** a complete motion sequence that preserves the supplied temporal
evidence.

The benchmark tracks are:

- **Prediction:** continue from an observed prefix.
- **In-betweening:** fill motion between observed boundaries.
- **Sparse keyframes:** complete motion around fixed or adaptive observed
  frames.
- **TP2M:** caption-guided continuation from an observed motion prefix.

These are tracks of one task, not separate Model Zoo tasks.

### Kinematic Motion Control

**Input:** numeric keyframes, trajectories, joint positions or rotations,
end-effector targets, or another explicit geometric condition.

**Output:** one motion satisfying the supplied kinematic constraints.

Trajectory control, joint control, root control, and end-effector control are
tracks under this contract.

### Part-Level Motion Control

**Input:** semantic prompts or constraints assigned to named body regions and
temporal intervals.

**Output:** one motion satisfying the requested part-level behavior while
preserving or generating the complementary body motion.

This task describes semantic body-region composition. Numeric joint or
trajectory evidence belongs to Kinematic Motion Control.

### Motion Editing

**Input:** a source motion and a semantic edit instruction or attribute.

**Output:** an edited motion that applies the requested change while preserving
unmodified content.

Style/content editing and free-form MotionFix instructions are benchmark tracks
of the same task.

### Motion Repair

**Input:** a corrupted motion and, when required by the method, an observed
repair support or method-native corruption estimate.

**Output:** a restored motion that preserves valid source content.

Oracle-mask and predicted-mask protocols are benchmark tracks, not separate
tasks. The public [BrokenAMASS leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard)
shows each support track explicitly. The
[training-free source-only D3 pipeline](motion_repair_source_only.md) remains a
supplementary pipeline rather than a ranked row in this paired benchmark.

### Motion Reconstruction

**Input:** one motion sequence.

**Output:** a reconstruction of the same sequence through a tokenizer,
autoencoder, codec, or representation bottleneck.

Reconstruction is a motion-to-motion task. MPJPE, root error, FID, codebook
usage, and physical diagnostics are evaluation measures rather than task names.

### Music-to-Dance

**Input:** music audio or synchronized music features, optionally paired with a
text condition.

**Output:** a synchronized dance motion.

`M2D` is an accepted API alias. PRISM-MCM, Bailando, EDGE, and TM2D are methods,
not tasks.

### Dance-to-Music

**Input:** one dance motion.

**Output:** synchronized music audio or music tokens.

`D2M` is an accepted API alias.

### Speech-to-Gesture

**Input:** speech audio or speech features, optionally paired with a semantic
caption.

**Output:** a synchronized co-speech gesture motion.

`S2G` is an accepted API alias.

### Monocular Motion Capture

**Input:** one monocular RGB video, optionally with a target-person track or
crop.

**Output:** time-aligned body parameters, joints, or mesh geometry in explicitly
declared camera and, when estimated, world coordinate systems.

The registered 3DPW Test benchmark uses official person tracks and camera-space
metrics. EMDB-1 camera-space and EMDB-2 world-space manifests, materialization,
and evaluators are implemented, but become public benchmark settings only
after licensed data is available for a complete verified run.

### Motion Tracking

**Input:** a time-indexed reference motion plus the current robot/controller
state required by a declared policy observation contract.

**Output:** one policy action/joint target, or a complete physical rollout with
trajectory, termination, and metric metadata.

`infer_motion_tracking` runs one policy step. `rollout_motion_tracking` runs a
registered physical backend and returns persisted qpos, actions, termination,
and metrics. Reference preparation remains explicit, and results from MuJoCo
and Isaac Lab are never mixed. See the
[Motion Tracking runtime guide](motion_tracking.md) for the released SONIC,
ProtoMotions, Any2Track, and BeyondMimic contracts.

## Naming contract

- Public task fields use only labels from
  [`taxonomy.json`](taxonomy.json).
- Every benchmark entry in `taxonomy.json` owns one immutable `protocol_id`
  and canonical `artifact_root` under the shared
  [evaluation artifact layout](../evaluation/artifact_layout.md).
- Multi-setting leaderboards use `Task · Setting`, for example
  `Text-to-Motion · SMPL Skeleton` and
  `Text-to-Motion · Unitree G1 Skeleton`.
- Prediction, in-betweening, sparse keyframes, and TP2M stay under Temporal
  Motion Completion.
- Human, multi-person, and robot skeletons are benchmark settings or output
  layouts, not top-level task families.
- Architecture terms such as diffusion, autoregressive, streaming, latent,
  zero-shot, and multimodal never become task labels.
- Representation conversion, body-model conversion, retargeting, and character
  export remain Motion Toolkit operations.
