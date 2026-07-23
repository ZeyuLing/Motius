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
objectives are not task names. The machine-readable source is
[`taxonomy.json`](taxonomy.json).

<p align="center">
  <a href="../model_zoo/README.md">📦 Model Zoo</a> ·
  <a href="../leaderboards/README.md">📊 Benchmark Hub</a> ·
  <a href="../motion/README.md">🔄 Motion Toolkit</a>
</p>

## Task Matrix 🧭

| Family | Task | Input | Output / principal tracks |
| --- | --- | --- | --- |
| 💬 Language and motion | [Text-to-Motion](#text-to-motion) | Text, optional duration | Motion · API alias `T2M` |
| 💬 Language and motion | [Motion-to-Text](#motion-to-text) | Motion | Caption · API alias `M2T` |
| 💬 Language and motion | [Sequential Text-to-Motion](#sequential-text-to-motion) | Ordered prompts and intervals | One continuous multi-action motion |
| 💬 Language and motion | [Text-to-Multi-Person Motion](#text-to-multi-person-motion) | Interaction description | Shared-frame motion for two or more actors |
| 🎛️ Conditioned motion | [Temporal Motion Completion](#temporal-motion-completion) | Observed frames, optional text | Prediction · in-betweening · keyframes · TP2M |
| 🎛️ Conditioned motion | [Kinematic Motion Control](#kinematic-motion-control) | Numeric pose or trajectory constraints | Joint · root · trajectory · end-effector control |
| 🎛️ Conditioned motion | [Part-Level Motion Control](#part-level-motion-control) | Semantic body-region conditions | Composed full-body motion |
| ✂️ Transformation and restoration | [Motion Editing](#motion-editing) | Source motion and semantic edit | Style · content · instruction editing |
| ✂️ Transformation and restoration | [Motion Repair](#motion-repair) | Corrupted motion and repair support | Restored motion |
| ✂️ Transformation and restoration | [Motion Reconstruction](#motion-reconstruction) | Motion | Tokenizer, codec, or autoencoder reconstruction |
| 🎵 Audio and motion | [Music-to-Dance](#music-to-dance) | Music, optional text | Dance · API alias `M2D` |
| 🎵 Audio and motion | [Dance-to-Music](#dance-to-music) | Dance | Music · API alias `D2M` |
| 🎵 Audio and motion | [Speech-to-Gesture](#speech-to-gesture) | Speech, optional caption | Co-speech gesture · API alias `S2G` |
| 🤖 Embodied motion | [Robot Motion Control](#robot-motion-control) | Commands, state, or primitives | Executable robot-state sequence |

## Language And Motion 💬

### Text-to-Motion

**Input:** one natural-language motion description, with an optional requested
duration.

**Output:** one motion sequence.

`T2M` is an accepted API alias. Offline, streaming, diffusion,
flow-matching, autoregressive, masked-token, and latent generators all remain
the same task.

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
`Multi-Person Motion` are not separate public task labels.

## Conditioned Motion 🎛️

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

## Motion Transformation And Restoration ✂️

### Motion Editing

**Input:** a source motion and a semantic edit instruction or attribute.

**Output:** an edited motion that applies the requested change while preserving
unmodified content.

Style/content editing and free-form MotionFix instructions are benchmark tracks
of the same task.

### Motion Repair

**Input:** a corrupted motion and an observed repair support or method-native
corruption estimate.

**Output:** a restored motion that preserves valid source content.

Oracle-mask and predicted-mask protocols are benchmark tracks, not separate
tasks.

### Motion Reconstruction

**Input:** one motion sequence.

**Output:** a reconstruction of the same sequence through a tokenizer,
autoencoder, codec, or representation bottleneck.

Reconstruction is a motion-to-motion task. MPJPE, root error, FID, codebook
usage, and physical diagnostics are evaluation measures rather than task names.

## Audio And Motion 🎵

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

## Embodied Motion 🤖

### Robot Motion Control

**Input:** navigation commands, target velocities, control state, or a motion
primitive schedule for a specified robot.

**Output:** an executable robot-state or joint-control sequence.

Human-to-robot retargeting is a
[Motion Toolkit](../motion/README.md) conversion route, not this task.

## Naming Contract ✅

- Public task fields use only labels from
  [`taxonomy.json`](taxonomy.json).
- Leaderboards use `Task · Dataset/Protocol`, for example
  `Text-to-Motion · HumanML3D`.
- Prediction, in-betweening, sparse keyframes, and TP2M stay under Temporal
  Motion Completion.
- Architecture terms such as diffusion, autoregressive, streaming, latent,
  zero-shot, and multimodal never become task labels.
- Representation conversion, body-model conversion, retargeting, and character
  export remain Motion Toolkit operations.
