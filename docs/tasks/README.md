# Motius Task Registry

This registry is the single vocabulary used by pipelines, model cards, the
Model Zoo, and benchmark documentation. It deliberately keeps four entities
separate:

- A **task** is an input/output contract.
- A **track** narrows a task with a conditioning pattern or operating setting.
- A **benchmark** binds a task to a dataset, split, selection rule, evaluator,
  and metric implementation.
- A **method** implements one or more tasks.

Dataset names, model architectures, motion representations, and training
objectives are not task names. The machine-readable source is
[`taxonomy.json`](taxonomy.json).

## Language And Motion

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

## Conditioned Motion

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

## Motion Transformation And Restoration

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

## Audio And Motion

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

## Embodied Motion

### Robot Motion Control

**Input:** navigation commands, target velocities, control state, or a motion
primitive schedule for a specified robot.

**Output:** an executable robot-state or joint-control sequence.

Human-to-robot retargeting is a
[Motion Toolkit](../motion/README.md) conversion route, not this task.

## Naming Contract

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
