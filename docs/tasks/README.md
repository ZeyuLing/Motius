# Motius Task Definitions

This page is the canonical vocabulary for public Motius pipelines, model cards,
Model Zoo indexes, and leaderboards. A task is defined by its input and output
contract. Model architecture, training objective, inference schedule, and
deployment mode are properties of a method, not separate tasks.

Use the full labels below in documentation. The abbreviations `T2M`, `M2T`, and
`TP2M` remain valid API and benchmark shorthand, but must not replace the
canonical labels in task fields.

## Text-to-Motion

**Input:** one natural-language motion description, with an optional requested
duration.

**Output:** one motion sequence.

Use `Text-to-Motion` for both offline and streaming generators. Diffusion,
flow-matching, autoregressive, masked-token, latent, and zero-shot variants all
belong to this task. Results use the
[T2M HumanML3D leaderboard](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
when they follow its selected-caption protocol.

## Motion-to-Text

**Input:** one complete motion sequence.

**Output:** one natural-language motion description.

Motion captioning and motion understanding systems use `Motion-to-Text`.
Results use the
[M2T HumanML3D leaderboard](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
when they follow its full-motion protocol.

## Temporal Condition

**Input:** observed motion frames or frame-level constraints, optionally paired
with text.

**Output:** a completed motion sequence that preserves the supplied temporal
evidence.

`Temporal Condition` is the parent task for:

- **Prediction:** continue from an observed prefix.
- **In-betweening:** fill motion between observed boundaries.
- **Keyframe Condition:** generate around sparse or adaptive observed frames.
- **TP2M:** text-guided continuation from an observed motion prefix.

These are tracks of the
[Temporal Condition leaderboard](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
not separate top-level Model Zoo tasks.

## Sequential Generation

**Input:** an ordered sequence of text prompts, each with an interval or
duration.

**Output:** one continuous motion containing the requested actions and their
transitions.

Online prompt updates and multi-prompt composition use `Sequential Generation`
when they produce one continuous sequence. The canonical public protocol is the
[BABEL Sequential Generation leaderboard](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard).

## Body-Part Condition

**Input:** body-part-specific prompts or constraints assigned to temporal
intervals.

**Output:** one motion satisfying the requested part-level behavior while
preserving or generating the complementary body motion.

Use `Body-Part Condition` for semantic body-region control. Numeric joint,
trajectory, and end-effector evidence belongs to `Kinematic Control`. Public
results use the
[Body-Part Condition leaderboard](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard).

## Kinematic Control

**Input:** numeric keyframes, trajectories, joint positions or rotations,
end-effector targets, or other explicit kinematic evidence.

**Output:** one motion satisfying the supplied geometric constraints.

`Kinematic Control` replaces ambiguous labels such as `Motion Control`, `Joint
Control`, `Trajectory Control`, and `Spatial Control` in task fields. Those
phrases may still describe a specific control mode inside a model card.

## Motion Editing

**Input:** a source motion and a semantic edit condition.

**Output:** an edited motion that applies the requested change while preserving
unmodified content.

Style editing, content editing, and attention-based semantic editing use
`Motion Editing`. Free-form source-motion instructions are evaluated separately
on the
[Instruction Editing leaderboard](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard);
style/content protocols use the
[Motion Editing leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard).

## Music-to-Dance

**Input:** music audio or synchronized music features, optionally paired with a
text condition.

**Output:** a dance motion sequence.

Results use the
[Music-to-Dance leaderboard](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard).
PRISM-MCM is a method in this task, not a separate task.

## Dance-to-Music

**Input:** a dance motion sequence.

**Output:** synchronized music audio or music tokens.

Results use the
[Dance-to-Music leaderboard](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard).

## Speech-to-Gesture

**Input:** speech audio or speech features, optionally paired with a caption.

**Output:** a synchronized co-speech gesture sequence.

Results use the
[Speech-to-Gesture leaderboard](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard).

## Two-Person Text-to-Motion

**Input:** one natural-language interaction description.

**Output:** two synchronized person motions in one shared world frame.

Use `Two-Person Text-to-Motion`, not `Two-Person T2M`, `Interaction T2M`, or
`Multi-Person Motion`, when this is the released input/output contract.

## Robot Motion Control

**Input:** navigation commands, target velocities, control state, or a motion
primitive schedule for a specified robot.

**Output:** an executable robot-state or joint-control sequence.

Use `Robot Motion Control` for online locomotion and whole-body robot control.
Text-conditioned generation or human-to-robot retargeting remain separate
capabilities and should be labeled by their actual task or conversion route.

## Naming Rules

- A model card `Task` or `Tasks` field uses only canonical labels from this
  page.
- `TP2M`, prediction, in-betweening, and keyframe generation stay nested under
  `Temporal Condition`.
- `Motion Control` is not a public task label; choose `Temporal Condition`,
  `Body-Part Condition`, or `Kinematic Control`.
- `Streaming`, `online`, `zero-shot`, `diffusion`, `flow-matching`,
  `autoregressive`, `latent`, and `multimodal` describe implementation or scope.
- Motion representation conversion, body-model conversion, retargeting, and
  character export are [Motion Toolkit](../motion/README.md) operations rather
  than generative tasks.
