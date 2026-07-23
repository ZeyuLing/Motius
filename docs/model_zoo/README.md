# Motius Model Zoo

The Model Zoo is the method-level index for Motius. Every entry links to a model
card that records the released task API, native motion representation,
checkpoint, evaluation protocol, paper, original code, and license terms.

Task labels follow the
[canonical Motius task definitions](../tasks/README.md). Architecture properties
such as autoregressive, diffusion, latent, streaming, and zero-shot stay in the
method description rather than becoming additional tasks.

**Browse by task:** [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) ·
[Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) ·
[Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) ·
[Sequential Generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) ·
[Body-Part Condition](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) ·
[Kinematic Control](../tasks/README.md#kinematic-control) ·
[Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) ·
[Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) ·
[Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) ·
[Two-Person Text-to-Motion](../tasks/README.md#two-person-text-to-motion) ·
[Robot Motion Control](../tasks/README.md#robot-motion-control)

## Text And Motion

- **[MDM](mdm.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d)
- **[T2M-GPT](t2mgpt.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d)
- **[MoMask](momask.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d)
- **[MoGenTS](mogents.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d)
- **[MLD](mld.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d)
- **[MotionLCM](motionlcm.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D latent` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d)
- **[MotionGPT](motiongpt.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT-HumanML3D)
- **[MotionGPT3](motiongpt3.md)** - [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D)
- **[TM2T](tm2t.md)** - [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D)
- **[VerMo](vermo.md)** - [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `VerMo-138` · [Weights](https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D)
- **[MotionMillion](motionmillion.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `MotionStreamer-272` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272)
- **[HY-Motion T2M](hymotion_t2m.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HY-Motion-201` · [Full](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0)
  · [Lite](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite)
- **[ViMoGen](vimogen.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `DART276` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d)

## Temporal, Editing, And Control

- **[FlowMDM](flowmdm.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
  · `HumanML3D-263 / BABEL-135` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d)
- **[MotionStreamer](motionstreamer.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
  · `MotionStreamer-272` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272)
- **[PRISM](prism.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
  · `PRISM Motion-138` · [1.0](https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d)
  · [KT](https://huggingface.co/ZeyuLing/motius-prism-kt-humanml3d)
- **[DART](dart.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Kinematic Control](../tasks/README.md#kinematic-control)
  · `DART276` · [Weights](https://huggingface.co/ZeyuLing/motius-dart-humanml3d)
- **[CondMDI](condmdi.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Kinematic Control](../tasks/README.md#kinematic-control)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d)
- **[MaskControl](maskcontrol.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Kinematic Control](../tasks/README.md#kinematic-control)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/motius-maskcontrol-humanml3d)
- **[OmniControl](omnicontrol.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Kinematic Control](../tasks/README.md#kinematic-control)
  · `HumanML3D-263` · [Paper](https://arxiv.org/abs/2310.08580)
- **[MotionCLR](motionclr.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d)
- **[KIMODO](kimodo.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Condition](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard),
  [Kinematic Control](../tasks/README.md#kinematic-control)
  · `SOMA / G1 / SMPL-X` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp)
- **[ARDY](ardy.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Sequential Generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard),
  [Kinematic Control](../tasks/README.md#kinematic-control)
  · `ARDY-330 / G1` · [Weights](https://huggingface.co/collections/nvidia/ardy)

MaskControl also exposes experimental
[Body-Part Condition](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard)
and Sequential Generation routes. Their current validation boundary is recorded
in its [model card](maskcontrol.md#validation-status), rather than being
advertised as a release-complete capability.

## Audio-Driven Motion

- **[Bailando](bailando.md)** - [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard)
  · `AIST++ SMPL-24 joints` · [Weights](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP)
- **[EDGE](edge.md)** - [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard)
  · `EDGE-151` · [Weights](https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP)
- **[TM2D](tm2d.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard)
  · `TM2D-287` · [Weights](https://huggingface.co/ZeyuLing/Motius-TM2D-HumanML3D-AISTPP)
- **[UniMuMo](unimumo.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard),
  [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard),
  [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard)
  · `HumanML3D-263 / Encodec audio` · [Weights](https://huggingface.co/ZeyuLing/Motius-UniMuMo)

## Interaction And Robotics

- **[InterGen](intergen.md)** - [Two-Person Text-to-Motion](../tasks/README.md#two-person-text-to-motion)
  · `paired InterHuman-262` · [Weights](https://huggingface.co/ZeyuLing/motius-intergen-interhuman)
- **[InterMask](intermask.md)** - [Two-Person Text-to-Motion](../tasks/README.md#two-person-text-to-motion)
  · `paired InterHuman-262` · [Weights](https://huggingface.co/ZeyuLing/motius-intermask-interhuman)
- **[MotionBricks](motionbricks.md)** - [Robot Motion Control](../tasks/README.md#robot-motion-control)
  · `MotionBricks G1 413D / 414D / 418D` ·
  [Official code and weights](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)

## Integration Contract

A complete public entry provides:

1. A `ModelBundle` that owns modules, checkpoint metadata, and serialization.
2. A task-facing pipeline with stable physical-space outputs.
3. A model card with exact tasks, representation, FPS, weights, and attribution.
4. Evaluation results generated by a named protocol and persisted artifact.
5. Conversion or retargeting diagnostics whenever model-native and evaluation
   representations differ.

Read the [task definitions](../tasks/README.md),
[release policy](release_policy.md), [architecture guide](../architecture.md),
and [development guide](../development.md) before adding a method.
