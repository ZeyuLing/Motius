# Motius Model Zoo

The Model Zoo indexes integrated **method packages** and their public artifacts.
Task definitions live in the
[Task Registry](../tasks/README.md), while measured results live in the
[Leaderboard Hub](../leaderboards/README.md). A method may implement several
tasks and therefore appears in every applicable task index below, but only once
in the alphabetical method catalog.

Architecture, training objective, inference schedule, dataset, and motion
representation are method metadata. They are never used as task categories.

## Task Index

### Language And Motion

- **[Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard):**
  [ARDY](ardy.md), [CondMDI](condmdi.md), [DART](dart.md),
  [FlowMDM](flowmdm.md), [HY-Motion T2M](hymotion_t2m.md),
  [KIMODO](kimodo.md), [MaskControl](maskcontrol.md), [MDM](mdm.md),
  [MLD](mld.md), [MoGenTS](mogents.md), [MoMask](momask.md),
  [MotionCLR](motionclr.md), [MotionGPT](motiongpt.md),
  [MotionLCM](motionlcm.md), [MotionMillion](motionmillion.md),
  [MotionStreamer](motionstreamer.md), [OmniControl](omnicontrol.md),
  [PRISM](prism.md), [T2M-GPT](t2mgpt.md), [TM2D](tm2d.md),
  [UniMuMo](unimumo.md), and [ViMoGen](vimogen.md).
- **[Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard):**
  [MotionGPT](motiongpt.md), [MotionGPT3](motiongpt3.md),
  [TM2T](tm2t.md), [UniMuMo](unimumo.md), and [VerMo](vermo.md).
- **[Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard):**
  [ARDY](ardy.md), [FlowMDM](flowmdm.md), [KIMODO](kimodo.md),
  [MotionStreamer](motionstreamer.md), and [PRISM](prism.md).
- **[Text-to-Multi-Person Motion](../tasks/README.md#text-to-multi-person-motion):**
  [InterGen](intergen.md) and [InterMask](intermask.md).

### Conditioned Motion

- **[Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard):**
  [CondMDI](condmdi.md), [FlowMDM](flowmdm.md), [KIMODO](kimodo.md),
  [MaskControl](maskcontrol.md), [MotionStreamer](motionstreamer.md),
  [OmniControl](omnicontrol.md), and [PRISM](prism.md).
- **[Kinematic Motion Control](../tasks/README.md#kinematic-motion-control):**
  [ARDY](ardy.md), [CondMDI](condmdi.md), [DART](dart.md),
  [KIMODO](kimodo.md), [MaskControl](maskcontrol.md), and
  [OmniControl](omnicontrol.md).
- **[Part-Level Motion Control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard):**
  no release-complete package yet. MaskControl exposes an experimental route
  whose boundary is recorded in its
  [model card](maskcontrol.md#validation-status).

### Motion Transformation

- **[Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard):**
  [MotionCLR](motionclr.md).

Motion Reconstruction and Motion Repair currently have benchmark protocols but
no standalone release-complete Model Zoo package.

### Audio And Motion

- **[Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard):**
  [Bailando](bailando.md), [EDGE](edge.md), [TM2D](tm2d.md), and
  [UniMuMo](unimumo.md).
- **[Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard):**
  [UniMuMo](unimumo.md).
- **[Speech-to-Gesture](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard):**
  no release-complete package yet.

### Embodied Motion

- **[Robot Motion Control](../tasks/README.md#robot-motion-control):**
  [MotionBricks](motionbricks.md).

Human-to-robot retargeting belongs to the
[Motion Toolkit](../motion/README.md), not the Robot Motion Control task.

## Method Catalog

- **[ARDY](ardy.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard),
  [Kinematic Motion Control](../tasks/README.md#kinematic-motion-control)
  · `ARDY-330 / G1` · [Weights](https://huggingface.co/collections/nvidia/ardy)
- **[Bailando](bailando.md)** - [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard)
  · `AIST++ SMPL-24 joints` · [Weights](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP)
- **[CondMDI](condmdi.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Kinematic Motion Control](../tasks/README.md#kinematic-motion-control)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d)
- **[DART](dart.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Kinematic Motion Control](../tasks/README.md#kinematic-motion-control)
  · `DART276` · [Weights](https://huggingface.co/ZeyuLing/motius-dart-humanml3d)
- **[EDGE](edge.md)** - [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard)
  · `EDGE-151` · [Weights](https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP)
- **[FlowMDM](flowmdm.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
  · `HumanML3D-263 / BABEL-135` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d)
- **[HY-Motion T2M](hymotion_t2m.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HY-Motion-201` · [Full](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0)
  · [Lite](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite)
- **[InterGen](intergen.md)** - [Text-to-Multi-Person Motion](../tasks/README.md#text-to-multi-person-motion)
  · `paired InterHuman-262` · [Weights](https://huggingface.co/ZeyuLing/motius-intergen-interhuman)
- **[InterMask](intermask.md)** - [Text-to-Multi-Person Motion](../tasks/README.md#text-to-multi-person-motion)
  · `paired InterHuman-262` · [Weights](https://huggingface.co/ZeyuLing/motius-intermask-interhuman)
- **[KIMODO](kimodo.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard),
  [Kinematic Motion Control](../tasks/README.md#kinematic-motion-control)
  · `SOMA / G1 / SMPL-X` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp)
- **[MaskControl](maskcontrol.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Kinematic Motion Control](../tasks/README.md#kinematic-motion-control)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/motius-maskcontrol-humanml3d)
- **[MDM](mdm.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d)
- **[MLD](mld.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d)
- **[MoGenTS](mogents.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d)
- **[MoMask](momask.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d)
- **[MotionBricks](motionbricks.md)** - [Robot Motion Control](../tasks/README.md#robot-motion-control)
  · `MotionBricks G1 413D / 414D / 418D` ·
  [Official code and weights](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)
- **[MotionCLR](motionclr.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d)
- **[MotionGPT](motiongpt.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT-HumanML3D)
- **[MotionGPT3](motiongpt3.md)** - [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D)
- **[MotionLCM](motionlcm.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D latent` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d)
- **[MotionMillion](motionmillion.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `MotionStreamer-272` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272)
- **[MotionStreamer](motionstreamer.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
  · `MotionStreamer-272` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272)
- **[OmniControl](omnicontrol.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Kinematic Motion Control](../tasks/README.md#kinematic-motion-control)
  · `HumanML3D-263` · [Paper](https://arxiv.org/abs/2310.08580)
- **[PRISM](prism.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard),
  [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
  · `PRISM Motion-138` · [1.0](https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d)
  · [KT](https://huggingface.co/ZeyuLing/motius-prism-kt-humanml3d)
- **[T2M-GPT](t2mgpt.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d)
- **[TM2D](tm2d.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard)
  · `TM2D-287` · [Weights](https://huggingface.co/ZeyuLing/Motius-TM2D-HumanML3D-AISTPP)
- **[TM2T](tm2t.md)** - [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `HumanML3D-263` · [Weights](https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D)
- **[UniMuMo](unimumo.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard),
  [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard),
  [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard),
  [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard)
  · `HumanML3D-263 / Encodec audio` · [Weights](https://huggingface.co/ZeyuLing/Motius-UniMuMo)
- **[VerMo](vermo.md)** - [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
  · `VerMo-138` · [Weights](https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D)
- **[ViMoGen](vimogen.md)** - [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)
  · `DART276` · [Weights](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d)

## Package Contract

A release-complete method provides:

1. A `ModelBundle` that owns modules, checkpoint metadata, and serialization.
2. A task-facing pipeline with stable physical-space outputs.
3. A model card with exact tasks, native representation, FPS, artifacts, and
   attribution.
4. Evaluation artifacts generated by a named benchmark protocol.
5. Conversion diagnostics whenever native and evaluation representations
   differ.

Read the [release policy](release_policy.md),
[architecture guide](../architecture.md), and
[development guide](../development.md) before adding a method.
