# Motius Model Zoo

The Model Zoo indexes integrated **method packages** and their public artifacts.
Task definitions live in the [Task Registry](../tasks/README.md); measured
results live in the [Benchmark Hub](../leaderboards/README.md). A multi-task
method appears in every applicable task row below, but only once in the
alphabetical method catalog. Model integrations without a stable Motius task
contract remain in the catalog but do not appear in the Task Index.

<p align="center">
  <a href="../tasks/README.md">🧭 Task Registry</a> ·
  <a href="../leaderboards/README.md">📊 Benchmark Hub</a> ·
  <a href="../motion/README.md">🔄 Motion Toolkit</a> ·
  <a href="release_policy.md">✅ Release Policy</a>
</p>

Architecture, training objective, dataset, and motion representation are method
metadata. They are never used as task categories.

## Task Index 🧭

| Task | Contract | Integrated methods |
| --- | --- | --- |
| [Text-to-Motion](../leaderboards/README.md#text-to-motion) | Text → motion · [HumanML3D](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) · [Unitree G1](../leaderboards/t2m_unitree_g1.md) | [ARDY](ardy.md) · [CondMDI](condmdi.md) · [DART](dart.md) · [FlowMDM](flowmdm.md) · [HY-Motion T2M](hymotion_t2m.md) · [KIMODO](kimodo.md) · [MaskControl](maskcontrol.md) · [MDM](mdm.md) · [MLD](mld.md) · [MoGenTS](mogents.md) · [MoMask](momask.md) · [MotionCLR](motionclr.md) · [MotionGPT](motiongpt.md) · [MotionLCM](motionlcm.md) · [MotionMillion](motionmillion.md) · [MotionStreamer](motionstreamer.md) · [OmniControl](omnicontrol.md) · [PRISM](prism.md) · [T2M-GPT](t2mgpt.md) · [TM2D](tm2d.md) · [UniMuMo](unimumo.md) · [ViMoGen](vimogen.md) |
| [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | Motion → caption | [MotionGPT](motiongpt.md) · [MotionGPT3](motiongpt3.md) · [TM2T](tm2t.md) · [UniMuMo](unimumo.md) · [VerMo](vermo.md) |
| [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | Ordered prompts → continuous motion | [ARDY](ardy.md) · [FlowMDM](flowmdm.md) · [KIMODO](kimodo.md) · [MotionStreamer](motionstreamer.md) · [PRISM](prism.md) |
| [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | Interaction text → shared-frame actors | [InterGen](intergen.md) · [InterMask](intermask.md) |
| [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) | Observed frames ± text → complete motion | [CondMDI](condmdi.md) · [FlowMDM](flowmdm.md) · [KIMODO](kimodo.md) · [MaskControl](maskcontrol.md) · [MotionStreamer](motionstreamer.md) · [OmniControl](omnicontrol.md) · [PRISM](prism.md) |
| [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | Numeric pose or trajectory constraints → motion | [ARDY](ardy.md) · [CondMDI](condmdi.md) · [DART](dart.md) · [KIMODO](kimodo.md) · [MaskControl](maskcontrol.md) · [OmniControl](omnicontrol.md) |
| [Part-Level Motion Control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | Body-region semantics → composed motion | No release-complete package; [MaskControl](maskcontrol.md#validation-status) exposes an experimental route |
| [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | Motion + edit → edited motion | [MotionCLR](motionclr.md) |
| [Motion Repair](../leaderboards/README.md#motion-repair-fixed-support-protocol) | Corrupted motion + support → restored motion | Benchmark protocol available; no standalone package |
| [Motion Reconstruction](../leaderboards/README.md#motion-reconstruction-humanml3d) | Motion → bottleneck reconstruction | Benchmark protocol available; no standalone package |
| [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | Music ± text → dance | [Bailando](bailando.md) · [EDGE](edge.md) · [TM2D](tm2d.md) · [UniMuMo](unimumo.md) |
| [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) | Dance → music | [UniMuMo](unimumo.md) |
| [Speech-to-Gesture](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) | Speech ± caption → gesture | No release-complete package |

MotionBricks remains a Model Zoo method integration, but it is intentionally
absent from the Task Index until Motius defines a stable task and benchmark
contract for its upstream G1 runtime.

## Method Catalog 📦

| Method | Task coverage | Native space | Artifacts |
| --- | --- | --- | --- |
| [ARDY](ardy.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `ARDY-330 / G1` | [↗ Official weights](https://huggingface.co/collections/nvidia/ardy) |
| [Bailando](bailando.md) | [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | `AIST++ SMPL-24 joints` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP) |
| [CondMDI](condmdi.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d) |
| [DART](dart.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `DART276` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-dart-humanml3d) |
| [EDGE](edge.md) | [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | `EDGE-151` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP) |
| [FlowMDM](flowmdm.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | `HumanML3D-263 / BABEL-135` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d) |
| [HY-Motion T2M](hymotion_t2m.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HY-Motion-201` | [📦 Full](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0) · [📦 Lite](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) |
| [InterGen](intergen.md) | [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | `paired InterHuman-262` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-intergen-interhuman) |
| [InterMask](intermask.md) | [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | `paired InterHuman-262` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-intermask-interhuman) |
| [KIMODO](kimodo.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `SOMA / G1 / SMPL-X` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp) |
| [MaskControl](maskcontrol.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-maskcontrol-humanml3d) |
| [MDM](mdm.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) |
| [MLD](mld.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) |
| [MoGenTS](mogents.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) |
| [MoMask](momask.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d) |
| [MotionBricks](motionbricks.md) | **Not registered** | `G1 413D / 414D / 418D` | [↗ Official code and weights](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks) |
| [MotionCLR](motionclr.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d) |
| [MotionGPT](motiongpt.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT-HumanML3D) |
| [MotionGPT3](motiongpt3.md) | [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D) |
| [MotionLCM](motionlcm.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D latent` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) |
| [MotionMillion](motionmillion.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `MotionStreamer-272` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272) |
| [MotionStreamer](motionstreamer.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | `MotionStreamer-272` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) |
| [OmniControl](omnicontrol.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `HumanML3D-263` | [📄 Paper](https://arxiv.org/abs/2310.08580) |
| [PRISM](prism.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | `PRISM Motion-138` | [📦 1.0](https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d) · [📦 KT](https://huggingface.co/ZeyuLing/motius-prism-kt-humanml3d) |
| [T2M-GPT](t2mgpt.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) |
| [TM2D](tm2d.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | `TM2D-287` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-TM2D-HumanML3D-AISTPP) |
| [TM2T](tm2t.md) | [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D) |
| [UniMuMo](unimumo.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard), [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard), [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) | `HumanML3D-263 / Encodec audio` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-UniMuMo) |
| [VerMo](vermo.md) | [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `VerMo-138` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D) |
| [ViMoGen](vimogen.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `DART276` | [📦 Weights](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) |

## Package Contract ✅

| Component | Public requirement |
| --- | --- |
| `ModelBundle` | Owns modules, checkpoint metadata, and serialization |
| Task pipeline | For registered coverage, exposes stable task-facing inputs and physical-space outputs |
| Runtime integration | May remain unregistered when Motius has no stable task and benchmark contract |
| Model card | Declares exact task coverage or unregistered status, native representation, FPS, artifacts, and attribution |
| Evaluation | Persists results from a named benchmark protocol |
| Representation bridge | Reports conversion diagnostics whenever native and evaluation spaces differ |

Read the [release policy](release_policy.md),
[architecture guide](../architecture.md), and
[development guide](../development.md) before adding a method.
