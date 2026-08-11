# Motius Model Zoo

The Model Zoo indexes integrated **method packages** and their public artifacts.
Task definitions live in the [Task Registry](../tasks/README.md); measured
results live in the [Benchmark Hub](../leaderboards/README.md). A multi-task
method appears in every applicable task row below, but only once in the
alphabetical method catalog. Model integrations without a stable Motius task
contract remain in the catalog but do not appear in the Task Index.

<p align="center">
  <a href="../tasks/README.md">🧭 Task Registry</a> ·
  <a href="../datasets/README.md">Dataset Hub</a> ·
  <a href="../leaderboards/README.md">📊 Benchmark Hub</a> ·
  <a href="../training/README.md">Training Hub</a> ·
  <a href="../motion/README.md">🔄 Motion Toolkit</a> ·
  <a href="release_policy.md">✅ Release Policy</a>
</p>

Architecture, training objective, dataset, and motion representation are method
metadata. They are never used as task categories.

<!-- MOTIUS_MODEL_ZOO_METRICS:START -->

## Canonical Benchmark Snapshot

These compact rows are generated from the same machine-readable snapshots as the public Leaderboards and Model Cards. `Motius FID` is always computed in per-sample L2-normalized embedding space; `—` means that value has not been recomputed.

### Text-to-Motion · SMPL Skeleton

| Method | Version | MotionStreamer R@3 ↑ | MotionStreamer FID ↓ | Motius R@3 ↑ | Motius FID (normalized) ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| [HY-Motion T2M](hymotion_t2m.md) | 1.0B · 360f | 0.9516 | 13.5194 | 0.9122 | 0.0147 |
| [HY-Motion T2M](hymotion_t2m.md) | 0.46B · 360f | 0.9487 | 10.2711 | 0.9143 | 0.0157 |
| [MotionCanvas](motioncanvas.md) | 0.46B · 360f | 0.9521 | 8.2765 | 0.9103 | 0.0108 |
| [PRISM](prism.md) | KAFS cfg5 · e20 | 0.9477 | 11.8101 | 0.9192 | 0.0162 |
| [PRISM](prism.md) | 1.0 | 0.9241 | 19.0359 | 0.8725 | 0.0804 |
| [MotionStreamer](motionstreamer.md) | official | 0.8498 | 12.2110 | 0.6811 | 0.0488 |
| [UniMuMo](unimumo.md) | zero-shot | 0.1617 | 373.2192 | 0.2093 | 0.6788 |
| [TM2D](tm2d.md) | official E0190/E0020 | 0.2674 | 548.5562 | 0.4427 | 0.6703 |
| [MotionMillion / GoToZero](motionmillion.md) | 7B-train | 0.9236 | 3.0807 | 0.8579 | 0.0183 |
| [MotionMillion / GoToZero](motionmillion.md) | 3B-train | 0.9229 | 3.0658 | 0.8569 | 0.0185 |
| [ViMoGen](vimogen.md) | 1.3B prompt-rewrite | 0.6518 | 152.2095 | 0.5198 | 0.4911 |
| [DART](dart.md) | official | 0.7937 | 127.8302 | 0.7019 | 0.2029 |
| [CondMDI](condmdi.md) | official | 0.7016 | 121.8374 | 0.7024 | 0.1919 |
| [FlowMDM](flowmdm.md) | official | 0.7312 | 36.3767 | 0.7111 | 0.1239 |
| [MotionCLR](motionclr.md) | official | 0.5960 | 298.9693 | 0.6458 | 0.5746 |
| [MaskControl](maskcontrol.md) | official | 0.8475 | 107.8315 | 0.6615 | 0.5457 |
| [T2M-GPT](t2mgpt.md) | official | 0.7788 | 25.4913 | 0.7359 | 0.1148 |
| [MDM](mdm.md) | official | 0.7701 | 35.5169 | 0.7262 | 0.1438 |
| [MoMask](momask.md) | official | 0.8609 | 21.0729 | 0.8356 | 0.0782 |
| [MoGenTS](mogents.md) | official | 0.8138 | 109.8191 | 0.7138 | 0.0864 |
| [MotionGPT](motiongpt.md) | official | 0.6944 | 23.6811 | 0.6617 | 0.1024 |
| [MotionGPT3](motiongpt3.md) | official | 0.8817 | 20.9913 | 0.8425 | 0.0936 |
| [KIMODO](kimodo.md) | SMPL-X RP | 0.5818 | 117.0279 | 0.5570 | 0.4881 |
| [MLD](mld.md) | canonical | — | — | 0.7701 | 0.1399 |
| [MotionLCM](motionlcm.md) | canonical | — | — | 0.7736 | 0.1537 |

[Open the complete Text-to-Motion Leaderboard](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) for all three evaluators, physical metrics, protocol details, and case-level SMPL visualization.

### Music-to-Dance · AIST++

| Method | Version | FID_k ↓ | FID_g ↓ | Motius FID (normalized) ↓ | BeatAlign ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| [Bailando](bailando.md) | official epoch 10 | 28.1072 | 9.7006 | 0.3138 | 0.2268 |
| [EDGE](edge.md) | official checkpoint, seed 20260721 | 38.0561 | 20.6397 | 0.2503 | 0.2562 |
| [TM2D](tm2d.md) | official E0190/E0020, reference-free seed | 43.4186 | 20.4244 | 0.2623 | 0.1903 |
| [UniMuMo](unimumo.md) | official checkpoint, Motius parity-verified conversion | 17.7250 | 38.6446 | 0.2823 | 0.2430 |

[Open the complete Music-to-Dance Leaderboard](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) for diversity, physical metrics, synchronized audio, and interactive 3D results.

<!-- MOTIUS_MODEL_ZOO_METRICS:END -->

## Model Card Standard

Every catalog entry follows one public information contract: visual evidence,
model overview, runnable task APIs, measured evaluation, exact motion
representation, native training motion FPS, public preview FPS, and attribution.
Training FPS and preview FPS are separate clocks: a 30 fps preview never implies
that its checkpoint was trained at 30 fps. Public motion videos render every
timeline frame through the shared Three.js floor scene. Audio-conditioned
results retain synchronized audio, and methods without a validated SMPL bridge
show their native representation instead of a misleading fitted body. The cards
share the same navigation and section order while retaining method-specific
setup and reproduction notes.
See the [Model Card Template](MODEL_CARD_TEMPLATE.md) before adding or updating
an integration.

Validate the full catalog with:

```bash
python tools/normalize_model_cards.py
python tools/export_t2m_leaderboard_results.py --check
python tools/audit_model_card_media.py
python tools/sync_model_card_content.py
python tools/audit_model_card_format.py
python tools/audit_model_card_content.py
```

## Training Support

Training support is narrower than inference support. The following Model Zoo
packages currently have a registered Motius trainer, a public config, and a
documented resume path:

| Method | Training mode | Recipe |
| --- | --- | --- |
| [HY-Motion T2M](hymotion_t2m.md#training) | Flow-matching training; optional weight warm start | [`configs/hymotion_t2m/train_hymotion_t2m.py`](../../configs/hymotion_t2m/train_hymotion_t2m.py) |
| [PRISM](prism.md#training) | Continued training/fine-tuning from a released PRISM artifact | [`configs/prism/train_prism.py`](../../configs/prism/train_prism.py) |
| [HYMotion M2M (MotionCanvas)](motioncanvas.md#training) | Full M2M training from a HY-Motion T2M warm start | [`configs/motioncanvas/train_motioncanvas_0p46b.py`](../../configs/motioncanvas/train_motioncanvas_0p46b.py) |
| [ProtoMotions](protomotions.md#training) | Simulator-owned PPO motion-tracking training | [`configs/motion_tracking/protomotions_g1_bones_seed.yaml`](../../configs/motion_tracking/protomotions_g1_bones_seed.yaml) |
| [SONIC](sonic.md#training) | PPO/TRL universal-token tracking training | [`configs/motion_tracking/sonic_g1_bones_seed.yaml`](../../configs/motion_tracking/sonic_g1_bones_seed.yaml) |

The Motius Joint-Position TMR model is documented separately in the
[Evaluator Zoo](../evaluator_zoo/motius_joint_position.md#training). See the
[Training Hub](../training/README.md) for launch, data, precision, resume,
checkpoint, and output contracts. Other Model Zoo packages remain
checkpoint/inference integrations unless their card explicitly includes a
Motius training section.

## Unified Loading

Every release-complete artifact is self-describing and loads through one API:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MDM-HumanML3D",
    device="cuda",
)
motion = pipe.infer_text_to_motion(
    ["a person turns left and walks forward"],
    [120],
)
```

Artifacts name their supported tasks in `model_index.json`; each task is
available as the exact `infer_{task}` method. Required text encoders and
model-side inference dependencies live inside the same artifact. Separately
licensed body models and optional raw-input frontends are disclosed in the
corresponding card. The loader uses a closed Motius class allowlist and never
executes remote checkpoint code. Top-level keyword arguments configure the
Pipeline; model-construction options belong in `bundle_kwargs`.

## Inference Route Compatibility

The task API does not introduce a second inference implementation. Before the
unified contract, methods were invoked through several method-specific routes:

| Method family | Previous inference route |
| --- | --- |
| Most T2M and M2T packages | `infer_t2m` or `infer_m2t` |
| ARDY | `generate` and repeated `stream_step` calls |
| KIMODO | `text_to_motion`, `multi_prompt`, `constrained_motion`, and `infer_tp2m` |
| PRISM | `text_to_motion`, `temporal_condition`, and `sequential_generation` |
| MotionBricks | `rollout` |
| HY-Motion T2M | `pipeline(batch)` |
| UniMuMo dance tasks | `infer_music_to_motion` and `infer_motion_to_music` |

Canonical methods now resolve to those same routes. Native task methods remain
unchanged; simple compatibility names are the same Python function object as
their legacy entrypoint. Five adapters have strict parity coverage: ARDY
sequential streaming, HY-Motion T2M batch construction, and KIMODO scalar/batch
T2M, sequential prompt forwarding, plus native SOMA prefix completion. Their
arguments and nested outputs are covered by strict equality tests.

Run the release gate with:

```bash
python tools/audit_pipeline_task_apis.py
pytest -q tests/test_pipeline_task_api_parity.py
```

The current registry contains 47 artifacts, 81 artifact-task bindings, and 73
unique task methods across 42 public method Pipeline classes: 25 native
implementations, 43 identity aliases, five verified adapters, and zero
unverified routes. This includes the restricted official-runtime
`PromptHMRPipeline`, even though its licensed upstream checkpoint cannot be
redistributed as a Motius Hugging Face artifact.

All 47 classes named `Pipeline` or `*Pipeline` under `motius/pipelines` are
accounted for. The only exclusions are the abstract `BasePipeline`, the
automatic artifact-loading `Pipeline` facade, PRISM's internal Diffusers
backend, and the MotionCLIP evaluator/retrieval API.

The catalog below contains 42 model packages: 40 artifact-bearing method
Pipeline families, the trainable user-checkpoint BeyondMimic package, and the restricted
PromptHMR integration. Training-free, checkpoint-free utilities such as the source-only
D3 Motion Repair Pipeline are linked from the Task Index instead of being
presented as model packages.

## Task Index 🧭

| Task | Contract | Integrated methods |
| --- | --- | --- |
| [Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) | Text → motion · [SMPL Skeleton](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) · [Unitree G1 Skeleton](https://huggingface.co/spaces/ZeyuLing/t2m-unitree-g1-leaderboard) | [ARDY](ardy.md) · [CondMDI](condmdi.md) · [DART](dart.md) · [FlowMDM](flowmdm.md) · [HY-Motion T2M](hymotion_t2m.md) · [KIMODO](kimodo.md) · [MaskControl](maskcontrol.md) · [MDM](mdm.md) · [MLD](mld.md) · [MoGenTS](mogents.md) · [MoMask](momask.md) · [MotionCanvas](motioncanvas.md) · [MotionCLR](motionclr.md) · [MotionGPT](motiongpt.md) · [MotionGPT3](motiongpt3.md) · [MotionLCM](motionlcm.md) · [MotionMillion](motionmillion.md) · [MotionStreamer](motionstreamer.md) · [OmniControl](omnicontrol.md) · [PRISM](prism.md) · [T2M-GPT](t2mgpt.md) · [TM2D](tm2d.md) · [UniMuMo](unimumo.md) · [ViMoGen](vimogen.md) |
| [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | Motion → caption | [MotionGPT](motiongpt.md) · [MotionGPT3](motiongpt3.md) · [TM2T](tm2t.md) · [UniMuMo](unimumo.md) · [VerMo](vermo.md) |
| [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | Ordered prompts → continuous motion | [ARDY](ardy.md) · [FlowMDM](flowmdm.md) · [KIMODO](kimodo.md) · [MotionCanvas](motioncanvas.md) · [MotionStreamer](motionstreamer.md) · [PRISM](prism.md) |
| [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | Interaction text → shared-frame actors | [InterGen](intergen.md) · [InterMask](intermask.md) |
| [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) | Observed frames ± text → complete motion | [CondMDI](condmdi.md) · [FlowMDM](flowmdm.md) · [HYMotion M2M (MotionCanvas)](motioncanvas.md) · [KIMODO](kimodo.md) · [MaskControl](maskcontrol.md) · [MotionStreamer](motionstreamer.md) · [OmniControl](omnicontrol.md) · [PRISM](prism.md) · [ProjFlow](projflow.md) |
| [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | Numeric pose or trajectory constraints → motion | [ARDY](ardy.md) · [CondMDI](condmdi.md) · [HYMotion M2M (MotionCanvas)](motioncanvas.md) · [KIMODO](kimodo.md) · [MaskControl](maskcontrol.md) · [OmniControl](omnicontrol.md) · [ProjFlow](projflow.md) |
| [Part-Level Motion Control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | Body-region semantics → composed motion | [MaskControl](maskcontrol.md) · [MotionCanvas](motioncanvas.md) · [ProjFlow](projflow.md) |
| [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | Motion + edit → edited motion | [HYMotion M2M (MotionCanvas)](motioncanvas.md) |
| [Motion Repair](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard) | Corrupted motion ± support → restored motion | MoGenDiT · StableMotion · [MotionCanvas](motioncanvas.md) |
| [Motion Reconstruction](../leaderboards/README.md#motion-reconstruction-humanml3d) | Motion → bottleneck reconstruction | Benchmark protocol available; no standalone package |
| [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | Music ± text → dance | [Bailando](bailando.md) · [EDGE](edge.md) · [TM2D](tm2d.md) · [UniMuMo](unimumo.md) |
| [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) | Dance → music | [UniMuMo](unimumo.md) |
| [Speech-to-Gesture](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard) | Speech ± caption → gesture | No release-complete package |
| [Monocular Motion Capture](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) | Monocular RGB video → camera/world body motion | [GEM-SMPL](gem_smpl.md) · [GEM-X](gem_x.md) · [GVHMR](gvhmr.md) |
| [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | Reference motion + robot state → action | [Any2Track](any2track.md) · [BeyondMimic](beyondmimic.md) · [HumanoidGPT](humanoid_gpt.md) · [ProtoMotions](protomotions.md) · [SONIC](sonic.md) |

MotionBricks remains a Model Zoo integration without a registered public task.
Physical controllers use the [registered task contract](../tasks/motion_tracking.md).
MuJoCo and Isaac Lab are maintained as separate leaderboard settings so
engine-specific physics results are never mixed.

## Method Catalog 📦

| Method | Task coverage | Native space | Artifacts |
| --- | --- | --- | --- |
| [Any2Track](any2track.md) | [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | `G1 policy observation 156D` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2) |
| [ARDY](ardy.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `ARDY-330 / G1` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/motius-ardy-330-horizon40) |
| [Bailando](bailando.md) | [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | `AIST++ SMPL-24 joints` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP) |
| [BeyondMimic](beyondmimic.md) | [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | `G1 policy observation + embedded reference` | [Train and export](beyondmimic.md#training) |
| [CondMDI](condmdi.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d) |
| [DART](dart.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `DART276` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-dart-humanml3d) |
| [EDGE](edge.md) | [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | `EDGE-151` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP) |
| [FlowMDM](flowmdm.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | `HumanML3D-263 / BABEL-135` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-FlowMDM-HumanML3D) |
| [GEM-SMPL](gem_smpl.md) | [Monocular Motion Capture](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) | `SMPL-24` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-GEM-SMPL) |
| [GEM-X](gem_x.md) | [Monocular Motion Capture](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) | `SOMA-77` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-GEM-X) |
| [GVHMR](gvhmr.md) | [Monocular Motion Capture](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) | `SMPL-24` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-GVHMR) |
| [HY-Motion T2M](hymotion_t2m.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HY-Motion-201 / G1-38` | [📦 Full](https://huggingface.co/ZeyuLing/Motius-HYMotion-T2M-1.0) · [📦 Lite](https://huggingface.co/ZeyuLing/Motius-HYMotion-T2M-1.0-Lite) · [📦 G1](https://huggingface.co/ZeyuLing/Motius-HYMotion-G1) |
| [HumanoidGPT](humanoid_gpt.md) | [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | `G1-5010 policy observation 136D` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-HumanoidGPT-G1) |
| [InterGen](intergen.md) | [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | `paired InterHuman-262` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-intergen-interhuman) |
| [InterMask](intermask.md) | [Text-to-Multi-Person Motion](../leaderboards/text_to_multi_person_interhuman.md) | `paired InterHuman-262` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-intermask-interhuman) |
| [KIMODO](kimodo.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `SOMA / G1 / SMPL-X` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-KIMODO-SOMA-RP) |
| [MaskControl](maskcontrol.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md), [Part-Level Motion Control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-maskcontrol-humanml3d) |
| [MDM](mdm.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MDM-HumanML3D) |
| [MLD](mld.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MLD-HumanML3D) |
| [MoGenTS](mogents.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MoGenTS-HumanML3D) |
| [MoMask](momask.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MoMask-HumanML3D) |
| [MotionBricks](motionbricks.md) | **Not registered** | `G1 413D / 414D / 418D` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/motius-motionbricks-g1) |
| [HYMotion M2M (MotionCanvas)](motioncanvas.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md), [Motion Editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard), [Motion Repair](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard) | `MotionCanvas-198` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-MotionCanvas-0.46B) |
| [MotionCLR](motionclr.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d) |
| [MotionGPT](motiongpt.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT-HumanML3D) |
| [MotionGPT3](motiongpt3.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D) |
| [MotionLCM](motionlcm.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D latent` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionLCM-HumanML3D) |
| [MotionMillion](motionmillion.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `MotionStreamer-272` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionMillion-7B-HumanML272) |
| [MotionStreamer](motionstreamer.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | `MotionStreamer-272` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-MotionStreamer-HumanML272) |
| [OmniControl](omnicontrol.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md) | `HumanML3D-263` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/motius-omnicontrol-humanml3d) |
| [PRISM](prism.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Sequential Text-to-Motion](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | `PRISM Motion-138` | [📦 1.0](https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d) · [📦 KT](https://huggingface.co/ZeyuLing/motius-prism-kt-humanml3d) |
| [ProjFlow](projflow.md) | [Temporal Motion Completion](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard), [Kinematic Motion Control](../leaderboards/kinematic_motion_control.md), [Part-Level Motion Control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | `HumanML3D SMPL-22 joints` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/motius-projflow-humanml3d) |
| [ProtoMotions](protomotions.md) | [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | `G1 current state + four future references` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED) |
| [PromptHMR-Video](prompthmr.md) | **Restricted upstream runtime** | `SMPL-X` | [↗ Official weights](https://github.com/yufu-wang/PromptHMR#installation) |
| [SONIC](sonic.md) | [Motion Tracking](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | `G1 64D universal token + policy state` | [📦 Motius checkpoint](https://huggingface.co/ZeyuLing/Motius-SONIC-G1) |
| [T2M-GPT](t2mgpt.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-T2M-GPT-HumanML3D) |
| [TM2D](tm2d.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) | `TM2D-287` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-TM2D-HumanML3D-AISTPP) |
| [TM2T](tm2t.md) | [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `HumanML3D-263` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D) |
| [UniMuMo](unimumo.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion), [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard), [Music-to-Dance](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard), [Dance-to-Music](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard) | `HumanML3D-263 / Encodec audio` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-UniMuMo) |
| [VerMo](vermo.md) | [Motion-to-Text](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) | `VerMo-138` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D) |
| [ViMoGen](vimogen.md) | [Text-to-Motion](../leaderboards/README.md#text-to-motion) | `DART276` | [📦 Weights](https://huggingface.co/ZeyuLing/Motius-ViMoGen-1.3B-HumanML3D) |

## Package Contract ✅

| Component | Public requirement |
| --- | --- |
| `ModelBundle` | Owns modules, checkpoint metadata, and serialization |
| Task pipeline | For registered coverage, exposes stable task-facing inputs and physical-space outputs |
| Runtime integration | May remain unregistered when Motius has no stable task and benchmark contract |
| Model card | Declares exact task coverage or unregistered status, native representation, native training FPS, public preview FPS, artifacts, attribution, and Motius training recipe when supported |
| Evaluation | Persists results from a named benchmark protocol |
| Representation bridge | Reports conversion diagnostics whenever native and evaluation spaces differ |

Read the [release policy](release_policy.md), the
[Task Registry](../tasks/README.md), and the
[repository guide](../../README.md) before adding a method.
