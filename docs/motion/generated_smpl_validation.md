# Generated HumanML3D and SOMA → SMPL audit (2026-11-09)

This extends [the HumanML272 audit](hml263_motion272_validation.md). Its four
walking/kicking/waving/golf inputs were **dataset motions**, not model predictions.
The earlier 27 published-asset checks established decodability and pipeline
fingerprints, not fidelity to generated source joints. This follow-up tests
actual saved model predictions and published model result assets.

A [2026-11-16 follow-up](conversion_validation_20261116.md) fixes additional
timing, twist and grounding failures and supplies same-source fixed-world
validation. The rejection below applies to this historical audit and its assets.

## Acceptance status after visual review

**Not accepted as evidence that the current retargeting is visually correct.**
The overlay GIF and low mean joint errors are insufficient to certify SMPL mesh
quality. Independent FK agreement checks serialization self-consistency only;
it does not validate twist, anatomical plausibility, foot contact or temporal
continuity. Reinspection of multiple GIF frames does not isolate whether the
visible problems originate in generation, retargeting or presentation.

The published golf sample also has different reference/output durations:
MoMask and MDM archived tracks contain 60 display frames at 20 fps, whereas the
current mesh descriptors contain 93 at 30 fps. The audit normalizes timelines,
which conceals this duration difference; it is not a frame-rate-preservation
test. A matching full-precision input/output pair and unaligned, ground-referenced
mesh comparison are needed before accepting the published conversion quality.
Per-frame pelvis alignment in the existing GIF removes root motion and makes
foot sliding or ground contact impossible to assess reliably. Mesh rotation
quality remains unresolved; the numerical results below are diagnostic only.

## Full-precision generated inputs

Inputs are saved outputs of released-checkpoint source-parity runs, not ground
truth or synthetic poses. They are under
`outputs/validation/native_method_architecture/`:

- MoMask: `momask/native.pt`, physical (denormalized) HumanML263, three clips.
  Provenance runner: `tools/verify_momask_source_parity.py`.
- MDM: `mdm_inference_source/native.pt`, four HumanML263 clips from full diffusion
  inference. Provenance runner: `tools/verify_mdm_family_training_source.py`.
- KIMODO: `kimodo_text_migration/native.pt`, six SOMA77 predictions across single,
  batch, multiprompt, postprocessed multiprompt and root-control runs.
  Provenance runner: `tools/verify_kimodo_source_parity.py`.

MoMask and MDM use `retarget_hml263_clip` with 20→30 fps and no refinement:
physical HumanML263 → decoded positions → shared position IK → SMPL motion135.
Lengths change from 40/60 to 60/90 and from 48/40 to 72/60, respectively.

Each SOMA prediction is tested twice: positions through the shared IK with the
retargeter's default refinement/landmark constraints, and real SOMA global
rotations through rest-frame rotation retargeting. The latter is a genuine
rotation-to-rotation mapping, not the deleted HumanML incoming-bone shortcut.
`retarget_file` prefers the rotation route when global rotations are present.

| Input / route | Clips | Per-clip mean position error, mm | Per-clip pelvis-aligned mean, mm |
| --- | ---: | ---: | ---: |
| MoMask HumanML263 | 3 | 25.85–49.45 | 25.85–49.45 |
| MDM HumanML263 | 4 | 27.33–50.90 | 27.33–50.90 |
| KIMODO SOMA positions | 6 | 68.09–75.16 | 80.89–85.75 |
| KIMODO SOMA rotations | 6 | 49.93–58.83 | 76.22–80.40 |

Errors compare fitted SMPL22 joints with mapped source joints at the output time
samples. Pelvis alignment is translation only. SOMA root/foot alignment and
refinement can lower absolute error by compensating for skeletal differences;
subtracting each skeleton's pelvis removes that compensation. These are
approximate retargets, **not exact source-pose reconstruction**. SOMA residuals
are substantial; this audit does not attribute all residuals to bone lengths.
The position route's largest per-clip p99 rotation step is 41.35 degrees, versus
19.55 degrees for the rotation route, so temporal smoothness is not guaranteed.

For all 19 conversions, independently decoding emitted motion135 and running
NumPy FK matches the converter's fitted joints within **0.000651 mm** maximum.
This checks rotation serialization, hierarchy and translation conventions. It
must not be confused with source-to-target pose accuracy or with preservation
of original twist, which cannot be uniquely recovered from positions.

The audit uses the actual gallery's neutral beta=0 SMPL export: exact exported
rest joints, four skin influences and rank-64 pose correctives. The SOMA model
file loader is replaced by an adapter for this rig because the original neutral
SMPL model file is unavailable locally. Conversion algorithms are unchanged.
This validates the published viewer rig, not every gender/shape/full-model asset.

## Published generated result pages

Sampled the first 12 cases of the live T2M gallery for each of MDM, MoMask and
KIMODO: **36 actual published model motion assets**, not their GT entries.
Compared current quantized motion135 FK with the matching model/case joint
tracks from the [archived skeleton viewer manifest](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard/resolve/f11e3b394ec6/cases/manifest.json).

The archived joints are sampled at 10 fps; current mesh motions at 15 fps.
Reference joints are linearly interpolated using normalized display-frame time,
then each skeleton is pelvis-aligned. There is no rotation/scale alignment.
This is a comparison between published representations, not a comparison with
full-precision raw predictions from the exact historical inference run. The
archive does not identify whether its builder received decoded HumanML or
already materialized evaluation joints.

| Published method | Cases | Mean of case MPJPE, mm | Case mean range, mm |
| --- | ---: | ---: | ---: |
| MoMask | 12 | 20.86 | 18.53–26.33 |
| MDM | 12 | 31.26 | 26.42–36.97 |
| KIMODO | 12 | 31.99 | 26.20–43.41 |

These sampled result assets follow the generated poses on average and do not
show the systematic HumanML incoming/outgoing-frame failure. **They are not
uniformly accurate at every joint/frame**: case 004822 reaches 317.41 mm for MDM
and 174.85 mm for MoMask at the left foot. Sparse sampling/interpolation can
contribute, but without matching original full-precision historical predictions
this audit cannot establish the cause. Do not interpret low means or attractive
mesh renders as certification of every frame or all 4,042 cases.

The public viewer consumes precomputed SMPL motion135, not the former broken
HumanML263→272 bridge. No public assets, benchmark scores or evaluation caches
were replaced in this audit. Browser interaction was not automated; published
binary decoding, FK and mesh skinning were independently exercised.

## Artifacts and reproduction

All outputs are under `outputs/validation/generated_smpl_20261109/`:

- `generated_metrics.json`: all 19 conversions, input paths, source fit, FK and
  rotation-step metrics.
- `published_comparison.json`: all 36 published comparisons, including maxima.
- `generated_mesh_overlay.{png,gif,mp4}`: saved generated MoMask/MDM/KIMODO inputs
  converted with the current code.
- `published_mesh_overlay.{png,gif,mp4}`: actual published golf-caption outputs
  of those three models (case 014457).
- `soma_mesh_overlay.{png,gif,mp4}`: same generated SOMA multiprompt clip through
  the position and rotation routes.
- Per-clip NPZs retain source/fitted joints, rotations and, for fresh conversions,
  mesh vertices. Orange lines are source joints; blue surfaces are actual skinned
  meshes. Overlays use per-frame pelvis alignment and a fixed camera. Different
  models need not generate the same action timing or semantic quality.

From the repository root, with NumPy, PyTorch, smplx, SciPy, Matplotlib, requests,
imageio and imageio-ffmpeg installed:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python outputs/validation/generated_smpl_20261109/audit_generated_conversion.py
PYTHONPATH=. python outputs/validation/generated_smpl_20261109/check_published_generated.py
PYTHONPATH=. python outputs/validation/generated_smpl_20261109/render_audit.py
```

The scripts rely on the saved inference outputs, gallery rig and downloaded
manifest snapshots; they do not run fresh neural inference. The published check
also downloads asset byte ranges and therefore requires network access. Artifact
files live in the ignored output directory, while this report is repository
source. No additional production-code changes were necessary for this audit.
