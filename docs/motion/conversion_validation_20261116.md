# Conversion validation — 2026-11-16

## Acceptance withdrawn after user visual review

The user rejected all four presented videos on 2026-11-16. The numerical
checks below do not establish visually correct SMPL retargeting. This work
remains open; the previous completion claim is withdrawn.

## Numerical results (not visual acceptance)

The reproduced timing, single-bone antipode flip, SOMA parent-frame flip and
mesh-floor errors are fixed. **73 relevant tests pass** in the current checkout.
The new audit converts **26 full-precision cases**: seven saved MoMask/MDM
predictions with both source-root-preserving and grounded settings, plus six
KIMODO predictions through both position and rotation routes. No neural inference
was rerun. The actual gallery neutral rig is used; this does not certify all
SMPL shapes, historical published assets or perfect anatomical twist.

The prior November 9 overlay audit was insufficient: it removed root motion
frame by frame and normalized unequal timelines. The replacement uses one
physical output clock, fixed world coordinates, two fixed views and Y=0 ground.
Source skeletons are not aligned to fitted meshes in the rendered overlays.

## Reproduced failures and fixes

- FPS conversion stretched the sample interval using endpoint `linspace`.
  For 40 frames at 20→30 fps, a unit-speed trajectory deviated by up to
  16.38 mm from its physical timestamps. Sampling now uses `k / target_fps`,
  retains the frame-count duration convention and holds the last observation.
  Explicit `target_len` continues to mean intentional time warping.
- A smooth 1°/frame bone crossing its rest antipode produced a 32.8° local
  rotation step on the gallery rig. Direct transport from the previous frame
  removes that rest-pose singularity. The separate projection utility no longer
  mutates its caller's bone vector. HumanML135 enables continuity by default.
- SOMA's default weak joint-to-parent reference becomes unstable for nearly
  collinear spine vectors. The generated batch/walk cases reach 171.65°/161.71°
  steps in the old default. The default is now child-bone IK with continuity;
  real observed head/hand/toe landmark constraints remain in the shared solver.
- Five Adam steps at 0.02 introduced foot jitter in the root-control case.
  At fixed bone-mode initialization, 30 steps at 0.005 reduced mean output foot
  speed during source contact from 0.225 to 0.050 m/s (source: 0.032 m/s).
- Joint centers do not locate mesh soles. HumanML `floor_align=True` now applies
  one additional clip-wide mesh offset, recorded in output metadata. Explicit
  source translations take precedence. Skeleton-only adapters retain the joint
  behavior and explicitly report `mesh_floor_aligned=False`.
- SOMA position mode previously shifted its root independently in every frame
  based on foot-joint minima. It now uses the same single mesh-ground offset as
  rotation mode. Root displacement is not changed by this grounding operation.

## Same-source SOMA position results

Errors below compare with original source joints in world coordinates, including
intentional floor offsets. Contact means source foot-joint height within 4 cm of
that joint's sequence minimum and horizontal speed below 0.1 m/s at both interval
ends. This is a diagnostic heuristic, not annotated contact ground truth.

| Clip | MPJPE, mm (before → after) | Maximum local step, degrees | Output contact speed, m/s |
| --- | ---: | ---: | ---: |
| single_0 | 71.45 → 42.94 | 31.42 → 10.89 | 0.138 → 0.071 |
| batch_0 | 68.50 → 42.31 | 171.65 → 8.48 | 0.195 → 0.053 |
| batch_1 | 79.45 → 43.46 | 8.96 → 11.07 | 0.007 → 0.009 |
| multiprompt_0 | 70.82 → 47.12 | 161.71 → 24.32 | 0.516 → 0.089 |
| multiprompt_postprocess_0 | 75.94 → 57.04 | 101.21 → 11.97 | 0.412 → 0.058 |
| root_control_0 | 69.03 → 46.09 | 43.21 → 10.45 | 0.154 → 0.050 |

The seven ungrounded HumanML outputs preserve sampled source root positions to
under 0.001 mm. Their decoded target positions agree with independent
`numpy.interp` physical-time sampling to under 0.001 mm. All 26 emitted rotations
pass independent FK comparison with fitted joints (maximum 0.000651 mm).
All 19 explicitly grounded outputs have minimum vertex Y within 1e-5 m of zero.
Grounding tests also verify unchanged frame-to-frame root displacement and
source-translation precedence.

## Remaining representational limits

This is approximate skeleton retargeting. Unobserved twist cannot be recovered
uniquely, and different bone lengths mean preserving root motion and source
joint positions simultaneously is generally impossible. Grounding establishes
a clip-wide floor, not foot locking. Residual foot sliding is measured rather
than hidden: position-route contact speeds are 0.009–0.089 m/s; the unchanged
rotation route includes about 0.150 m/s on the postprocessed multiprompt clip.
True SOMA rotations remain preferred where available. No claim of physically
perfect contacts or uniform accuracy over every published case is made.

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/validate_generated_conversion.py \
  --inputs outputs/validation/native_method_architecture \
  --output-dir outputs/validation/conversion_fix_20261116/accepted --render
```

Requires PyTorch, NumPy, SciPy, mmengine, smplx, Matplotlib, imageio and
imageio-ffmpeg. The current run used `/tmp/motius-conversion-venv/bin/python`.
KIMODO is 30 fps, verified in the saved released-checkpoint config; MoMask/MDM
physical HumanML features are 20 fps. Source tensor files and every gallery rig
file are SHA-256 hashed in `metrics.json`. Each NPZ contains the same-source
reference, fitted joints, emitted motion135, actual skinned vertices and physical
timestamps. MP4/PNG pairs cover grounded MoMask, grounded MDM, and both SOMA
routes on the same multiprompt input. Four midpoint panels and a 16-frame
contact sheet were visually inspected; `temporal_contact_sheet.jpg` records the
sampled beginning, middle and ending frames.

The validation directory retains the pre-change metrics, intermediate runs,
test log and debugging history. Historical files are not overwritten. Existing
native 272 outputs and SMPL previews produced through changed paths must be
regenerated by their owners before reuse. No public benchmark assets or scores
were republished, and no unrelated local changes were committed.
