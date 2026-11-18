# Reopened SMPL visual review — 2026-11-16

**Status: candidate under review, not accepted.** The user rejected all four
previous videos and identified twisted bodies / reversed joints. The preceding
completion claim is withdrawn. Numeric self-consistency is not a visual gate.

## Reproduced causes

A known SMPL pose with shoulder internal rotation and flexed elbows/knees was
forward-kinematically converted to its own joint positions, then reconstructed
with the previous position IK. Joint discrepancy was below 4e-8 m, yet mean
shoulder/elbow rotation error was about 35 degrees. Thus the mesh can twist
while all joint-position checks pass. On a separate known torso pose with
three 30-degree spine rotations, the previous reconstruction placed about
82 degrees at the branching chest joint.

The candidate uses positive SMPL elbow/knee hinge axes and the observed limb
bend plane to resolve this missing twist. Near-straight segments use twist
interpolated between observable bends in a transported frame, avoiding noisy
cross-product signs. Spine frames interpolate the lateral hip and shoulder
orientations, preserving outgoing bone directions while spreading axial twist.
These are anatomical priors, not recovery of arbitrary original twist.

A combined known-pose mesh comparison (including torso turns) gives:

| Reconstruction | Mean vertex error | Maximum vertex error |
| --- | ---: | ---: |
| Previous IK | 9.014 mm | 101.496 mm |
| Anatomical-frame candidate | 0.000242 mm | 0.018651 mm |

The two new regression tests check actual authored limb/spine rotations rather
than only joint positions. 75 related tests pass. Neither these tests nor the
controlled synthetic clip certify arbitrary generated actions.

## Generated-joint diagnostics

The previously presented MDM `mdm_0_1_grounded` sample had a minimum signed
elbow/knee flexion of -112.3 degrees and a 121.3-degree p95 off-axis component.
The candidate measures -5.0 degrees and 4.2 degrees, respectively. The MoMask
`batch2_gumbelTrue_1` sample changes from -121.9-degree minimum signed flexion
to +11.1 degrees. These are SMPL axis-angle diagnostics against anatomical hinge
priors, not labeled pose ground truth. Some near-extension and refined SOMA
frames retain about 5 degrees of negative flexion; they are not called perfect.
The reproducible script and per-clip before/candidate measurements are saved as
`measure_joint_axes.py` and `joint_axis_comparison.json` in the review directory.

## Videos

Artifacts are under `outputs/validation/conversion_twist_review_20261116/`:

- `controlled_twist.mp4`: known correct SMPL / rejected IK / candidate.
- `mdm_review.mp4`: separate source skeleton / rejected SMPL / candidate.
- `momask_batch2_gumbelTrue_1_grounded_review.mp4`: same layout for MoMask.
- `kimodo_multiprompt_0_positions_review.mp4`: same layout for SOMA positions.

All panels use the same input timeline and fixed camera/scale; no per-frame
alignment. Ground offsets are those of each explicitly grounded converter.
Meshes are real skinning results from the exported neutral SMPL rig. Controlled
reference and model-generated samples are labeled separately.

`generated/metrics.json` records a rerun of the 26 same-source numerical cases;
source tensor and rig hashes remain included. The direct SOMA rotation-transfer
route has **not** been changed by this candidate and is not re-certified. No
public assets are replaced. Current changes are uncommitted.

The comparison renderer is `tools/render_retarget_review.py`; the exact review
construction script and the rejected solver snapshot are retained alongside the
artifacts. The normal full-precision audit remains
`tools/validate_generated_conversion.py`.

All four video files completed successfully. Beginning, intermediate and ending
frames (16 sampled frames total) were inspected in `review_contact_sheet.jpg`.
This inspection is not a replacement for user acceptance of the visible result.

The user rejected this candidate again. The subsequent
[torso follow-up](conversion_torso_fix_20261116.md) reproduces chest/neck folding
with independent full SMPL-H skinning and adds whole-chain regularized fitting
for HumanML → SMPL135. The videos above remain rejected historical outputs.
