# HumanML → SMPL torso follow-up, 2026-11-16

The previous limb-frame candidate was rejected visually. Its controlled tests
and small joint errors did not establish plausible skinned bodies. This report
supersedes its visual-quality claims, not the saved evidence.

## Reproduced defect

On the same MDM `0_1` sample, independent short-bone IK produced chest local
X rotations of 72–90 degrees and neck rotations of -137 to -107 degrees.
The source's short spine segments pointed in different directions even though
the complete torso was approximately upright. Aligning each segment separately
concentrated opposing bends on adjacent vertebrae and folded the neck.
An independent full licensed SMPL-H male model reproduced the deformation at
the same poses, ruling out the gallery's compressed skinning as the sole cause.

## Change

`retarget_hml263_clip` now initializes from shared IK, then fits absolute
root-relative joint positions over the whole SMPL chain. A squared pose prior
has 30 times the weight on spine/neck joints; weaker limb-initialization and
pose-acceleration penalties retain limb orientation and temporal stability.
The root trajectory is held fixed. Existing clip-wide mesh floor alignment
remains separate. KIMODO is not routed through this new step.

Defaults: `anatomical_fit_iters=300`, `anatomical_prior_weight=0.001`.
Setting iterations to zero reproduces the unregularized diagnostic path.
This is approximate retargeting, not ground-truth twist recovery. The neutral
pose prior can attenuate extreme intended spinal/neck bends; the tested walking
and arm-raising clips do not certify acrobatics or all possible generations.

## Evidence

All files are in `outputs/validation/independent_smpl_20261116/`.

- `mdm.mp4`: rejected poses on the gallery and full independent SMPL-H model.
- `generated/`: all 26 repeated conversion cases, full-precision arrays and
  source/rig SHA-256 hashes in `metrics.json`.
- `comparison.json`: all seven HumanML clips' torso angles; all 12 KIMODO
  outputs are bitwise unchanged from the preceding candidate.
- `mdm_0_1_grounded_current.mp4` and
  `momask_batch2_gumbelTrue_1_grounded_current.mp4`: original source joints,
  rejected result, and current public converter; same source/time and camera.

For MDM `0_1`, chest X is now -0.3 to 7.0 degrees and neck X -24.7 to -17.9.
For MoMask `batch2_gumbelTrue_1`, chest X is now 2.3 to 4.9 degrees and neck X
-25.6 to -21.4. These are diagnostic measurements, not visual acceptance by
the user. Side/front inspection shows removal of the previously visible chest
fold and head inversion. No claim is made that the source motion itself is
perfect or that foot contact is solved for all clips.

Validation: 32 targeted tests and 45 broader regression tests passed. A new
regression introduces contradictory short-spine/head landmarks and checks
that full-chain fitting removes opposing large bends, improves position fit,
preserves root positions and works inside a caller's `torch.no_grad()` context.

The two final public-route videos were also extracted in full (60 consecutive
frames each) and inspected in order in `*_all_frames.jpg`. No remaining chest
fold/head inversion was observed in these two tracks. The other five HumanML
tracks received numerical regression checks, not a claim of exhaustive visual
acceptance. User acceptance remains open.
