# HumanML3D → MotionStreamer-272 validation (2026-11-09)

The native bridge now derives SMPL outgoing-frame local rotations from the
recovered positions on `bone_offsets_canon272.npy`. It no longer maps HumanML
incoming-bone frames through the SMPL parent tree. Single-child twist is
stabilized against the preceding frame. Native joint positions remain intact,
up to MS272's floor, origin and initial-heading canonicalization.

The subsequent cleanup removes the alternate `bridge` and `ik_kwargs` options
entirely. Both 272 and SMPL output adapters use the shared position IK module;
`source_fps`, `target_fps` and `target_len` are direct API parameters. Historical
HumanML rotation initializers, end-effector mixing, CLI selectors and the VerMo
initializer option are deleted. SOMA extra landmarks supply weighted constraints
to the same solver, rather than using a duplicate implementation.

The cleanup passes **85 related tests**, including explicit rejection of
removed options, timing checks on both outputs, landmark constraints through
the shared solver, and independence from the redundant HumanML rotation
channels. Walking and golf outputs are bitwise identical to the preceding
position-IK fix; all four real-motion errors below are unchanged. The former
SOMA plain/guided solvers were compared with the shared solver in both `bone`
and `parent_frame` modes: all four comparisons were bitwise equal.

## Real-motion measurements

All measurements below use physical-scale HumanML features at 20 fps, converted
to 30 fps, and compare decoded stored positions with independent NumPy SMPL FK
on the canonical 272 offsets. No per-frame rigid alignment or pose optimization
is applied to these errors. Root positions coincide to numerical precision.

| HumanML ID | Motion | Before mean error (mm) | After mean error (mm) |
| --- | --- | ---: | ---: |
| 004822 | Walking | 116.54 | 38.77 |
| 001840 | Kicking | 136.81 | 37.32 |
| 014160 | Waving | 78.68 | 39.20 |
| 014457 | Golf swing | 175.60 | 36.65 |

For walking, right-wrist mean error falls from **473.37 to 62.46 mm**, and
the maximum joint/frame error falls from **557.51 to 78.12 mm**. The remaining
36–39 mm mean error is not an exact-consistency claim: HumanML positions retain
their original skeleton proportions, whereas FK uses canonical 272 bone lengths
and branch geometry. A separate articulated sequence using matching canonical
bone geometry passes a **2e-6 m** stored-position/FK tolerance. Native rotations
remain a position-based approximation, not recovery of original SMPL twist.

Walking comes from the committed `tests/assets/humanml3d/004822_protocol.npz`.
The other three original `.npy` clips were read from the corresponding members
of [HumanML3D_new/new_joint_vecs.zip](https://huggingface.co/datasets/NamYeongCho/HumanML3D_new/blob/main/new_joint_vecs.zip).
Their source URLs, member names and SHA-256 hashes are saved in
`outputs/validation/hml263_motion272_20261109/sources/provenance.json`.

## Mesh comparison

Reproduce the walking comparison with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/validate_hml263_motion272.py \
  --input tests/assets/humanml3d/004822_protocol.npz \
  --baseline-motion272 outputs/validation/hml263_motion272_20261109/walk/motions.npz \
  --output-dir outputs/validation/hml263_motion272_cleanup/walk
```

The same tool accepts physical-scale `(T,263)` `.npy` files. It saves the input,
old and fixed representations, metrics, a PNG, an MP4 and a GIF. The old algorithm has been deleted, including from the validation tool.
Comparison rendering requires a saved baseline `.npy` or `.npz` containing
`before_272` at the same frame rate and length. Use `--no-render` without a
baseline to validate only the current converter.

Both sides use the committed Benchmark Hub **neutral SMPL** template, its four
skinning influences and rank-64 pose correctives. No additional mesh fitting is
performed. MS272 root positions are treated as pelvis positions, not raw SMPL
`transl`. Before/after headings are aligned by one constant rigid rotation;
camera, floor and timeline are shared. Meshes are rendered with Matplotlib Agg
on CPU. These are actual skinned meshes, not generated illustrations.

Results for this run live under
`outputs/validation/hml263_motion272_20261109/{walk,golf}/comparison.{png,gif,mp4}`.

## Benchmark Hub audit

The broken default native bridge is **not the conversion performed by the
inspected gallery viewers**. They load precomputed `smpl_motion135` assets,
decode quantized translations/rotations, and skin the neutral SMPL template.

Source evidence:

- `tools/materialize_hml263_smpl_joints.py` defaults to `position_ik` and writes
  `motion_135`; `tools/convert_hml263_predictions.py` calls the same retargeter.
- `tools/materialize_tokenizer_reconstruction.py` explicitly uses
  the shared position IK for HumanML reconstructions.
- `tools/build_smpl_motion_gallery.py` loads prepared motion135/SMPL parameters
  through `tools/smpl_gallery_assets.py`, rather than converting HumanML263 to272.
- Temporal and part-control gallery builders read `motion_135` and
  `gt_motion_135` from evaluation NPZs.
- M2T reuses available input/GT motion135 and falls back to explicit position IK
  when converting HumanML features. Its GT rotations need not have identity
  terminal joints.

Live manifests, descriptor chunks and binary rotation ranges were fetched on
2026-11-09. The audit sampled the first three cases of each listed method:

| Live gallery | Published population | Sampled methods | Asset samples |
| --- | ---: | --- | ---: |
| T2M HumanML3D | 4,042 | MDM, MoMask | 6 |
| Motion reconstruction | 4,042 | T2M-GPT, MoMask | 6 |
| Part control, left-wrist sparse XYZ | 4,012 | CondMDI, MaskControl | 6 |
| Temporal completion, first-frame condition | 4,012 | CondMDI, FlowMDM | 6 |
| M2T HumanML3D | 4,400 | Input/GT SMPL | 3 |

All 27 sampled assets decode to finite motion135 rotations. The 24 generated
samples have identity terminal local rotations, consistent with position IK
and the source builders, rather than the erroneous incoming-frame mapping.
This fingerprint is supporting evidence of the pipeline, **not** an independent
ground-truth accuracy test. M2T's three input/GT motions retain nonidentity
terminal rotations, which is expected for existing SMPL poses.

Therefore the good Benchmark visuals are compatible with the native bridge bug:
they use a different path. This audit does **not** certify every published case,
all settings, or metric caches. Previously generated native 272 files must be
regenerated if used downstream; no published gallery assets or benchmark scores
were overwritten by this fix. Browser quantized assets are visualization-only;
they are not inputs to evaluation metrics.

The downloaded live snapshots, sampled rotation hashes and per-case records
are in `outputs/validation/hml263_motion272_20261109/benchmark_audit.json`.

A [follow-up generated-motion audit](generated_smpl_validation.md) distinguishes
these dataset inputs from model predictions, tests saved MoMask/MDM/KIMODO
inference outputs, and compares 36 published model assets against archived
joint tracks, including residuals and outliers.
