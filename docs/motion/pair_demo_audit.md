# Two-person demo: audit and rebuild contract

## Status

The encoder/exporter corrections are covered by regression tests. The existing
InterX `G021T002A012R014` GIF and browser data **have not been regenerated** and
must not be described as corrected ground truth or a contact demo. They remain
accessible to avoid silently deleting published media.

This clip is action A012, **Point finger at**, according to the
[official Inter-X categories](https://github.com/liangxuy/Inter-X/blob/main/datasets/action_setting.txt).
The MotionHub source contains 145 frames at 30 fps; the old viewer uses only 72.
Its lack of touch is not, by itself, a spacing bug.

## Corrections

| Problem | Corrected contract |
| --- | --- |
| Skeleton canonicalized, mesh left in source frame | `return_transform=True` returns the same rigid transform for all geometry |
| Independent per-person floor offsets | One pair-wide floor and common display translation |
| Zero `betas` replaced with `raw_betas` | Honor the converted source shape, including intentionally zero shape |
| Original hands replaced with zeros | Require and pass both 45D hand poses; use explicit flat hand mean |
| Missing gender model silently replaced with neutral | Fail with the missing asset path |
| Default 72-frame crop | Default to the complete encoded clip (`T-1` frames) |
| GIF frame durations rounded down | Distribute centisecond durations over the requested frame rate |

Tests check all cross-actor joint distances, exact wrist contacts, elevated
actors, yaw changes, raw/Y-up input, velocities, source immutability, source
hand/shape forwarding and quantized skeleton/mesh alignment. These are
correctness tests, not a claim that a particular model generates good contacts.

```bash
pytest -q tests/test_interhuman262.py tests/test_interhuman_pair_geometry.py tests/test_interhuman_demo_builder.py
```

## Rebuilding the SMPL-H reference

Use already licensed SMPL-H models matching the source gender. Do not publish
model files. Choose a real contact action (A001 handshake, A000 hug, A016
high-five), inspect the whole sequence, and retain its source ID and parameters.

```bash
python tools/build_interhuman_representation_demo.py \
  --source interx-smplh --sample-id G001T000A001R005 \
  --data-root data/motionhub/interx --model-dir /path/to/licensed/models \
  --out-dir outputs/interx_handshake --device cpu --write-npz --write-mp4
```

Do not bring actors together by independent translation, scale, contact snapping,
or per-frame floor correction. Verify distances before/after conversion and
inspect opening, contact and separation phases before promoting the render.

## FBX without SMPL model files

A rigged FBX contains its own geometry, rest skeleton and skin weights. It can
be animated and rendered without downloading SMPL. Motius's Blender joint-track
bridge (`tools/blender_retarget_smpl22_joints.py`) supports this route.

However, converting **SMPL-H pose parameters to the original SMPL-H mesh** still
requires the body model. FBX is an output format, not a substitute for missing
geometry. A different FBX character has different proportions: label its output
as character retargeting, and measure contact error rather than presenting it as
an exact SMPL-H reference. Preserve one world frame for the pair throughout.

### FBX-only feasibility check (2026-09-09)

A local Blender EEVEE prototype transferred all 135 frames of handshake
`G001T000A001R005` onto two copies of the existing textured SMPL22 child rig,
using source axis-angle rotations and translations, without SMPL model files.
The maximum cross-actor pelvis-offset error against the source translations
was `4.0e-7 m`. However, the smallest nearest-vertex distance between the two
rendered character meshes was about `0.135 m`. This is a sampled vertex metric,
not a signed surface-contact/penetration test, and does **not** establish touch.

The prototype passes the no-SMPL rendering feasibility check but has **not passed
contact validation**. It is not included in public demo media. Source body
proportions and the target rig differ; preserving world transforms alone cannot
guarantee hand contact on another body. No independent actor translation, scaling,
or contact snapping was used to disguise this limitation.
