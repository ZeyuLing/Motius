# MoGenDiT / MoreDiff source provenance

Recovered from the user-provided `MoGenDiT-before-deletion-9db68e7e.zip`.
The filename identifies the snapshot; no independent Git commit identity is asserted.
`source_manifest.json` records the archive SHA-256 and original SHA-256 of each
imported file, before newline normalization and the changes below.

Included: MoreDiff and legacy MoGenDiT networks, Aplus BaseModel, EasyDiffusion,
Occam motion representation and degradation helpers, MoreDiffRefiner, required
articulate rotation math and Animo skeleton/optional ground simulation code.
Original comments and embedded attribution notices are retained. The included
PyTorch3D license is retained in `PYTORCH3D_LICENSE`. The supplied
archive contains no top-level license; no additional license grant is inferred.

Port changes:

- Normalize newlines and namespace imports under the corresponding Motius model, motion and diffusion modules;
  import only Aplus BaseModel rather than its unrelated model collection.
- Add empty package initializers where necessary. Import articulate math without
  its unrelated mesh/evaluator modules.
- Fix the non-windowed refiner's undefined `_mask` / `_keep_mask` variables.
- Pass the input tensor's device to both diffusion sampler calls, instead of
  using their import-time CUDA-0 default.
- Clamp displacement-triggered window cutoffs to the actual sampled window.
- Add Motius bundle/pipeline wrappers and strict local checkpoint loading. Network
  parameter names, RoPE formula, diffusion schedule and adaptive 0.1 threshold
  remain unchanged.

The Motius wrapper defaults to adaptive repair and windowing; ground simulation
is explicit opt-in. The original refiner defaults remain intact. Infrastructure settings, papers and deprecated experiments are not imported.

## Dedicated training migration

`motius/trainers/mogendit` ports the method-specific objective from
`trainer/my_trainer.py`, the mask scheduler, sampling helper, and the original
AdHocMotionData loading/cropping contract. The original training entrypoint and
configuration determine the defaults. `training_original_file_sha256` records
these sources in the manifest. Imports from missing `Aplus.data` and the unused
SMPL mesh geometric-loss helper are not needed by the active training objective;
the active kinematic consistency loss is already in OccamMotionRep.

Changes: use Motius BaseTrainer/registries and bundle forwarding; explicit data
paths; remove wall-clock RNG reseeding; mask real end frames rather than padding;
apply corruption only to valid clip prefixes; handle empty velocity selections;
compute consistency only after activation. A synchronous single-process/torchrun
entrypoint implements original AdamW, clipping, delayed EMA and full checkpoint
resumption with per-rank RNG and sampler cursors. See the training documentation
for the precise supported data format and scope. Legacy non-RoPE model classes
remain available, but the recovered active training path constructs MoreDiff.

## Native architecture migration

The active network definitions now live in `motius/models/mogendit/network`.
The bundle holds networks and checkpoint I/O only. Diffusion is implemented in
`motius/diffusion/mogendit`; the pipeline owns sampling and repair. Occam motion
encoding and skeletal math live in `motius/motion/representation/mogendit`.
The trainer owns corruption, mask scheduling and all loss computations; the
dataset is registered under `motius/datasets/mogendit.py`. The unrelated Aplus
training/export base, HumanML263 representation and commented prototype are
removed. Original source hashes remain in `source_manifest.json`.
