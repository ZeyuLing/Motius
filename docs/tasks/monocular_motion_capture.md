# Monocular Motion Capture

Motius treats monocular capture as video-to-motion estimation with explicit
camera/world coordinate systems, person tracks, valid-frame masks, body-model
identity, and immutable runtime provenance.

## Licensed test data

Download benchmark data only from the official sites after accepting each
license:

- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
- [EMDB](https://eth-ait.github.io/emdb/)
- [SMPL](https://smpl.is.tue.mpg.de/)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)

RICH is not part of the first public snapshot. It will be added after a
licensed local copy is available.

Keep these datasets outside Git. Build relative-path-only manifests under
`outputs/`:

```bash
tools/stage_3dpw_test_data.sh /local/scratch/3DPW

python tools/build_monocular_capture_manifests.py \
  --3dpw-root /local/scratch/3DPW \
  --output-dir outputs/evaluation/monocular_capture/manifests
```

The staging helper reads the private local
`data/3DPW/{sequenceFiles.zip,test_imageFiles.tar}` snapshot and verifies all
24 test sequences and 26,240 frames. These licensed archives remain untracked
and must never be published.

The generated JSON contains no images, videos, annotations, or absolute private
paths.

Materialize protocol-locked ground truth with user-licensed SMPL files:

```bash
python tools/materialize_monocular_capture_ground_truth.py \
  --manifest outputs/evaluation/monocular_capture/manifests/3dpw_test.json \
  --data-root /private/path/3DPW \
  --smpl-model male=/private/models/SMPL_MALE.pkl \
  --smpl-model female=/private/models/SMPL_FEMALE.pkl \
  --smpl-model-version 1.0.0 \
  --output-dir outputs/evaluation/monocular_capture/ground_truth/3dpw
```

The materializer writes pickle-free canonical NPZ artifacts and records the
annotation and body-model SHA-256 values. It does not copy licensed inputs.

## Protocols

### 3DPW Test camera coordinates

Protocol key: `3dpw_test_camera_v1`.

- Population: every person track in `sequenceFiles/test`.
- Validity: official `campose_valid` intersected with non-empty `poses2d`, as
  implemented by the pinned official 3DPW evaluator.
- Metrics: PA-MPJPE, pelvis-aligned MPJPE, compatible-body-model PVE, and
  fps-aware acceleration error.
- Native 3DPW global translation is not an evaluation target. Its human and
  camera trajectories share substantial time-varying reconstruction drift;
  only their relative camera-space pose is sufficiently stable. Use EMDB-2 for
  global trajectory metrics.

### EMDB-1 camera coordinates

Protocol key: `emdb_1_camera_v1`.

- Population: sequences whose official annotation has `emdb1=True`.
- Validity: official `good_frames_mask`.
- Metrics: PA-MPJPE, pelvis-aligned MPJPE, compatible-body-model PVE, and
  fps-aware acceleration error.

### EMDB-2 world coordinates

Protocol key: `emdb_2_global_v1`.

- Population: sequences whose official annotation has `emdb2=True`.
- Validity: official `good_frames_mask`.
- Metrics follow the released GVHMR/WHAM implementation:
  - W-MPJPE: similarity alignment from the first two frames of each 100-frame
    chunk.
  - WA-MPJPE: similarity alignment over every complete 100-frame chunk.
  - RTE: rigidly aligned root error normalized by ground-truth path length.
  - Jitter: third finite difference at 30 fps using the GVHMR display scale.
  - Foot sliding: predicted displacement of the four official SMPL foot
    vertices during ground-truth contact.

## Cross-body-model policy

GEM-X predicts SOMA-77, while 3DPW and EMDB ground truth use SMPL. Motius keeps
the native SOMA representation and compares unlike body models only on
`common_hmr15_named_v1`. Joints are selected by audited names, never by matching
array dimensions. PVE and SMPL foot-vertex metrics are unavailable for native
SOMA submissions.

## Methods

- GVHMR: official `zju3dv/GVHMR` isolated runtime, SMPL camera and world output.
- PromptHMR-Video: official `yufu-wang/PromptHMR` isolated runtime. World
  metrics remain unavailable unless the upstream output explicitly contains a
  world trajectory.
- GEM-SMPL: official `NVlabs/GENMO` runtime (the paper/repository was originally
  named GENMO).
- GEM-X: official `NVlabs/GEM-X` runtime with native SOMA-77 outputs. The
  pinned demo uses an identity camera trajectory when no external visual
  odometry is supplied, so current verified results are camera-space only.
- HYMotion-V2M: first-class Motius runtime with native SMPL-H output,
  SAM-3D-Body image tokens, and per-target 3DPW crop inference.

Every result must record the source revision, checkpoint SHA-256, external
detector/tracker provenance, output fps, coordinate systems, and per-track
coverage. Missing frames are not silently interpolated for metric evaluation.

## Publishing policy

The public leaderboard is verified-only. A row requires:

1. complete manifest coverage;
2. immutable checkpoint and source revisions;
3. a successful Motius metric run;
4. upstream parity on at least one official sample or evaluation split;
5. no redistribution of restricted benchmark media or annotations.

Paper-only values and incomplete smoke tests may appear in model cards as audit
targets, but never as ranked rows.
