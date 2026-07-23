# Sequential Text-to-Motion · BABEL Evaluation

The Sequential Text-to-Motion · BABEL benchmark evaluates one continuous motion
conditioned on an ordered list of action captions. The protocol uses every
eligible episode in the processed official BABEL validation split. It contains
1,295 episodes, 7,285 captioned action intervals, and 5,990 transitions.

## Protocol

Explicit BABEL transition labels are removed and neighboring actions are cut at
the transition midpoint. Adjacent short actions are greedily merged until each
conditioned interval contains at least 30 frames. Every resulting caption is
rewritten by the supplied precomputed LLM rewrite cache. Raw `proc_label` text
is not used as the evaluation caption.

All reference and generated motions are stored as 30 fps, neutral zero-beta
SMPL-22 joints66. The complete episode is canonicalized before it is written:
first-pelvis XZ origin, first-frame body facing +Z, and floor height zero. The
evaluator independently reapplies the same transform to every captioned
interval and transition window as a defensive boundary check.

| Group | Metrics | Reference |
| ----- | ------- | --------- |
| Semantic subsequences | R@1/2/3, MM-Dist, FID, Diversity | The 7,285 paired GT intervals from the same 1,295 episodes |
| 30-frame transitions | Transition FID, Diversity, Peak Jerk, AUJ gap | The 5,990 paired GT transition windows at the same boundaries |

Retrieval and embedding metrics use
[`ZeyuLing/motius-evaluator-universal-smplh-joints66`](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66).
Every uTMR FID first L2-normalizes each motion embedding, then estimates the
mean and covariance in that normalized space. Raw latent-space FID is not a
Motius reporting metric because encoder feature magnitude can dominate it.
Each captioned action interval is sliced first and independently canonicalized
from that subclip's first frame before it enters uTMR. This removes the global
position and heading inherited from the preceding action. Transition metrics
use one 30-frame window spanning both sides of a boundary and canonicalize that
window only once; the relative position, heading, velocity, and acceleration
gap across the boundary therefore remains measurable.
Diversity and absolute Peak Jerk are diagnostic statistics, not ranked quality
objectives. GT/reference rows do not participate in best/second-best styling.

BABEL captions are not unique, and distinct raw labels may denote the same
action. Exact-caption grouping incorrectly treats examples such as `walk`,
`walking`, and `walking forward` as negatives. R-Precision therefore uses the
official BABEL [`act_cat` taxonomy](https://babel.is.tue.mpg.de/data.html): an
interval's positive group is the ordered sequence of its source actions, with
the sorted `act_cat` set retained for each action. This merges synonymous
labels while preserving annotated modifiers; for example, `walk back` remains
distinct because it also has `backwards movement`. `proc_label` is used only
when an official category is unavailable.

The 7,285 intervals form 1,738 action groups; 6,186 intervals belong to one of
639 repeated groups. Every candidate with the same action signature is a valid
positive within its 32-sample recall batch. No interval is removed. MM-Dist
uses the nearest positive, while FID and Diversity continue to use all 7,285
intervals.

## Measured Results

| Method | R@1 | R@2 | R@3 | Normalized FID | MM-Dist | Diversity | Normalized Transition FID | Transition Diversity | Peak Jerk | AUJ Gap |
| ------ | --: | --: | --: | --: | ------: | --------: | -------------: | -------------------: | --------: | ------: |
| BABEL GT | 0.3614 | 0.5284 | 0.6317 | 0.0000 | 47.8378 | 49.2886 | 0.0000 | 43.7889 | 56.34 | 0.0000 |
| FlowMDM | 0.2504 | 0.3925 | 0.4818 | 0.0467 | 50.8503 | 50.3880 | 0.0555 | 44.9407 | 335.67 | 34.4040 |
| MotionStreamer | 0.2130 | 0.3303 | 0.4175 | 0.0610 | 52.0339 | 46.5873 | 0.0702 | 40.8189 | 206.22 | 76.2889 |
| MotionLab | 0.2580 | 0.3793 | 0.4536 | 0.2011 | 51.3873 | 41.6184 | 0.2499 | 34.3793 | 204.67 | 25.7259 |
| PRISM (epoch 14) | 0.2453 | 0.3716 | 0.4555 | 0.0574 | 51.8148 | 50.2988 | 0.0732 | 47.2516 | 392.99 | 106.4423 |

This is a single deterministic seed-42 generation and one retrieval repeat.
R-Precision uses 32-sample recall batches, covering 7,264 of the 7,285 paired
segments, and accepts every same-action candidate as a positive. Distribution
metrics use the full set. `--chunk-size 32` controls the recall candidate set;
`--batch-size` controls only evaluator encoding throughput and does not alter
the metric. PRISM uses `checkpoint-epoch_14` and fixes every internal model call to a
360-frame canvas; all 1,295 outputs passed exact-length and fixed-canvas
validation. MotionLab uses its native five-frame autoregressive context and
emits HML263. Its SMPL-22 evaluation input is recovered from the HML263 position
channels and fitted to neutral SMPL; HML263 rotations are not passed directly
into SMPL FK. Both rows use all 1,295 episodes.

Open the [Three.js sequence audit](../leaderboards/hf_space_babel_sequential/audit/index.html)
to compare BABEL GT, FlowMDM, MotionStreamer, PRISM, and MotionLab frame by frame. Every
subclip has a fixed color, and the synchronized caption list exposes its exact
half-open frame interval.
Each row also reports the nearest three texts for the GT and generated motion,
plus the exact text-to-motion positive rank in the seed-0, 32-candidate batch
used by the leaderboard.

## Data Layout

Download BABEL motions and annotations under its research license, then place
the files as follows:

```text
data/babel/processed/
├── manifests/val.jsonl
├── ms272/val/{episode_id}.npz
└── babel_shortmerge_caption_rewrites.json

data/babel/babel-teach/val.json
checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
```

The JSONL must be produced from the official BABEL validation annotations. The
MS272 files may contain either the full source motion or the already clipped
episode span. The rewrite cache is keyed by the merged source-label sequence.
Official `val.json` supplies `act_cat` for action-level positive groups.

## Reproduce FlowMDM

```bash
python tools/build_babel_sequential_manifest.py \
  --processed-manifest data/babel/processed/manifests/val.jsonl \
  --motion272-dir data/babel/processed/ms272/val \
  --rewrite-cache data/babel/processed/babel_shortmerge_caption_rewrites.json \
  --babel-annotations data/babel/babel-teach/val.json \
  --smpl-model checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --output-root outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1

python tools/generate_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest_actiongroups_v3.json \
  --model ZeyuLing/motius-flowmdm-babel \
  --output-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42 \
  --device cuda --seed 42

python tools/eval_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest_actiongroups_v3.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/joints66 \
  --method FlowMDM \
  --output outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/metrics_actiongroups_v3_normalized_fid.json \
  --device cuda --batch-size 32 --chunk-size 32 --n-repeats 1

python tools/export_babel_retrieval_audit.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest_actiongroups_v3.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/joints66 \
  --output outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/retrieval_audit.json \
  --device cuda --batch-size 128 --chunk-size 32 --top-k 3 --seed 0

python tools/build_babel_sequential_viewer.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest_actiongroups_v3.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/joints66 \
  --prediction MotionStreamer=outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/motionstreamer_latest_seed42/joints66 \
  --prediction MotionLab=outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/motionlab_f5_actiongroups_v4_smplfit/joints66 \
  --smpl-parameters MotionLab=outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/motionlab_f5_actiongroups_v4_smplfit/smpl \
  --prediction 'PRISM (epoch 14)=outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/prism_epoch14_actiongroups_v4/joints66' \
  --smpl-parameters 'PRISM (epoch 14)=outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/prism_epoch14_actiongroups_v4/smplx' \
  --retrieval-audit outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/retrieval_audit.json \
  --output-dir outputs/visualization/babel_sequential_audit
```

The audit viewer renders native or fitted SMPL parameters when a method exposes
them; otherwise it fits the exact canonical SMPL-22 joints used by evaluation
to a neutral SMPL body. This avoids inventing unobservable head and terminal
joint twists from positions alone. Each method retains its own global XZ
trajectory; the floor, trajectory trace, and per-frame body-facing arrow make
canonicalization and displacement errors visible.
The cyan `Body facing` arrow is estimated from the current hips and shoulders; it is
not the root-velocity direction. It aligns with forward locomotion, but can
legitimately oppose backward motion or differ during sideways motion.

Both generation and evaluation accept deterministic sharding for cluster runs.
Generated artifacts and metrics must remain under `outputs/`.

MotionStreamer uses a separate exact-length runner because its latent tokens
span four frames. Long actions are generated continuously across bounded latent
blocks; only the zero-to-three-frame token-alignment remainder is linearly
resampled within each segment, preserving the manifest's original boundaries.

```bash
python tools/generate_babel_sequential_motionstreamer.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest_actiongroups_v3.json \
  --model ZeyuLing/hftrainer-motionstreamer-humanml272 \
  --output-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/motionstreamer_seed42 \
  --device cuda --seed 42
```

MotionLab emits HumanML3D-263 features. Materialize them through the dedicated
position-driven SMPL fitting route:

```bash
python tools/materialize_hml263_smpl_joints.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest_actiongroups_v3.json \
  --hml263-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/motionlab_f5_actiongroups_v3/motion272/_hml263 \
  --output-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/motionlab_f5_actiongroups_v4_smplfit \
  --smpl-model-dir checkpoints/body_models/smpl \
  --device cuda --refine-iters 80 --rotation-init hml263_end_effectors \
  --max-fit-mpjpe-mm 50
```

This conversion resamples 20-fps HML263 to the manifest's 30-fps frame count,
fits neutral SMPL against recovered joint positions, writes the fitted SMPL
parameters alongside joints66, canonicalizes the complete episode once, and
records per-case fitting error. The measured mean fit MPJPE over all 1,295
episodes is 16.4 mm. A case whose mean fitting error exceeds 50 mm is rejected
before any SMPL parameters or joints are published.

## Submission Contract

For another sequential method, write one `joints66/{case_id}.npy` file per
manifest case. Each array must have shape `(T, 66)`, 30 fps, and cover every
half-open segment interval listed in the manifest. The stored episode must
already use first-pelvis XZ origin, first-frame `+Z` facing, and floor height
zero. Evaluation never imports the method's original repository at runtime.
