# Model Zoo Release Policy

A Model Zoo entry is release-complete only when all of the following artifacts
are present and verified:

| Area | Required Artifact |
| ---- | ----------------- |
| Checkpoint | Public checkpoint link for every advertised variant |
| Demo | One task-specific GitHub-native H.264 video player for every public `infer_{task}` API; motion outputs use an inline SMPL/SMPL-H/native-body Mesh preview |
| HumanML3D Official | T2M leaderboard metrics with the selected-caption HumanML3D protocol |
| MotionStreamer Evaluator | Metrics after the checked MotionStreamer conversion path |
| Motius Joint-Position Evaluator | Metrics with the unified SMPL-22 joint-position evaluator; FID is measured only in per-sample L2-normalized embedding space |
| Representation | The model's native motion representation, with conversion helpers clearly marked as adapters |
| Frame-rate contract | Native training FPS for every checkpoint branch, plus the public preview FPS and any duration-preserving resampling |

## Task Taxonomy

README and model-card task fields must use labels from the machine-readable
[task taxonomy](../tasks/taxonomy.json), documented in the
[Task Registry](../tasks/README.md). The release audit reads this file directly;
do not duplicate a second task list in model documentation or source code.

Prediction, in-betweening, keyframes, and TP2M are `Temporal Motion Completion`
tracks. `Motion Control`, `Joint Control`, `Two-Person T2M`, and generic
`multimodal motion tasks` are not valid task-field labels.

Zero-shot, streaming, latent, diffusion, and autoregressive describe how a
method is trained or executed; they are not separate tasks. Keep those terms in
the model summary and method description instead of the task field.

In the Model Zoo index, every task with a published Motius leaderboard must
link to that page. Model-card task rows keep canonical plain-text labels so the
cards remain portable to Hugging Face.

Model cards must not use adapter outputs as the model's native representation.
For example, HY-Motion T2M is `HY-Motion-201`; DART is `DART276`. SMPL,
SMPL-H, MotionStreamer, or HumanML3D conversions can be documented only as
rendering/evaluation adapters.

The generated Task Demos table must match the public task catalog exactly.
One task's media cannot satisfy another task's release gate. Each primary
preview must be an H.264 MP4 uploaded through GitHub's attachment service and
embedded as a native `<video>` player. Repository-relative MP4 files and
GIF/WebP images do not satisfy this gate because GitHub does not render them as
video players. A Three.js all-case viewer is supplementary and can never
replace the inline video;
when present, it must filter to the selected method so a Model Card does not
request every method asset and trigger rate limits. Inline HumanML3D previews
must cover different selected-caption cases, show only the input text and
result, and use compact 512px / 30fps H.264 videos. Do not expose dataset case
IDs in preview tables. Attachment URLs are recorded in
`video_attachments.json`; `tools/publish_model_card_videos.py` publishes or
refreshes them before the generated Model Card content is synchronized.
The 30fps preview clock is not a claim about the checkpoint's training rate.
Every card must expose both clocks in its generated Frame-Rate Contract. A
20fps checkpoint may be shown at 30fps only after duration-preserving temporal
resampling; relabeling the original frames as 30fps is forbidden.

Canonical metric blocks are synchronized from the machine-readable
Leaderboard snapshots. For the joint-position evaluator, `Motius FID` and
`uTMR FID` always refer to per-sample L2-normalized embedding-space FID.
Historical raw-space FID values remain provenance only and must never be
substituted when the normalized value has not been recomputed.

The shared release facts live in
[`release_manifest.json`](release_manifest.json). Update that manifest whenever
a checkpoint, demo, metric row, or native representation changes, then sync the
README and the corresponding model card from the same facts.

Generated audit reports should be written under `outputs/`, for example:

```bash
python tools/export_t2m_leaderboard_results.py --check
python tools/publish_model_card_videos.py
python tools/sync_model_card_content.py
python tools/audit_model_card_content.py
python tools/audit_model_zoo_release.py --check-hf \
  --output outputs/model_zoo_release_audit.md
```

## Monocular Motion Capture Releases

Monocular capture packages use a task-specific release gate because licensed
video datasets and parametric body assets cannot be redistributed:

- record the pinned official source revision and every evaluated checkpoint
  SHA-256;
- complete at least one real-video inference path and emit a pickle-free
  `MonocularCaptureResult`;
- declare the native body model and camera/world coordinate availability
  without fabricating cross-topology vertices or camera trajectories;
- reference, but never redistribute, separately licensed detectors, body
  models, and benchmark media;
- disclose tracking coverage, input protocol, body-model compatibility, and
  numerical limitations.

A 3DPW or EMDB row is publishable only after the complete licensed split is
evaluated under its registered protocol. Partial runs and paper-reported values
remain diagnostic.
