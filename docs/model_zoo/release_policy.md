# Model Zoo Release Policy

A Model Zoo entry is release-complete only when all of the following artifacts
are present and verified:

| Area | Required Artifact |
| ---- | ----------------- |
| Checkpoint | Public checkpoint link for every advertised variant |
| Demo | At least three verified qualitative demos rendered from released outputs |
| HumanML3D Official | T2M leaderboard metrics with the selected-caption HumanML3D protocol |
| MotionStreamer Evaluator | Metrics after the checked MotionStreamer conversion path |
| Motius Joint-Position Evaluator | Metrics with the unified SMPL-22 joint-position evaluator |
| Representation | The model's native motion representation, with conversion helpers clearly marked as adapters |

## Task Taxonomy

README and model-card task fields must use only these capability labels:

| Label | Meaning |
| ----- | ------- |
| [`T2M`](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard) | Text-to-motion generation |
| `M2T` | Motion-to-text generation |
| `TP2M` | Text-guided generation conditioned on a motion prefix |
| `Multi-Prompt T2M` | A sequence composed from multiple text prompts |
| `Motion Control` | Motion generation under general control signals |
| `Kinematic Control` | Motion generation under keyframe, trajectory, or end-effector constraints |

Zero-shot, streaming, latent, diffusion, and autoregressive describe how a
method is trained or executed; they are not separate tasks. Keep those terms in
the model summary and method description instead of the task field.

In the root Model Zoo table, every task with a published Motius leaderboard
must link to that page. Model-card task rows keep the canonical plain-text
labels so the cards remain portable to Hugging Face.

Model cards must not use adapter outputs as the model's native representation.
For example, HY-Motion T2M is `HY-Motion-201`; DART is `DART276`. SMPL,
SMPL-H, MotionStreamer, or HumanML3D conversions can be documented only as
rendering/evaluation adapters.

Demo tables must cover different selected-caption HumanML3D test cases and show
the input text next to each preview. Use compact 512px / 30fps GIFs for inline
model-card previews; a single oversized image is not release-complete.

The shared release facts live in
[`release_manifest.json`](release_manifest.json). Update that manifest whenever
a checkpoint, demo, metric row, or native representation changes, then sync the
README and the corresponding model card from the same facts.

Generated audit reports should be written under `outputs/`, for example:

```bash
python tools/audit_model_zoo_release.py --check-hf \
  --output outputs/model_zoo_release_audit.md
```
