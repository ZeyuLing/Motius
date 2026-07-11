<h1 align="center">MotionCLIP Evaluator Card</h1>

<p align="center">
  <strong>SMPL-22 text-motion contrastive evaluator packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2203.08063">Paper</a> |
  <a href="https://guytevet.github.io/motionclip-page/">Project Page</a> |
  <a href="https://github.com/GuyTevet/MotionCLIP">Original GitHub</a>
</p>

MotionCLIP-135 exposes a compact evaluator API for text-motion embedding,
pairwise cosine scoring, and retrieval. This Motius package keeps the evaluator
separate from generation methods so leaderboard code can call it explicitly
without treating it as a text-to-motion model.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | MotionCLIP-135 |
| Task | Text-motion contrastive scoring and retrieval |
| Motion representation | SMPL-22, 135-dim per frame: global translation + 22 joints in 6D rotation |
| Text encoder | CLIP ViT-B/32 text tower, 256-token context in this checkpoint family |
| Motion encoder | Transformer motion tower, 512-dim projection |
| Checkpoint | Pending HF artifact; current Motius artifact format is `bundle_config.json` + `motionclip_model.safetensors` |
| Pipeline | `motius.pipelines.motion_clip.MotionCLIPPipeline` |

## Usage

Install Motius and the standard runtime dependencies:

```bash
python -m pip install -e ".[dev]"
```

Load a local exported MotionCLIP artifact:

```python
import torch
from motius.pipelines.motion_clip import MotionCLIPPipeline

pipe = MotionCLIPPipeline.from_pretrained(
    "checkpoints/motion_clip/motionclip_base_1p_aug_hq",
    device="cuda",
)

motion = torch.randn(2, 120, 135, device="cuda")
scores = pipe.score(
    ["a person walks forward", "a person jumps twice"],
    motion,
    num_frames=[120, 120],
)
```

The pipeline also exposes:

| API | Output |
| --- | ------ |
| `encode_text(texts)` | L2-normalized text embeddings |
| `encode_motion(motion, num_frames)` | L2-normalized motion embeddings |
| `retrieve_motion_from_text(query_texts, motion_db, top_k)` | similarity matrix and top-k motion indices |
| `retrieve_text_from_motion(query_motions, text_db, top_k)` | similarity matrix and top-k captions |

## Evaluation Protocol

This evaluator expects motions already converted to the shared SMPL-22 135-dim
format used by the internal MotionCLIP evaluation scripts. If callers pass raw
HumanML3D-263, MotionStreamer-272, or SMPL-H motion vectors directly, scores are
not comparable.

The evaluator can normalize motion internally when the bundle is created with a
registered SMPL pose processor. Public inference artifacts can also be used with
`already_normalized=True` when the caller has applied the matching statistics.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motion_clip.MotionCLIPPipeline` |
| Bundle | `motius.models.motion_clip.MotionCLIPBundle` |
| Model | `motius.models.motion_clip.MotionCLIPModel` |
| Text tower | `motius.models.motion_clip.MotionCLIPTextModelWithProjection` |
| Motion tower | `motius.models.motion_clip.MotionCLIPMotionModelWithProjection` |

## Citation

```bibtex
@inproceedings{tevet2022motionclip,
  title={MotionCLIP: Exposing Human Motion Generation to CLIP Space},
  author={Tevet, Guy and Gordon, Brian and Hertz, Amir and Bermano, Amit H. and Cohen-Or, Daniel},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
