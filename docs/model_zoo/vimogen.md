<h1 align="center">ViMoGen Model Card</h1>

<p align="center">
  <strong>Generalizable motion generation with visual generative priors, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.26794">Paper</a> ·
  <a href="https://motrixlab.github.io/2026_iclr_vimogen/">Project Page</a> ·
  <a href="https://github.com/MotrixLab/ViMoGen">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-ViMoGen-1.3B-HumanML3D">Motius Checkpoint</a>
</p>

ViMoGen is the motion model from *The Quest for Generalizable Motion
Generation: Data, Model, and Evaluation*. This Motius release packages the
released 1.3B HumanML3D checkpoint behind the same bundle/pipeline API used by
the rest of the Model Zoo.

<!-- MOTIUS_MODEL_CARD_NAV:START -->
<p align="center">
  <a href="#visual-results">Visual Results</a> ·
  <a href="#model-overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#motion-representation">Motion Representation</a>
</p>
<!-- MOTIUS_MODEL_CARD_NAV:END -->

## Visual Results

<!-- MOTIUS_TASK_DEMOS:START -->

### Task Demos

| Task | Input / condition | Rendered output | More |
| --- | --- | --- | --- |
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/ca6ff21c-dc97-437a-8edf-593efcb0afb9" controls></video> | [MP4](https://github.com/user-attachments/assets/ca6ff21c-dc97-437a-8edf-593efcb0afb9) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=vimogen) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/ca6ff21c-dc97-437a-8edf-593efcb0afb9" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/caa20e26-40ea-4a01-8cef-45886c21fda2" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/37059650-6449-4c65-b47c-3242fbe68c36" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 20 fps (DART276 / HumanML3D) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | ViMoGen 1.3B |
| Tasks | Text-to-Motion |
| Venue | ICLR 2026 |
| Motion representation | DART276, 20 fps |
| Text encoder | Wan2.1 T2V-1.3B UMT5-XXL encoder |
| Backbone | WanVideoTM2M 1.3B flow-matching DiT |
| Default sampler | Flow matching, 50 inference steps |
| Checkpoint | [`ZeyuLing/Motius-ViMoGen-1.3B-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-ViMoGen-1.3B-HumanML3D) |
| Pipeline | `motius.pipelines.vimogen.ViMoGenPipeline` |

The checkpoint artifact contains `model.pt`, `model_index.json`, and
`assets/meta/{mean,std}.npy`. The Wan2.1 base assets are resolved from the
public `Wan-AI/Wan2.1-T2V-1.3B` Hub repo declared by `wan_repo_id`.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.vimogen.ViMoGenPipeline` |
| Bundle | `motius.models.vimogen.ViMoGenBundle` |
| Runtime | `motius.models.vimogen.network` |
| Scheduler | `motius.models.vimogen.network.vimogen.trainer.scheduler` |

The runtime vendors the required ViMoGen transformer modules and scheduler, so
inference does not import the upstream checkout.

## Quick Start

Install the ViMoGen extra dependencies:

```bash
python -m pip install -e ".[dev,vimogen]"
```

Run text-to-motion inference:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-ViMoGen-1.3B-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["Full-body shot, stable camera. A person walks forward at an average pace."],
    [200],
    seed=0,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 276)` and is
denormalized to ViMoGen's DART276 physical scale. For leaderboard-style
generation, use the prompt rewrite workflow used by the internal evaluator
scripts, then score the result with the shared evaluator protocol.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | 1.3B prompt-rewrite | 3,970 | 0.283 | 0.438 | 0.547 | 8.371 | 4.894 | 6.709 | Measured |
| MotionStreamer Evaluator | 1.3B prompt-rewrite | 4,042 | 0.4291 | 0.5687 | 0.6518 | 152.2095 | 21.0737 | 24.1803 | Measured |
| Motius Joint-Position Evaluator | 1.3B prompt-rewrite | 4,034 | 0.3041 | 0.4330 | 0.5198 | 0.4911 | 47.0574 | 55.6162 | Measured |

## Motion Representation

ViMoGen emits DART276, the global DART-style representation:

```text
text -> UMT5-XXL embeddings -> WanVideoTM2M DiT -> denormalized DART276
```

The released pipeline returns DART276 directly. Converting DART276 to SMPL mesh
or cross-representation evaluator inputs should be done through a checked
conversion path before reporting metrics.

## Citation and License

```bibtex
@article{lin2025quest,
  title={The Quest for Generalizable Motion Generation: Data, Model, and Evaluation},
  author={Lin, Jing and Wang, Ruisi and Lu, Junzhe and Huang, Ziqi and Song, Guorui and Zeng, Ailing and Liu, Xian and Wei, Chen and Yin, Wanqi and Sun, Qingping and Cai, Zhongang and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2510.26794},
  year={2025}
}
```

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
