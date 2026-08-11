<h1 align="center">MotionGPT3 Model Card</h1>

<p align="center">
  <strong>A continuous-latent bimodal motion-language model, packaged for HumanML3D motion captioning.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.24086">Paper</a> ·
  <a href="https://motiongpt3.github.io/">Project Page</a> ·
  <a href="https://github.com/OpenMotionLab/MotionGPT3">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D">Motius Checkpoint</a>
</p>

MotionGPT3 separates text and motion processing into modality-specific branches
with shared attention. Unlike tokenized motion-language models, it represents
motion in a continuous VAE latent space. The Motius artifact packages the final
official multi-task checkpoint and all model/tokenizer configuration required by
`Pipeline.from_pretrained`.

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
| Text-to-Motion | Native task input | <video src="https://github.com/user-attachments/assets/9d48815f-6495-4438-aea3-43f9f753321f" controls></video> | [MP4](https://github.com/user-attachments/assets/9d48815f-6495-4438-aea3-43f9f753321f) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motiongpt3) |
| Motion-to-Text | SMPL motion input | <video src="https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333" controls></video> | [MP4](https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333) · [All cases](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motiongpt3) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->

| Input | MotionGPT3 SMPL Preview |
| --- | --- |
| person walking at a average pace forward, swaying arms and torso with a sense of swagger | <video src="https://github.com/user-attachments/assets/9d48815f-6495-4438-aea3-43f9f753321f" controls></video> |

Use the unified
[HumanML3D all-case comparison](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motiongpt3)
for Text-to-Motion and the
[M2T case explorer](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motiongpt3)
for Motion-to-Text. Both views expose the input caption, prediction, selected
reference, and synchronized SMPL Mesh motion for every published case.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Motion-to-Text | `infer_motion_to_text` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 20 fps (HumanML3D) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Tasks | Text-to-Motion, Motion-to-Text |
| Input representation | HumanML3D-263, 20 fps |
| Motion latent | Continuous temporal VAE latent |
| Language model | GPT-2-family bimodal Transformer |
| Checkpoint provenance | Official final MotionGPT3 checkpoint |
| Checkpoint | [`ZeyuLing/Motius-MotionGPT3-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D) |
| Pipeline | `motius.pipelines.motiongpt3.MotionGPT3Pipeline` |

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius/pipelines/motiongpt3/pipeline.py` |
| Bundle | `motius/models/motiongpt3/bundle.py` |
| Bimodal GPT runtime | `motius/models/motiongpt3/mot_example_gpt2_sepattn.py` |
| Generation runtime | `motius/models/motiongpt3/mot_example_gpt2_sepattn_gen.py` |

## Quick Start

```python
import numpy as np
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionGPT3-HumanML3D",
    bundle_kwargs={"device": "cuda"},
)
motion = np.load("sample.npy")  # denormalized HumanML3D-263
caption = pipe.infer_motion_to_text([motion], lengths=[len(motion)])[0]

generated = pipe.infer_text_to_motion(
    ["a person walks forward and waves"],
    [120],
)[0]
```

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Motion-to-Text | [Published results](../leaderboards/hf_space_m2t_humanml3d/m2t_results.json) | HumanML3D M2T metrics |

### Canonical Motion-to-Text Snapshot

| Method | n | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MotionGPT3 | 4,400 | 0.4635 | 0.0651 | 0.3672 | 0.1063 | 0.8820 | 0.3007 | 0.5333 | 0.7178 | 0.8043 | 2.8533 |

### Canonical HumanML3D Semantic Results

| Evaluator | Variant | n | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HumanML3D Official | official | — | — | — | — | — | — | — | Not measured |
| MotionStreamer Evaluator | official | 4,042 | 0.6709 | 0.8242 | 0.8817 | 20.9913 | 17.5664 | 25.6889 | Measured |
| Motius Joint-Position Evaluator | official | 4,034 | 0.5997 | 0.7656 | 0.8425 | 0.0936 | 32.5604 | 56.1671 | Measured |

<!-- MOTIUS_CANONICAL_METRICS:END -->


| Protocol | Samples | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| -------- | ------: | -----: | ------: | ----: | -------: | --------------: | --: | --: | --: | -------: |
| [HumanML3D M2T](../tasks/m2t.md) | 4,400 | 0.0651 | 0.3672 | 0.1063 | 0.8820 | 0.3007 | 0.5333 | 0.7178 | 0.8043 | 2.8533 |

The [M2T case explorer](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard#case-explorer)
contains MotionGPT3's prediction for every one of the 4,400 evaluated clips.

### M2T Demo Cases

| Human reference | MotionGPT3 prediction | Motion |
| --------------- | --------------------- | ------ |
| a man kicks something or someone with his left leg. | a person kicks with left leg. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| person jogs around to the left and right | a person runs to the right, then back to where they started. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| a person jumping while raising both hands and moving apart legs. | a person performs jumping jacks. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

## Motion Representation

MotionGPT3 consumes and returns denormalized HumanML3D-263 motion at 20 FPS.
Its continuous temporal VAE maps that motion to a non-discrete latent sequence;
the bimodal Transformer processes motion latents and text tokens in separate
branches with shared attention. SMPL previews are produced only after the
standard HumanML3D-to-SMPL bridge and are not the model's native output.

## Citation and License

```bibtex
@misc{zhu2025motiongpt3,
  title={MotionGPT3: Human Motion as a Second Modality},
  author={Zhu, Bingfan and Jiang, Biao and Wang, Sunyi and Tang, Shixiang and Chen, Tao and Luo, Linjie and Zheng, Youyi and Chen, Xin},
  year={2025},
  eprint={2506.24086},
  archivePrefix={arXiv}
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
