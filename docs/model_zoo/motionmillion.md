<h1 align="center">MotionMillion / Go to Zero Model Card</h1>

<p align="center">
  <strong>Million-scale zero-shot text-to-motion generation, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.07095">Paper</a> ·
  <a href="https://vankouf.github.io/MotionMillion/">Project Page</a> ·
  <a href="https://github.com/VankouF/MotionMillion-Codes">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionMillion-7B-HumanML272">7B Checkpoint</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionMillion-3B-HumanML272">3B Checkpoint</a>
</p>

MotionMillion, also released as *Go to Zero: Towards Zero-shot Motion
Generation with Million-scale Data* (Fan et al., ICCV 2025), is a large
autoregressive text-to-motion model trained for zero-shot generalization. This
Motius release packages the FSQ HumanVQVAE tokenizer, LLaMA-style autoregressive
motion generator, Flan-T5-XL text encoder artifact, and MotionStreamer-272
normalization into a consistent inference pipeline.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/61f05618-b491-4287-9a36-63de5b61d6f8" controls></video> | [MP4](https://github.com/user-attachments/assets/61f05618-b491-4287-9a36-63de5b61d6f8) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=gotozero7b) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/61f05618-b491-4287-9a36-63de5b61d6f8" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/08387c9c-df62-4dd9-ae6c-66e491207255" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/60f77256-dbc6-4473-90f7-bd2ea92ab517" controls></video> |

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
| Training motion | 30 fps (MotionStreamer-272 / humanml3d_272) |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | Go to Zero / MotionMillion |
| Tasks | Text-to-Motion |
| Venue | ICCV 2025 |
| Motion representation | MotionStreamer-272 / humanml3d_272, 30 fps |
| Text encoder | Flan-T5-XL, frozen |
| Tokenizer | HumanVQVAE + FSQ, levels `[8,8,8,5,5,5]` |
| AR model | LLaMA-style 3B / 7B transformer |
| Checkpoints | [`7B train-only`](https://huggingface.co/ZeyuLing/Motius-MotionMillion-7B-HumanML272), [`3B train-only`](https://huggingface.co/ZeyuLing/Motius-MotionMillion-3B-HumanML272) |
| Pipeline | `motius.pipelines.motionmillion.MotionMillionPipeline` |

The checkpoint artifacts contain `fsq.safetensors`, `ar.safetensors`,
`mm_config.json`, `model_index.json`, `mean.npy`, `std.npy`, and a packaged
`text_encoder/` directory.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionmillion.MotionMillionPipeline` |
| Bundle | `motius.models.motionmillion.MotionMillionBundle` |
| Runtime | `motius.models.motionmillion.network` |

The runtime includes only the inference path: FSQ tokenizer, HumanVQVAE decoder,
and LLaMA-style autoregressive generator.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionMillion-7B-HumanML272",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person swings a golf club"],
    max_sample_steps=150,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 272)` and is
denormalized to the MotionStreamer-272 physical scale.

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
| HumanML3D Official | 7B train-only | 3,970 | 0.523 | 0.721 | 0.817 | 0.065 | 2.897 | 9.394 | Measured |
| MotionStreamer Evaluator | 7B train-only | 4,042 | 0.7403 | 0.8777 | 0.9236 | 3.0807 | 15.3706 | 27.5748 | Measured |
| Motius Joint-Position Evaluator | 7B train-only | 4,034 | 0.6277 | 0.7904 | 0.8579 | 0.0183 | 29.9675 | 53.4788 | Measured |
| HumanML3D Official | 3B train-only | 3,970 | 0.528 | 0.723 | 0.818 | 0.071 | 2.882 | 9.379 | Measured |
| MotionStreamer Evaluator | 3B train-only | 4,042 | 0.7401 | 0.8772 | 0.9229 | 3.0658 | 15.3806 | 27.5604 | Measured |
| Motius Joint-Position Evaluator | 3B train-only | 4,034 | 0.6228 | 0.7860 | 0.8569 | 0.0185 | 29.9661 | 54.6241 | Measured |

## Motion Representation

MotionMillion generates `humanml3d_272`, the same 272-dim, 30 fps layout used by
MotionStreamer:

```text
text -> Flan-T5-XL -> LLaMA AR -> FSQ dequantize
     -> HumanVQVAE decoder -> MotionStreamer-272 motion
```

Because this representation matches MotionStreamer-272 directly, native
evaluation does not need an additional rotation re-encoding step.

## Citation and License

```bibtex
@inproceedings{fan2025gotozero,
  title={Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data},
  author={Fan, Ke and Lu, Shunlin and Dai, Minyue and Yu, Runyi and Xiao, Lixing and Dou, Zhiyang and Dong, Junting and Ma, Lizhuang and Wang, Jingbo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
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
