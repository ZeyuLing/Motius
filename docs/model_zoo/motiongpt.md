<h1 align="center">MotionGPT Model Card</h1>

<p align="center">
  <strong>Motion-language generation with discrete motion tokens, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2306.14795">Paper</a> ·
  <a href="https://motion-gpt.github.io/">Project Page</a> ·
  <a href="https://github.com/OpenMotionLab/MotionGPT">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionGPT-HumanML3D">Motius Checkpoint</a>
</p>

MotionGPT is the motion-language baseline from *MotionGPT: Human Motion as a
Foreign Language* (Jiang et al., NeurIPS 2023). This Motius release packages
the HumanML3D motion tokenizer, FLAN-T5-base-style language model with motion
tokens, HumanML3D statistics, and task-facing text-to-motion / motion-to-text
pipeline methods without requiring the original checkout.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/39a0e36c-f1b6-4821-9348-148aaef66a30" controls></video> | [MP4](https://github.com/user-attachments/assets/39a0e36c-f1b6-4821-9348-148aaef66a30) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motiongpt) |
| Motion-to-Text | SMPL motion input | <video src="https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333" controls></video> | [MP4](https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333) · [All cases](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motiongpt) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/39a0e36c-f1b6-4821-9348-148aaef66a30" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/e0620fee-a7b5-4ef6-872d-ec7000cf48bb" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/f9d3f1f2-5a06-494e-8e70-e6653b5c8a67" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

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
| Method | MotionGPT, language modeling over text and motion tokens |
| Tasks | Text-to-Motion, Motion-to-Text |
| Venue | NeurIPS 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Language backbone | FLAN-T5-base-style encoder-decoder with motion tokens |
| Motion tokenizer | VQ-VAE, 512-code codebook |
| Checkpoint | [`ZeyuLing/Motius-MotionGPT-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MotionGPT-HumanML3D) |
| Pipeline | `motius.pipelines.motiongpt.MotionGPTPipeline` |

The checkpoint artifact contains `motiongpt_s3_h3d.tar`,
`assets/meta/mean.npy`, `assets/meta/std.npy`, `deps/flan-t5-base/`, and
`model_index.json`.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motiongpt.MotionGPTPipeline` |
| Bundle | `motius.models.motiongpt.MotionGPTBundle` |
| Runtime | `motius.models.motiongpt.network.mGPT.archs` |

Only the inference-time MotionGPT modules required by the bundle are included
in this public package.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionGPT-HumanML3D",
    bundle_kwargs={"local_files_only": False},
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale. The same pipeline also exposes
`infer_m2t` for captioning denormalized HumanML3D-263 motions.

```python
caption = pipe.infer_motion_to_text(
    [motions[0]],
    lengths=[len(motions[0])],
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
| MotionGPT | 4,400 | 0.3971 | 0.0460 | 0.3365 | 0.0782 | 0.8851 | 0.3193 | 0.5087 | 0.6880 | 0.7776 | 3.1121 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,962 | 0.434 | 0.600 | 0.686 | 0.156 | 3.920 | 9.747 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.4940 | 0.6352 | 0.6944 | 23.6811 | 19.6781 | 25.5410 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4325 | 0.5801 | 0.6617 | 0.1024 | 38.4534 | 56.8849 | Measured |

### Motion-to-Text

| Protocol | Samples | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| -------- | ------: | -----: | ------: | ----: | -------: | --------------: | --: | --: | --: | -------: |
| [HumanML3D M2T](../tasks/m2t.md) | 4,400 | 0.0460 | 0.3365 | 0.0782 | 0.8851 | 0.3193 | 0.5087 | 0.6880 | 0.7776 | 3.1121 |

The [M2T case explorer](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard#case-explorer)
contains MotionGPT's prediction for every one of the 4,400 evaluated clips.

### M2T Demo Cases

| Human reference | MotionGPT prediction | Motion |
| --------------- | -------------------- | ------ |
| a man kicks something or someone with his left leg. | a person kicks with his left hand. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| person jogs around to the left and right | a person jogs to the right and then to the left | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| a person jumping while raising both hands and moving apart legs. | a person doing jumping jacks. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

Motius encodes every input clip at its true length by default, so a caption is
independent of unrelated samples in the API batch. The released evaluator's
batch-dependent zero-padding can be reproduced explicitly with
`pad_to_batch_max=True`, but that diagnostic variant is excluded from ranking.

## Motion Representation

MotionGPT generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The VQ-VAE converts normalized HumanML3D features into discrete motion tokens.
MotionGPT then treats those tokens as a language vocabulary item alongside text
tokens.

## Citation and License

```bibtex
@inproceedings{jiang2023motiongpt,
  title={MotionGPT: Human Motion as a Foreign Language},
  author={Jiang, Biao and Chen, Xin and Liu, Wen and Yu, Jingyi and Yu, Gang and Chen, Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
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
