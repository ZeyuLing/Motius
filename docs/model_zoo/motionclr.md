<h1 align="center">MotionCLR Model Card</h1>

<p align="center">
  <strong>Attention-aware diffusion for HumanML3D text-to-motion generation.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.18977">Paper</a> ·
  <a href="https://lhchen.top/MotionCLR">Project Page</a> ·
  <a href="https://github.com/IDEA-Research/MotionCLR">Original GitHub</a> ·
  <a href="https://huggingface.co/EvanTHU/MotionCLR">Official Weights</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d">Motius Checkpoint</a>
</p>

MotionCLR is the method from *MotionCLR: Motion Generation and Training-free
Editing via Understanding Attention Mechanisms* (Chen et al., 2024). This
release reproduces the official HumanML3D text-to-motion inference path inside
Motius, including its U-Net denoiser, diffusion schedule, OpenAI CLIP ViT-B/32
text encoder, classifier-free guidance, and HumanML3D statistics. It does not
import an external MotionCLR checkout at runtime.

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
| Text-to-Motion | a person hops in place twice. | <video src="https://github.com/user-attachments/assets/4e09cb36-da6e-4150-8da5-b1d6452e7953" controls></video> | [MP4](https://github.com/user-attachments/assets/4e09cb36-da6e-4150-8da5-b1d6452e7953) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motionclr) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ------------------- | ------------ |
| a person is waving with their right hand. | <video src="https://github.com/user-attachments/assets/052fb09b-e0ab-4bcb-8e40-c68bfc382ea1" controls></video> |
| a person hops in place twice. | <video src="https://github.com/user-attachments/assets/4e09cb36-da6e-4150-8da5-b1d6452e7953" controls></video> |
| person walking at a average pace forward, swaying arms and torso with a sense of swagger | <video src="https://github.com/user-attachments/assets/0ff64b61-d760-41fe-bb15-6b51213f7eed" controls></video> |

512px / 30fps H.264 video previews rendered from the released HumanML3D test outputs.
The previews use cases whose HumanML3D-to-SMPL fitting MPJPE is below 25 mm;
the native model output remains HumanML3D-263 rather than SMPL parameters.

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
| Training motion | 20 fps (HumanML3D) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | MotionCLR attention-aware motion diffusion |
| Tasks | Text-to-Motion |
| Release | 2024 |
| Motion representation | HumanML3D-263 at 20 fps |
| Checkpoint | [`ZeyuLing/motius-motionclr-humanml3d`](https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d) |
| Pipeline | `motius.pipelines.motionclr.MotionCLRPipeline` |
| Upstream revision | `a6f44a791940682fe335c82f1b436bae05a1cebb` |
| License | IDEA License 1.0, included with the package and checkpoint |

The Motius artifact contains `model.safetensors`, HumanML3D mean/std arrays,
configuration and provenance metadata, and the frozen CLIP ViT-B/32 weight
under `clip/`. `from_pretrained` therefore loads a complete inference artifact
without a second model download.

### Capability Boundary

The released MotionCLR checkpoint and official inference code provide
HumanML3D text-to-motion generation and attention-map editing. They do not
define observed-prefix TP2M conditioning or BABEL multi-prompt sequential
generation, so MotionCLR is listed only on the T2M leaderboard.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionclr.MotionCLRPipeline` |
| Bundle | `motius.models.motionclr.MotionCLRBundle` |
| Network | `motius.models.motionclr.network` |
| Config | `configs/motionclr/motionclr_humanml3d.py` |

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-motionclr-humanml3d",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward and waves"],
    [120],
    seed=42,
)
```

`motions` is a list of unnormalized `(T, 263)` HumanML3D feature arrays at 20
fps. The official release settings use 10 DPM-Solver++ inference steps and
classifier-free guidance scale 2.5.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D official test motions with the leaderboard's fixed selected
caption for each sample. MotionStreamer and Motius Joint-Position results use
the shared neutral-SMPL conversion bridge; lower FID and MM-Dist are better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | 3,970 | 0.5527 | 0.7520 | 0.8488 | 0.1045 | 2.7019 | 9.6580 |
| MotionStreamer Evaluator | 4,042 | 0.3931 | 0.5208 | 0.5960 | 298.9693 | 22.5207 | 20.6074 |
| Motius Joint-Position Evaluator | 4,034 | 0.3795 | 0.5508 | 0.6458 | 0.5746 | 44.5428 | 50.3569 |

### Official Protocol Parity

The MotionCLR paper protocol is not the fixed selected-caption protocol above.
Its official HumanML3D loader randomly selects a caption and expands valid
time-tagged caption intervals into additional test samples. A one-run audit on
the 4,402 resulting entries reproduces the paper result closely:

| Source | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Multi-Modality |
| ------ | --: | --: | --: | --: | ------: | --------: | -------------: |
| Motius checkpoint with official loader | 0.5447 | 0.7406 | 0.8310 | 0.1076 | 2.8252 | 9.6821 | 1.8293 |
| MotionCLR paper, DPM-Solver | 0.542 | 0.733 | 0.827 | 0.099 | 2.981 | - | 2.145 |

The paper does not report Diversity in this table.

The released Motius network was also compared against the official EMA runtime
on identical prompts, lengths, seed, fp16 mode, and DPM-Solver schedule. The
three denormalized HML263 outputs had RMSE `0.00040`, `0.00539`, and `0.00041`;
the largest long-sequence discrepancy was concentrated in a foot-contact
channel. This audit separates implementation parity from caption-protocol
differences.

The cross-evaluator rows score the released samples after the same
HumanML3D-to-neutral-SMPL bridge used for every HML263 model. A diagnostic run
on the decoded pre-IK target joints reached joint-evaluator R@3 0.6468 and FID
1019.2886; the SMPL fit therefore contributes only a small part of the gap.
The GT sanity row for the same public evaluator reaches R@3 0.9058 and FID 0.

### Physical Quality

| Samples | Slide mm/frame | Float % | Jitter | Dynamic | PoseQ |
| ------: | -------------: | ------: | -----: | ------: | ----: |
| 4,042 | 3.9074 | 10.1052 | 8.7418 | 23.4910 | 3.1287 |

## Motion Representation

MotionCLR predicts the standard HumanML3D-263 representation:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| Root motion and height | 4 | root angular velocity, local XZ velocity, and height |
| Relative joint positions | 63 | 21 non-root joints in the root frame |
| Local joint rotations | 126 | 21 continuous 6D rotations |
| Local joint velocities | 66 | 22 joint velocities |
| Foot contacts | 4 | binary left/right heel and toe contacts |

Motius converts this representation to neutral-SMPL `motion135`, SMPL-22
`joints66`, and MotionStreamer-272 through its public motion APIs before
cross-evaluator reporting.

## Citation and License

```bibtex
@article{chen2024motionclr,
  title={MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms},
  author={Chen, Ling-Hao and Dai, Wenxun and Ju, Xuan and Lu, Shunlin and Zhang, Lei},
  journal={arXiv preprint arXiv:2410.18977},
  year={2024}
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
