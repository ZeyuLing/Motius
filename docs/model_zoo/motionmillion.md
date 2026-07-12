<h1 align="center">MotionMillion / Go to Zero Model Card</h1>

<p align="center">
  <strong>Million-scale zero-shot text-to-motion generation, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.07095">Paper</a> |
  <a href="https://vankouf.github.io/MotionMillion/">Project Page</a> |
  <a href="https://github.com/VankouF/MotionMillion-Codes">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272">7B Checkpoint</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272">3B Checkpoint</a>
</p>

MotionMillion, also released as *Go to Zero: Towards Zero-shot Motion
Generation with Million-scale Data* (Fan et al., ICCV 2025), is a large
autoregressive text-to-motion model trained for zero-shot generalization. This
Motius release packages the FSQ HumanVQVAE tokenizer, LLaMA-style autoregressive
motion generator, Flan-T5-XL text encoder artifact, and MotionStreamer-272
normalization into a consistent inference pipeline.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | Go to Zero / MotionMillion |
| Task | Zero-shot Text-to-Motion |
| Venue | ICCV 2025 |
| Motion representation | MotionStreamer-272 / humanml3d_272, 30 fps |
| Text encoder | Flan-T5-XL, frozen |
| Tokenizer | HumanVQVAE + FSQ, levels `[8,8,8,5,5,5]` |
| AR model | LLaMA-style 3B / 7B transformer |
| Checkpoints | [`7B train-only`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272), [`3B train-only`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272) |
| Pipeline | `motius.pipelines.motionmillion.MotionMillionPipeline` |

The checkpoint artifacts contain `fsq.safetensors`, `ar.safetensors`,
`mm_config.json`, `model_index.json`, `mean.npy`, `std.npy`, and a packaged
`text_encoder/` directory.

## Usage

```python
from motius.pipelines.motionmillion import MotionMillionPipeline

pipe = MotionMillionPipeline.from_pretrained(
    "ZeyuLing/hftrainer-gotozero-7b-train-humanml272",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person swings a golf club"],
    max_sample_steps=150,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 272)` and is
denormalized to the MotionStreamer-272 physical scale.

## Evaluation Results

Protocol: HumanML3D official-test caption protocol, native MotionStreamer-272
evaluation, `4,042` generated files and `4,032` evaluator-consumed samples
after the standard R-Precision batching. For FID and MM-Dist, lower is better.

| Evaluator | Model | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ----- | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | 7B train-only | - | - | - | - | - | - | Pending |
| HumanML3D Official | 3B train-only | - | - | - | - | - | - | Pending |
| MotionStreamer Evaluator | 7B train-only | 0.740 | 0.878 | 0.924 | 3.081 | 15.371 | 27.575 | Measured |
| MotionStreamer Evaluator | 3B train-only | 0.740 | 0.877 | 0.923 | 3.066 | 15.381 | 27.560 | Measured |
| Motius Joint-Position Evaluator | 7B train-only | - | - | - | - | - | - | Pending |
| Motius Joint-Position Evaluator | 3B train-only | - | - | - | - | - | - | Pending |

Ground-truth sanity row:

| Evaluator | Model | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ----- | --: | --: | --: | --: | ------: | --------: |
| MotionStreamer Evaluator | Real motions | 0.778 | 0.906 | 0.946 | 0.000 | 14.820 | 27.853 |

## Motion Representation

MotionMillion generates `humanml3d_272`, the same 272-dim, 30 fps layout used by
MotionStreamer:

```text
text -> Flan-T5-XL -> LLaMA AR -> FSQ dequantize
     -> HumanVQVAE decoder -> MotionStreamer-272 motion
```

Because this representation matches MotionStreamer-272 directly, native
evaluation does not need an additional rotation re-encoding step.

## Qualitative Results

Validated SMPL previews will be added to this card once the public qualitative
assets are rendered through the shared SMPL-H visualization path. The current
release keeps the model card focused on reproducible checkpoint loading and
numeric evaluation rather than shipping unverified preview media.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionmillion.MotionMillionPipeline` |
| Bundle | `motius.models.motionmillion.MotionMillionBundle` |
| Runtime | `motius.models.motionmillion.network` |

The runtime includes only the inference path: FSQ tokenizer, HumanVQVAE decoder,
and LLaMA-style autoregressive generator.

## Citation

```bibtex
@inproceedings{fan2025gotozero,
  title={Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data},
  author={Fan, Ke and Lu, Shunlin and Dai, Minyue and Yu, Runyi and Xiao, Lixing and Dou, Zhiyang and Dong, Junting and Ma, Lizhuang and Wang, Jingbo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
