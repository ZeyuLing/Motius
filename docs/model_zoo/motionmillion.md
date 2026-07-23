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

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![MotionMillion-7B HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionmillion/motionmillion_7b_train_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![MotionMillion-7B HumanML3D 004545 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionmillion/motionmillion_7b_train_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![MotionMillion-7B HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionmillion/motionmillion_7b_train_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | Go to Zero / MotionMillion |
| Tasks | Text-to-Motion |
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

Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | 7B train-only | 3,970 | 0.523 | 0.721 | 0.817 | 0.065 | 2.897 | 9.394 | Measured |
| MotionStreamer Evaluator | 7B train-only | 4,042 | 0.740 | 0.878 | 0.924 | 3.081 | 15.371 | 27.575 | Measured |
| Motius Joint-Position Evaluator | 7B train-only | 4,034 | 0.628 | 0.790 | 0.858 | 33.602 | 29.968 | 53.479 | Measured |
| HumanML3D Official | 3B train-only | 3,970 | 0.528 | 0.723 | 0.818 | 0.071 | 2.882 | 9.379 | Measured |
| MotionStreamer Evaluator | 3B train-only | 4,042 | 0.740 | 0.877 | 0.923 | 3.066 | 15.381 | 27.560 | Measured |
| Motius Joint-Position Evaluator | 3B train-only | 4,034 | 0.623 | 0.786 | 0.857 | 34.414 | 29.966 | 54.624 | Measured |

## Motion Representation

MotionMillion generates `humanml3d_272`, the same 272-dim, 30 fps layout used by
MotionStreamer:

```text
text -> Flan-T5-XL -> LLaMA AR -> FSQ dequantize
     -> HumanVQVAE decoder -> MotionStreamer-272 motion
```

Because this representation matches MotionStreamer-272 directly, native
evaluation does not need an additional rotation re-encoding step.

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
