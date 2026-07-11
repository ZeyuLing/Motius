<h1 align="center">ViMoGen Model Card</h1>

<p align="center">
  <strong>Generalizable motion generation with visual generative priors, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.26794">Paper</a> |
  <a href="https://motrixlab.github.io/2026_iclr_vimogen/">Project Page</a> |
  <a href="https://github.com/MotrixLab/ViMoGen">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d">Motius Checkpoint</a>
</p>

ViMoGen is the motion model from *The Quest for Generalizable Motion
Generation: Data, Model, and Evaluation*. This Motius release packages the
released 1.3B HumanML3D checkpoint behind the same bundle/pipeline API used by
the rest of the Model Zoo.

Validated SMPL previews will be added after the render outputs are checked
against the shared SMPL conversion path.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | ViMoGen 1.3B |
| Task | Text-to-Motion |
| Venue | ICLR 2026 |
| Motion representation | DART276, 20 fps |
| Text encoder | Wan2.1 T2V-1.3B UMT5-XXL encoder |
| Backbone | WanVideoTM2M 1.3B flow-matching DiT |
| Default sampler | Flow matching, 50 inference steps |
| Checkpoint | [`ZeyuLing/hftrainer-vimogen-1.3b-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) |
| Pipeline | `motius.pipelines.vimogen.ViMoGenPipeline` |

The checkpoint artifact contains `model.pt`, `model_index.json`, and
`assets/meta/{mean,std}.npy`. The Wan2.1 base assets are resolved from the
public `Wan-AI/Wan2.1-T2V-1.3B` Hub repo declared by `wan_repo_id`.

## Usage

Install the ViMoGen extra dependencies:

```bash
python -m pip install -e ".[dev,vimogen]"
```

Run text-to-motion inference:

```python
from motius.pipelines.vimogen import ViMoGenPipeline

pipe = ViMoGenPipeline.from_pretrained(
    "ZeyuLing/hftrainer-vimogen-1.3b-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
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

Protocol: HumanML3D official test split (`n=4,042`) with the shared corrected
caption set. ViMoGen generation uses a ViMoGen-style prompt rewrite derived
from the corrected caption, while semantic evaluators are computed against the
same corrected caption protocol as other methods. For FID and MM-Dist, lower is
better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | ---: | ---: | ---: | ---: | ------: | --------: |
| MotionStreamer-272, HML round-trip GT | 4,042 | 0.429 | 0.569 | 0.652 | 152.210 | 21.074 | 24.180 |
| MotionCLIP-135 no-L2, HML round-trip GT | 4,042 | 0.357 | 0.499 | 0.589 | 457.544 | 44.410 | 21.681 |

### Physical Diagnostics

| Slide | Float | Jitter | Dynamic | Penet |
| ----: | ----: | -----: | ------: | ----: |
| 6.949 | 23.727 | 4.437 | 16.384 | 0.000 |

## Motion Representation

ViMoGen emits DART276, the global DART-style representation:

```text
text -> UMT5-XXL embeddings -> WanVideoTM2M DiT -> denormalized DART276
```

The released pipeline returns DART276 directly. Converting DART276 to SMPL mesh
or cross-representation evaluator inputs should be done through a checked
conversion path before reporting metrics.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.vimogen.ViMoGenPipeline` |
| Bundle | `motius.models.vimogen.ViMoGenBundle` |
| Runtime | `motius.models.vimogen.network` |
| Scheduler | `motius.models.vimogen.network.vimogen.trainer.scheduler` |

The runtime vendors the required ViMoGen transformer modules and scheduler, so
inference does not import the upstream checkout.

## Citation

```bibtex
@article{lin2025quest,
  title={The Quest for Generalizable Motion Generation: Data, Model, and Evaluation},
  author={Lin, Jing and Wang, Ruisi and Lu, Junzhe and Huang, Ziqi and Song, Guorui and Zeng, Ailing and Liu, Xian and Wei, Chen and Yin, Wanqi and Sun, Qingping and Cai, Zhongang and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2510.26794},
  year={2025}
}
```
