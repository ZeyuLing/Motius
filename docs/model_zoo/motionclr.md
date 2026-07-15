<h1 align="center">MotionCLR Model Card</h1>

<p align="center">
  <strong>Attention-aware diffusion for HumanML3D text-to-motion generation.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.18977">Paper</a> |
  <a href="https://lhchen.top/MotionCLR">Project Page</a> |
  <a href="https://github.com/IDEA-Research/MotionCLR">Original GitHub</a> |
  <a href="https://huggingface.co/EvanTHU/MotionCLR">Official Weights</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-motionclr-humanml3d">Motius Checkpoint</a>
</p>

MotionCLR is the method from *MotionCLR: Motion Generation and Training-free
Editing via Understanding Attention Mechanisms* (Chen et al., 2024). This
release reproduces the official HumanML3D text-to-motion inference path inside
Motius, including its U-Net denoiser, diffusion schedule, OpenAI CLIP ViT-B/32
text encoder, classifier-free guidance, and HumanML3D statistics. It does not
import an external MotionCLR checkout at runtime.

## Preview

| HumanML3D Sample | Selected Input Text | SMPL Preview |
| ---------------- | ------------------- | ------------ |
| `014160` | a person is waving with their right hand. | ![MotionCLR HumanML3D 014160 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionclr/motionclr_humanml3d_014160_smpl_mesh_512_30fps.gif) |
| `003424` | a person hops in place twice. | ![MotionCLR HumanML3D 003424 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionclr/motionclr_humanml3d_003424_smpl_mesh_512_30fps.gif) |
| `004822` | person walking at an average pace forward, swaying arms and torso with a sense of swagger. | ![MotionCLR HumanML3D 004822 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionclr/motionclr_humanml3d_004822_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from the released HumanML3D test outputs.
The previews use cases whose HumanML3D-to-SMPL fitting MPJPE is below 25 mm;
the native model output remains HumanML3D-263 rather than SMPL parameters.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MotionCLR attention-aware motion diffusion |
| Task | T2M |
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

## Usage

```python
from motius.pipelines.motionclr import MotionCLRPipeline

pipe = MotionCLRPipeline.from_pretrained(
    "ZeyuLing/motius-motionclr-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward and waves"],
    [120],
    seed=42,
)
```

`motions` is a list of unnormalized `(T, 263)` HumanML3D feature arrays at 20
fps. The official release settings use 10 DPM-Solver++ inference steps and
classifier-free guidance scale 2.5.

## Evaluation Results

Protocol: HumanML3D official test motions with the leaderboard's fixed selected
caption for each sample. MotionStreamer and Motius Joint-Position results use
the shared neutral-SMPL conversion bridge; lower FID and MM-Dist are better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | 3,970 | 0.5527 | 0.7520 | 0.8488 | 0.1045 | 2.7019 | 9.6580 |
| MotionStreamer Evaluator | 4,042 | 0.3931 | 0.5208 | 0.5960 | 298.9693 | 22.5207 | 20.6074 |
| Motius Joint-Position Evaluator | 4,034 | 0.3569 | 0.5250 | 0.6250 | 1063.7324 | 44.8058 | 50.4069 |

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

## Capability Boundary

The released MotionCLR checkpoint and official inference code provide
HumanML3D text-to-motion generation and attention-map editing. They do not
define observed-prefix TP2M conditioning or BABEL multi-prompt sequential
generation, so MotionCLR is listed only on the T2M leaderboard.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionclr.MotionCLRPipeline` |
| Bundle | `motius.models.motionclr.MotionCLRBundle` |
| Network | `motius.models.motionclr.network` |
| Config | `configs/motionclr/motionclr_humanml3d.py` |

## Citation

```bibtex
@article{chen2024motionclr,
  title={MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms},
  author={Chen, Ling-Hao and Dai, Wenxun and Ju, Xuan and Lu, Shunlin and Zhang, Lei},
  journal={arXiv preprint arXiv:2410.18977},
  year={2024}
}
```
