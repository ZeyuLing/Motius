---
license: other
license_name: s-lab-license-1.0
license_link: https://github.com/lisiyao21/Bailando/blob/master/LICENSE
library_name: motius
tags:
  - motion-generation
  - music-to-dance
  - aistplusplus
  - bailando
datasets:
  - yeok/danceba
---

<h1 align="center">Bailando Model Card</h1>

<p align="center">
  <strong>Actor-critic music-to-dance generation with a learned choreographic memory.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2203.13055">Paper</a> |
  <a href="https://www.mmlab-ntu.com/project/bailando/">Project Page</a> |
  <a href="https://github.com/lisiyao21/Bailando">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP">Motius Checkpoint</a> |
  <a href="https://github.com/ZeyuLing/Motius/blob/main/docs/tasks/music_to_dance.md">Task Protocol</a>
</p>

Bailando is the CVPR 2022 oral work *Bailando: 3D Dance Generation by
Actor-Critic GPT with Choreographic Memory*. It learns upper- and lower-body
VQ codebooks, then composes those dance units autoregressively from music. The
Motius release implements the model, audio frontend, AIST++ dataset, pipeline,
representation bridge, and evaluator without importing an external checkout at
runtime.

## Preview

<table>
  <tr>
    <td width="33%"><img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/bailando/bailando_aistpp_break_gBR_mBR0_smpl_mesh_512_30fps.gif" alt="Bailando break dance"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/bailando/bailando_aistpp_krump_gKR_mKR2_smpl_mesh_512_30fps.gif" alt="Bailando krump dance"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/bailando/bailando_aistpp_waacking_gWA_mWA0_smpl_mesh_512_30fps.gif" alt="Bailando waacking dance"></td>
  </tr>
  <tr>
    <td align="center"><sub>Break / <code>gBR...mBR0</code></sub></td>
    <td align="center"><sub>Krump / <code>gKR...mKR2</code></sub></td>
    <td align="center"><sub>Waacking / <code>gWA...mWA0</code></sub></td>
  </tr>
</table>

The previews are distinct AIST++ evaluation outputs rendered as neutral SMPL
meshes at 512x512 and 30 fps. Their position-IK fit errors are 13.75, 13.89,
and 13.71 mm. MP4 sources and fit reports are stored beside the GIF assets.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Task | Music-to-Dance |
| Dataset | AIST++ cross-modal split |
| Music input | 438D features at 7.5 fps, or raw audio through the bundled frontend |
| Native motion | AIST++ global SMPL-24 joint positions at 60 fps |
| Parameters | 173,368,139 |
| Checkpoint | [`ZeyuLing/Motius-Bailando-AISTPP`](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP) |
| Pipeline | `motius.pipelines.bailando.BailandoPipeline` |
| Upstream revision | `lisiyao21/Bailando@cc90b98bff81c9709570db413c9610c2562e27ca` |
| License | S-Lab License 1.0, non-commercial use |

The Hugging Face artifact contains both VQ-VAE branches and the actor-critic
GPT as safetensors, plus the complete architecture config, source hashes,
license, and attribution. It does not require an upstream repository or a
second checkpoint download.

## Usage

Install the music frontend dependencies:

```bash
python -m pip install -e '.[music-to-dance]'
```

Generate from an audio file:

```python
from motius.pipelines.bailando import BailandoPipeline

pipe = BailandoPipeline.from_pretrained(
    "ZeyuLing/Motius-Bailando-AISTPP",
    device="cuda",
)
result = pipe("music.wav")

print(result.joints.shape)       # (batch, frames, 24, 3)
print(result.music_features.shape)  # (batch, music_frames, 438)
```

For exact benchmark reproduction, pass the released 438D AIST++ feature stream
and paired initial motion. Only the first upper/lower VQ token initializes the
generation, matching the official script:

```python
result = pipe(
    music_features=music_features_7p5fps,
    initial_motion=paired_gt_smpl24,
)
```

Without `initial_motion`, the public demo seed `(423, 12)` is used. Raw-audio
inference is a convenience path; use released precomputed features when exact
paper parity across audio-library versions matters.

## Evaluation

Motius ran the converted official epoch-500 VQ-VAE and epoch-10 GPT on all 40
cross-modal validation/test cases. FID and diversity use the 1,320 valid motion
PKLs in the AIST++ v1 reference archive. Generated features use the first 1,200
frames; reference features use complete sequences; BeatAlign uses the complete
generated sequence and the paired 60 fps music-beat stream.

| Result | FID_k | FID_g | uTMR FID | Diversity_k | Diversity_g | BeatAlign |
| ------ | ----: | ----: | --------: | ----------: | ----------: | --------: |
| Motius reproduction | 28.11 | 9.70 | 0.3138 | 7.73 | 6.31 | 0.2268 |
| Bailando paper | 28.16 | 9.62 | - | 7.83 | 6.34 | 0.2332 |
| Motius GT | 17.16 | 10.66 | 0.1829 | 8.17 | 7.49 | 0.2247 |
| GT paper | 17.10 | 10.60 | - | 8.19 | 7.45 | 0.2374 |

Lower is better for FID, higher is better for BeatAlign, and diversity is
interpreted relative to GT. uTMR FID uses canonical 30 fps SMPL-22 joints and
per-sample L2-normalized embeddings against the same 1,320-motion reference
pool. The paper values are shown as parity targets and
are not copied into the reproduced row.

### Physical Diagnostics

These Motius joint-level diagnostics use the common SMPL-22 subset. They are
not metrics from the Bailando paper.

| Result | Jitter | Dynamic | Penetration | Float | Slide |
| ------ | -----: | ------: | ----------: | ----: | ----: |
| Bailando | 0.00558 | 0.02183 | 0.00000 | 0.21803 | 0.00428 |
| Paired GT | 0.00677 | 0.02276 | 0.00000 | 0.10658 | 0.00330 |

`Dynamic` is an expressiveness statistic to compare with GT rather than
minimize. The floor-dependent diagnostics are reported in native metric units.

## Motion Representation

The public representation name is `aistpp_smpl24_joints`, shape `(T,24,3)` in
metres with Y up. Joints `0:22` are the standard SMPL body chain and convert
exactly to `smpl22_joints`:

```python
from motius.motion import convert_motion

smpl22 = convert_motion(
    result.joints[0],
    source="aistpp_smpl24_joints",
    target="smpl22_joints",
)
```

Conversion to `motion135` and SMPL mesh uses position IK because the generated
tensor stores joint positions rather than local rotations. The three preview
reports expose the resulting fit errors instead of hiding this lossy step.

## Reproduction Audit

| Check | Result |
| ----- | ------ |
| Official checkpoint load | Zero missing and zero unexpected tensors |
| VQ-VAE source SHA-256 | `35670f42a3b3092438f73f0af3ace7b52e318a8b5c00b2b05c92078176b21716` |
| GPT source SHA-256 | `903863a4e1cac01fcec30f7939c591eac8ea89f74e9837b93babe3383eecb403` |
| Generated cases | 40/40, all finite |
| Full inference time | 80.17 seconds on one H20 |
| Reference pool | 1,365 PKLs minus the official 45-entry ignore list = 1,320 |
| SMPL-24 FK calibration | 0.000066 mm MPJPE over 540,384 joint-frames |

The model code remains under the upstream S-Lab License 1.0. AIST++
annotations are CC BY 4.0. See the artifact `LICENSE` and `ATTRIBUTIONS.md`
before redistribution or commercial use.

## Citation

```bibtex
@inproceedings{siyao2022bailando,
  title={Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory},
  author={Siyao, Li and Yu, Weijiang and Gu, Tianpei and Lin, Chunze and Wang, Quan and Qian, Chen and Loy, Chen Change and Liu, Ziwei},
  booktitle={CVPR},
  year={2022}
}
```
