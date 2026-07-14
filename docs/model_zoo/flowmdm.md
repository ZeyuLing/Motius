<h1 align="center">FlowMDM Model Card</h1>

<p align="center">
  <strong>Seamless multi-prompt human motion composition, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2402.15509">Paper</a> |
  <a href="https://barquerogerman.github.io/FlowMDM/">Project Page</a> |
  <a href="https://github.com/BarqueroGerman/FlowMDM">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d">HumanML3D Checkpoint</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-flowmdm-babel">BABEL Checkpoint</a>
</p>

FlowMDM is the motion composition baseline from *Seamless Human Motion
Composition with Blended Positional Encodings* (Barquero et al., CVPR 2024).
This Motius release packages the MDM-style diffusion model, blended positional
encoding sampler, HumanML3D statistics, and text-to-motion / multi-prompt
pipeline methods without requiring the original checkout.

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![FlowMDM HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/flowmdm/flowmdm_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![FlowMDM HumanML3D 004545 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/flowmdm/flowmdm_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![FlowMDM HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/flowmdm/flowmdm_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | FlowMDM, diffusion with blended positional encodings |
| Tasks | T2M, Sequential Generation, TP2M |
| Venue | CVPR 2024 |
| Motion representation | HumanML3D-263 at 20 fps; BABEL-135 at 30 fps |
| Checkpoints | [`HumanML3D`](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d), [`BABEL`](https://huggingface.co/ZeyuLing/motius-flowmdm-babel) |
| Pipeline | `motius.pipelines.flowmdm.FlowMDMPipeline` |

The HumanML3D artifact contains `model000500000.pt`, `args.json`, `Mean.npy`,
`Std.npy`, and `model_index.json`. The BABEL artifact contains the official
`model001300000.pt`, `args.json`, BABEL normalization statistics, license, and
Motius model index.

## Usage

```python
from motius.pipelines.flowmdm import FlowMDMPipeline

pipe = FlowMDMPipeline.from_pretrained(
    "ZeyuLing/hftrainer-flowmdm-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)
```

Sequential generation is exposed through the BABEL checkpoint with the same
pipeline class:

```python
babel_pipe = FlowMDMPipeline.from_pretrained(
    "ZeyuLing/motius-flowmdm-babel",
    bundle_kwargs={"device": "cuda"},
    device="cuda",
)
motions = babel_pipe.infer_sequential_t2m(
    [["a person walks forward", "then turns around"]],
    [[80, 80]],
)
```

`motions` is a list of NumPy arrays. HumanML3D outputs have shape `(T, 263)`;
BABEL outputs have shape `(T, 135)`. Both are returned in physical scale.

## Evaluation Results

Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,970 | 0.439 | 0.636 | 0.744 | 0.327 | 3.387 | 9.942 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.474 | 0.650 | 0.731 | 36.377 | 20.002 | 25.178 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.439 | 0.615 | 0.711 | 227.494 | 37.410 | 55.513 | Measured |

## BABEL Sequential Results

Protocol: 64 official BABEL validation compositions with 32 ordered prompts
per composition. Generated and reference clips are converted to canonical
SMPL-22 joints and evaluated with the Motius Joint-Position Evaluator. The
reference pools contain 2,048 independent BABEL validation segments and 2,048
30-frame transition windows.

| Method | Segments | R@1 | R@2 | R@3 | Semantic FID | MM-Dist | Transition FID | AUJ Gap |
| ------ | -------: | --: | --: | --: | -----------: | ------: | -------------: | ------: |
| FlowMDM BABEL | 2,048 | 0.2173 | 0.3389 | 0.4214 | 212.3355 | 49.9531 | 334.7499 | 29.2441 |

Full protocol and diagnostic statistics are maintained on the
[`BABEL Sequential Generation Leaderboard`](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard).


## TP2M Results

FlowMDM also supports prefix-conditioned TP2M evaluation under the published
[`Temporal Condition Leaderboard`](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard)
protocol. These results are separate from the multi-prompt BABEL benchmark
above.

| Condition Frames | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ----------------: | ------: | --: | --: | --: | --: | ------: | --------: |
| 1 | 3,968 | 0.449 | 0.630 | 0.706 | 83.773 | 19.872 | 26.365 |
| 5 | 3,968 | 0.481 | 0.654 | 0.729 | 75.853 | 19.456 | 26.467 |
| 9 | 3,968 | 0.490 | 0.664 | 0.742 | 71.338 | 19.262 | 26.625 |

## Motion Representation

FlowMDM generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |


## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.flowmdm.FlowMDMPipeline` |
| Bundle | `motius.models.flowmdm.FlowMDMBundle` |
| Runtime | `motius.models.flowmdm.network` |

The SMPL visualizer branch from the original implementation is stubbed for T2M
inference because the released HumanML3D checkpoint predicts HumanML3D-263
features directly.

## Citation

```bibtex
@inproceedings{barquero2024seamless,
  title={Seamless Human Motion Composition with Blended Positional Encodings},
  author={Barquero, German and Escalera, Sergio and Palmero, Cristina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
