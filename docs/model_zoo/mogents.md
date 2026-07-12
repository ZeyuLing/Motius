<h1 align="center">MoGenTS Model Card</h1>

<p align="center">
  <strong>Spatial-temporal joint token modeling for text-to-motion, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2409.17686">Paper</a> |
  <a href="https://aigc3d.github.io/mogents/">Project Page</a> |
  <a href="https://github.com/weihaosky/mogents">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d">Motius Checkpoint</a>
</p>

MoGenTS is the text-to-motion baseline from *MoGenTS: Motion Generation based
on Spatial-Temporal Joint Modeling* (Yuan et al., NeurIPS 2024). This Motius
release packages the dual-stream RVQ-VAE, 1D auxiliary token transformer, 2D
spatial-temporal token transformer, residual token refiners, optional length
estimator, CLIP ViT-B/32 text encoder loading, and HumanML3D-263
denormalization behind a consistent inference pipeline.

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![MoGenTS HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mogents/mogents_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![MoGenTS HumanML3D 004545 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mogents/mogents_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![MoGenTS HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mogents/mogents_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MoGenTS, spatial-temporal discrete motion tokens |
| Task | Text-to-Motion |
| Venue | NeurIPS 2024 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Tokenizer | Dual-stream RVQ-VAE, 1D auxiliary tokens plus 2D joint-token grid |
| Checkpoint | [`ZeyuLing/hftrainer-mogents-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) |
| Pipeline | `motius.pipelines.mogents.MoGenTSPipeline` |

The checkpoint artifact contains `vq.safetensors`, `mask_aux.safetensors`,
`mask_ts.safetensors`, `res_aux.safetensors`, `res_ts.safetensors`,
`length_est.safetensors`, `clip.safetensors`, `mogents_config.json`,
`Mean.npy`, and `Std.npy`.

## Usage

```python
from motius.pipelines.mogents import MoGenTSPipeline

pipe = MoGenTSPipeline.from_pretrained(
    "ZeyuLing/hftrainer-mogents-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then turns around"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale. If `lengths` is omitted, the packaged
length estimator samples a token length from the prompt embedding.

## Evaluation Results

Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL/SMPL-H evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,970 | 0.522 | 0.713 | 0.806 | 0.081 | 2.929 | 9.406 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.499 | 0.652 | 0.735 | 20.186 | 19.535 | 25.697 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.462 | 0.624 | 0.714 | 158.588 | 36.714 | 56.513 | Measured |


## Motion Representation

MoGenTS generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The model tokenizes motion into a 1D auxiliary stream and a 2D joint-token map.
The 2D stream preserves spatial-temporal structure before decoding back to the
standard 263-dim HumanML3D representation.


## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mogents.MoGenTSPipeline` |
| Bundle | `motius.models.mogents.MoGenTSBundle` |
| Runtime | `motius.models.mogents.network` |

The runtime is independent from the original checkout for inference. Raw
upstream checkpoint conversion remains outside this public release surface.

## Citation

```bibtex
@inproceedings{yuan2024mogents,
  title={MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling},
  author={Yuan, Weihao and Shen, Weichao and He, Yisheng and Dong, Yuan and Gu, Xiaodong and Dong, Zilong and Bo, Liefeng and Huang, Qixing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
