<h1 align="center">MotionBricks Runtime Integration</h1>

<p align="center">
  <strong>Real-time Unitree G1 motion primitives with modular latent generators.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.24833">Paper</a> |
  <a href="https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks">Original GitHub</a>
</p>

MotionBricks is NVIDIA's real-time whole-body control stack released inside
GR00T-WholeBodyControl. It combines a VQVAE motion tokenizer, a pose model, and
a root model to stream controllable Unitree G1 motion primitives.

Motius vendors the Apache-2.0 source runtime under
`motius.models.motionbricks.network`, exposes it through a standard
`MotionBricksBundle` / `MotionBricksPipeline`, and keeps the multi-GB pretrained
weights outside the repository.

## Integration Snapshot

| Item | Value |
| ---- | ----- |
| Upstream system | Modular latent generative model plus smart primitives |
| Motius role | External runtime wrapper and G1 representation utilities |
| Native representation | MotionBricks G1 global 414D, local 413D, dual-root 418D |
| Robot skeleton | Unitree G1 29-DOF MuJoCo model |
| Public output | G1 qpos-36 stream |
| Checkpoint source | Official Git LFS files in GR00T-WholeBodyControl `motionbricks/out` |
| Pipeline | `motius.pipelines.motionbricks.MotionBricksPipeline` |

This integration is not registered as a Motius task and does not appear in the
task-based Model Zoo. The repository currently exposes runtime and
representation utilities, but does not define a stable robot-control task
pipeline or benchmark contract.

## Checkpoints

Place the official LFS files under `checkpoints/motionbricks`:

```text
checkpoints/motionbricks/
  G1-clip.ckpt
  motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt
  motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt
  motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt
```

From an official checkout:

```bash
git lfs install
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull --include="motionbricks/out/**" --exclude=""
ln -s "$PWD/motionbricks/out" /path/to/Motius/checkpoints/motionbricks
```

`MotionBricksBundle.validate_checkpoints()` checks both missing files and
unresolved Git LFS pointer files before loading the runtime.

## Usage

Install optional runtime dependencies:

```bash
pip install -e ".[motionbricks]"
```

Run a headless qpos rollout:

```python
from motius.pipelines.motionbricks import MotionBricksPipeline

pipe = MotionBricksPipeline.from_pretrained(
    "checkpoints/motionbricks",
    bundle_kwargs={"device": "cuda", "controller": "random"},
)

result = pipe.rollout(steps=240)
qpos = result["qpos"]          # (T, 36), Unitree G1 MuJoCo qpos
fps = result["fps"]            # 30
```

Use `controller="wasd"` for the interactive controller and `controller="random"`
for automated smoke tests or offline previews.

## Representation Notes

MotionBricks and ARDY both target Unitree G1, but their tensors are different:

| Representation | Shape | Meaning |
| -------------- | ----: | ------- |
| `motionbricks_g1_414` | 414D | Global-root subset used by the root model |
| `motionbricks_g1_413` | 413D | Local-root subset used by pose/tokenizer modules |
| `motionbricks_g1_418` | 418D | Full dual-root feature tensor |
| `g1_38` | 38D | Motius compact G1 representation |
| `g1_qpos` | 36D | MuJoCo root pose plus 29-DOF robot state |

The official MotionBricks converter is kept inside the vendored runtime; Motius
exposes the checkpoint/runtime wrapper first and will route broader
representation conversion through the shared G1 qpos API.
