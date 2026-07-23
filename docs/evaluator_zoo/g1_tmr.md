<h1 align="center">Motius TMR-G1 Evaluator Card</h1>

<p align="center">
  <strong>Text-motion evaluator trained directly in Unitree G1 motion space.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2305.00976">TMR Paper</a> |
  <a href="https://mathis.petrovich.fr/tmr/">TMR Project Page</a> |
  <a href="https://github.com/Mathux/TMR">Original TMR GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-evaluator-g1-38d-tmr">Motius Checkpoint</a>
</p>

The Motius TMR-G1 Evaluator is a Motius-native reproduction of the TMR
architecture trained from scratch on retargeted Unitree G1 motion. It evaluates
robot-native text-to-motion methods without converting generated G1 joint
angles back through SMPL.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | Motius TMR-G1 Evaluator |
| Architecture | TMR-style text/motion encoders with reconstruction decoder |
| Motion representation | Canonicalized Unitree G1-38D at 30 fps |
| Input layout | root XY velocity + root height + root rotation 6D + 29 joint angles |
| Training corpus | HYMotion Data G1 materialization, 359,153 clips |
| Split | 344,966 train / 7,083 validation / 7,104 test |
| Training checkpoint | Epoch 139 |
| Metrics | R@1, R@2, R@3, normalized FID, MM-Dist, Diversity |
| Checkpoint | [ZeyuLing/motius-evaluator-g1-38d-tmr](https://huggingface.co/ZeyuLing/motius-evaluator-g1-38d-tmr) |
| Artifact format | Safetensors + G1-38D training statistics |

## Input Protocol

Every input motion must have shape `(T, 38)` and use the public `g1_38`
convention:

```text
[0:2]   root XY velocity
[2]     root Z height
[3:9]   root rotation 6D, first two matrix columns flattened row-wise
[9:38]  29 Unitree G1 joint angles in Motius order
```

The root is canonicalized to the frame-0 ground origin and zero heading before
encoding. Use `convert_motion(qpos, "g1_qpos", "g1_38")` for generated MuJoCo
qpos arrays rather than rebuilding these channels manually.

## Python API

```python
from motius.evaluation import TMRG1Evaluator
from motius.motion import convert_motion

evaluator = TMRG1Evaluator.from_pretrained(
    "ZeyuLing/motius-evaluator-g1-38d-tmr",
    device="cuda",
)

predicted_g1 = [
    convert_motion(qpos, "g1_qpos", "g1_38")
    for qpos in predicted_qpos
]
metrics = evaluator.evaluate(
    captions,
    predicted_g1,
    reference_motions=reference_g1,
    chunk_size=32,
    n_repeats=1,
)
```

Passing reference motions adds FID to the retrieval, MM-Dist, and Diversity
results. Without references, the evaluator reports R-Precision, MM-Dist, and
prediction Diversity only.

As with every Motius uTMR evaluator, FID is measured after per-sample L2
normalization of the reference and generated embeddings. Raw-space uTMR FID is
not a supported reporting metric.

## Command Line

Prepare a JSONL manifest with `id` and `caption` fields. Store one `(T, 38)`
`.npy` or `.npz` per ID in the prediction and optional reference directories.

```bash
python tools/eval_g1_tmr.py \
  --manifest outputs/evaluation/text_to_motion/text_to_motion_unitree_g1/unitree-g1-paper-eval-1024-v1/protocol/manifest.jsonl \
  --pred-dir outputs/evaluation/text_to_motion/text_to_motion_unitree_g1/unitree-g1-paper-eval-1024-v1/runs/my-method/release-1/predictions/g1_38 \
  --reference-dir outputs/evaluation/text_to_motion/text_to_motion_unitree_g1/unitree-g1-paper-eval-1024-v1/protocol/references/g1_38 \
  --output outputs/evaluation/text_to_motion/text_to_motion_unitree_g1/unitree-g1-paper-eval-1024-v1/runs/my-method/release-1/metrics/summary.json
```

The default protocol uses recall chunks of 32 and one deterministic repeat.
Set `--repeats` explicitly when a benchmark requires repeated shuffles.
The shared directory contract is documented in
[Evaluation Artifact Layout](../evaluation/artifact_layout.md).

## Provenance

The architecture is reproduced from **TMR: Text-to-Motion Retrieval Using
Contrastive 3D Human Motion Synthesis** and the official
[`Mathux/TMR`](https://github.com/Mathux/TMR) repository. The checkpoint is
trained by Motius on the G1 corpus above; it is not an official TMR checkpoint.

This evaluator is intended for G1-native generation and embodied-motion model
cards. It does not replace the three human-motion evaluator views used by the
standard Motius T2M leaderboard.
