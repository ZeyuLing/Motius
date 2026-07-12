<h1 align="center">Motius Joint-Position Evaluator Card</h1>

<p align="center">
  <strong>Universal SMPL-H joint-position evaluator for cross-model T2M reporting.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2305.00976">TMR Paper</a> |
  <a href="https://mathis.petrovich.fr/tmr/">TMR Project Page</a> |
  <a href="https://github.com/Mathux/TMR">Original TMR GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66">Motius Checkpoint</a>
</p>

The Motius Joint-Position Evaluator is a TMR architecture reproduction trained
on full HYMotion Data SFT and the full single-person MotionHub training union.
It scores canonicalized SMPL-H joint positions, avoiding the rotation-twist
ambiguity that can affect comparisons in SMPL rotation space.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | Motius Joint-Position Evaluator |
| Architecture | TMR-style text/motion encoders with reconstruction decoder |
| Motion representation | Canonicalized SMPL-H joints66, 22 joints in xyz at 30 fps |
| Training data | Full HYMotion Data SFT + full single-person MotionHub training union |
| Training checkpoint | Epoch 248, FP32 |
| Caption protocol | HumanML3D selected captions; MotionHub official test annotations |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint | [ZeyuLing/motius-evaluator-universal-smplh-joints66](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66) |
| Artifact format | Safetensors + corrected joints66 training statistics |

## Provenance

The architecture is reproduced from **TMR: Text-to-Motion Retrieval Using
Contrastive 3D Human Motion Synthesis** and the official
[`Mathux/TMR`](https://github.com/Mathux/TMR) repository. Motius reimplements
the architecture in its own model/trainer stack and trains this checkpoint from
scratch on the datasets above. Therefore, this is a **Motius-trained TMR
reproduction**, not an official TMR checkpoint.

The published artifact contains only the epoch-248 inference model and the
normalization statistics used by that corrected run. Optimizer state,
distributed random states, and local dataset caches are excluded.

## Download

```python
from huggingface_hub import snapshot_download

checkpoint_dir = snapshot_download(
    repo_id="ZeyuLing/motius-evaluator-universal-smplh-joints66"
)
```

The downloaded directory contains `model.safetensors`, `config.json`,
`preprocessor_config.json`, joints66 statistics, and an SHA256 manifest.

## Reporting Rule

Every T2M model card should include this row. If the joint-position evaluator
has not been run for a method yet, the row should be marked `Pending`.

## Notes

The evaluator expects a unified SMPL-H skeleton and canonicalized joint
positions. Methods that generate HumanML3D-263, MotionStreamer-272, SMPL,
SMPL-X, or DART-style motion must use the checked conversion path before
reporting this metric.
