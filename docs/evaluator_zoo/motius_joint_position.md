<h1 align="center">Motius Joint-Position Evaluator Card</h1>

<p align="center">
  <strong>Universal SMPL-22 joint-position evaluator for cross-model T2M reporting.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2305.00976">TMR Paper</a> |
  <a href="https://mathis.petrovich.fr/tmr/">TMR Project Page</a> |
  <a href="https://github.com/Mathux/TMR">Original TMR GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66">Motius Checkpoint</a>
</p>

The Motius Joint-Position Evaluator is a TMR architecture reproduction trained
on full HYMotion Data SFT and the full single-person MotionHub training union.
It scores canonicalized SMPL-22 joint positions, avoiding the rotation-twist
ambiguity that can affect comparisons in SMPL rotation space.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | Motius Joint-Position Evaluator |
| Architecture | TMR-style text/motion encoders with reconstruction decoder |
| Motion representation | Canonicalized SMPL-22 joints66, 22 joints in xyz at 30 fps |
| FK implementation | Neutral SMPL body model; the public body skeleton has 22 joints |
| Training data | Full HYMotion Data SFT + full single-person MotionHub training union |
| Training checkpoint | Epoch 248, FP32 |
| Caption protocol | HumanML3D selected captions; MotionHub official test annotations |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint | [ZeyuLing/motius-evaluator-universal-smplh-joints66](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66) |
| Artifact format | Safetensors + bundled DistilBERT + corrected joints66 training statistics |

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

## Usage

```python
from motius.evaluation.evaluators import TMRTextMotionEvaluator

evaluator = TMRTextMotionEvaluator.from_pretrained(
    "ZeyuLing/motius-evaluator-universal-smplh-joints66",
    device="cuda",
)

metrics = evaluator.evaluate(
    captions,
    predicted_joints66,
    reference_joints66,
    chunk_size=32,
    n_repeats=1,
)
```

`FID` is computed after independently L2-normalizing every reference and
generated motion embedding. MM-Dist, R-Precision, and Diversity retain the
native uTMR embedding scale. Raw-space uTMR FID must not be reported in Motius
leaderboards or model cards.

For a materialized HumanML3D selected-caption split, the repository CLI loads
the paired captions and GT joints, enforces one selected caption per sample,
and writes the complete metric JSON:

```bash
python tools/eval_t2m_joint_evaluator.py \
  --dataset-dir /path/to/universal_joints66_dataset \
  --split humanml3d_test \
  --predictions-dir outputs/evaluation/t2m/humanml3d_official_test/joints66/my_method \
  --method my_method \
  --output outputs/evaluation/t2m/humanml3d_official_test/metrics/my_method/joints66.json \
  --n-repeats 1
```

The downloaded directory contains `model.safetensors`, `config.json`,
`preprocessor_config.json`, `text_encoder/`, joints66 statistics, and an SHA256
manifest. The bundled tokenizer and DistilBERT weights allow the same API to
load from a local snapshot with `local_files_only=True` on offline workers.

## Reporting Rule

Every T2M model card should include this row and label its FID as normalized
uTMR FID. If the joint-position evaluator
has not been run for a method yet, the row should be marked `Pending`.

## Notes

The evaluator expects the unified SMPL-22 body skeleton and canonicalized joint
positions. SMPL-22 is the pelvis-to-wrist body subset shared by SMPL and
SMPL-H. Methods that generate HumanML3D-263,
MotionStreamer-272, SMPL, SMPL-X, or DART-style motion must use the checked
conversion path before reporting this metric.
