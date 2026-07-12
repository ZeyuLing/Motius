<h1 align="center">HumanML3D Official Evaluator Card</h1>

<p align="center">
  <strong>Official HumanML3D / T2M leaderboard metric view.</strong>
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html">Paper</a> |
  <a href="https://ericguo5513.github.io/text-to-motion/">Project Page</a> |
  <a href="https://github.com/EricGuo5513/text-to-motion">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-evaluator-humanml3d-official">Motius Checkpoint</a>
</p>

HumanML3D Official is the native metric view for methods that generate
HumanML3D-263 features. Motius follows the selected-caption HumanML3D test
protocol for public model-card reporting.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | HumanML3D Official |
| Architecture | BiGRU text encoder + movement encoder + BiGRU motion encoder |
| Motion representation | HumanML3D-263 at 20 fps |
| Caption protocol | Selected caption for the HumanML3D test split |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint | [ZeyuLing/motius-evaluator-humanml3d-official](https://huggingface.co/ZeyuLing/motius-evaluator-humanml3d-official) |
| Artifact format | Safetensors + HumanML3D stats + `our_vab` GloVe lookup |

## Provenance

This evaluator is reproduced from **Generating Diverse and Natural 3D Human
Motions From Text** and its official
[`EricGuo5513/text-to-motion`](https://github.com/EricGuo5513/text-to-motion)
repository. The Hugging Face artifact is a lossless inference-only conversion
of the released `text_mot_match/model/finest.tar`: Motius preserves the movement,
text, and motion encoder tensors and removes optimizer state. It is not retrained.

## Download

```python
from huggingface_hub import snapshot_download

checkpoint_dir = snapshot_download(
    repo_id="ZeyuLing/motius-evaluator-humanml3d-official"
)
```

The downloaded directory contains `model.safetensors`, `config.json`,
`preprocessor_config.json`, normalization statistics, word-vector files, and an
SHA256 manifest.

## Reporting Rule

Every T2M model card should include this row. If a method is not evaluated in
native HumanML3D-263 space yet, the row should be marked `Pending`, not replaced
by a different evaluator.

## Notes

For FID and MM-Dist, lower is better. For R-Precision and Diversity, higher is
usually better, but Diversity should be compared together with the ground-truth
row and the target dataset protocol.
