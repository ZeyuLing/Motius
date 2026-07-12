<h1 align="center">MotionStreamer Evaluator Card</h1>

<p align="center">
  <strong>MotionStreamer-272 text-motion evaluator for SMPL-aligned T2M results.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2503.15451">Paper</a> |
  <a href="https://zju3dv.github.io/MotionStreamer/">Project Page</a> |
  <a href="https://github.com/zju3dv/MotionStreamer">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-evaluator-motionstreamer-272">Motius Checkpoint</a>
</p>

MotionStreamer Evaluator is the second public metric view in Motius model
cards. It evaluates motions through the MotionStreamer-272 representation after
the method output has been converted through the checked SMPL/MotionStreamer
path.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | MotionStreamer Evaluator |
| Architecture | DistilBERT text encoder + ACTOR motion encoder, latent dim 256 |
| Motion representation | MotionStreamer-272 at 30 fps |
| Caption protocol | HumanML3D selected-caption protocol unless a card states otherwise |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint | [ZeyuLing/motius-evaluator-motionstreamer-272](https://huggingface.co/ZeyuLing/motius-evaluator-motionstreamer-272) |
| Artifact format | Safetensors + MotionStreamer-272 stats + DistilBERT tokenizer |

## Provenance

This evaluator is reproduced from **MotionStreamer: Streaming Motion Generation
via Diffusion-based Autoregressive Model in Causal Latent Space** and the
official [`zju3dv/MotionStreamer`](https://github.com/zju3dv/MotionStreamer)
repository. The Hugging Face artifact losslessly extracts the released
`Evaluator_272/epoch=99.ckpt` text and motion encoders. Lightning trainer state
and unrelated modules are omitted; the evaluator weights are not retrained.

## Download

```python
from huggingface_hub import snapshot_download

checkpoint_dir = snapshot_download(
    repo_id="ZeyuLing/motius-evaluator-motionstreamer-272"
)
```

The downloaded directory contains `model.safetensors`, `config.json`,
`preprocessor_config.json`, normalization statistics, tokenizer files, and an
SHA256 manifest.

## Reporting Rule

Every T2M model card should include this row. HumanML3D-263, SMPL, SMPL-X, and
DART-style outputs must first go through a checked conversion path before this
metric is reported.

## Notes

This evaluator remains useful as a strong semantic metric, but it is not the
only public metric view. Motius reports it together with HumanML3D Official and
the Motius Joint-Position Evaluator.
