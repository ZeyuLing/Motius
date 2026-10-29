# MotionStreamer native training and inference

The Bundle contains the causal TAE, autoregressive transformer, diffusion MLP
and frozen SentenceT5 network. Tokenization and batching use the data processor;
the Pipeline owns autoregressive generation, CFG and diffusion scheduling.
`MotionStreamerTrainer` implements TAE reconstruction and the original
two-forward T2M/streaming objectives through the shared AccelerateRunner.

Install `pip install -e '.[motionstreamer]'`. Native execution uses Transformers;
SentenceTransformers is needed only for the independent original-code checks.
The validated text path uses FP32, matching the original demo constructor even
when the text checkpoint is stored in FP16.

## Training

Set `MOTIUS_MS_ARTIFACT` to a complete artifact, `MOTIUS_MS_HML_ROOT` to the
processed HumanML3D-272 root, and `MOTIUS_MS_LATENTS` to the original TAE latent
cache. The corresponding root layout contains `motion_data`, `texts`, `split`
and `mean_std`. Streaming training additionally uses `MOTIUS_MS_BABEL_ROOT`
and `MOTIUS_MS_STREAM_ROOT`, with BABEL stream text and cached prefix lengths.

```bash
export MOTIUS_MS_ARTIFACT=/path/to/motionstreamer_artifact
export MOTIUS_MS_HML_ROOT=/path/to/humanml3d_272
export MOTIUS_MS_LATENTS=/path/to/t2m_latents
MOTIUS_MS_STAGE=ar python tools/train.py configs/motionstreamer/train_motionstreamer.py
```

`MOTIUS_MS_STAGE` selects `tae`, `ar` or `streaming`. Each stage initializes its
trained neural root from scratch. The recipe uses the published architecture
sizes; the artifact supplies statistics and frozen components. Use the corresponding joint TAE/statistics
and latent cache for streaming training. The recipe preserves the published
AdamW settings, AR two-forward objective, fourfold diffusion-head batch repeat,
and learning-rate clocks. AR runs include the original final iteration. TAE
runs use the original exclusive warmup range, followed by the milestone clock.

For multiple GPUs, use `accelerate launch --multi_gpu --num_processes 2` before
the same `tools/train.py` command. Static-graph DDP handles the original detached
attention-scale parameters and the diffusion MLP used from the Trainer.
Learning-rate scheduling advances once per optimizer update. `--auto-resume`
restores the complete native state.

Export the actual trained checkpoint with:

```bash
python tools/export_motionstreamer.py \
  --checkpoint /path/to/checkpoint-iter_100001 \
  --artifact "$MOTIUS_MS_ARTIFACT" --output /path/to/exported
```

The export includes current text, AR and TAE tensors and tokenizer metadata.
`--exclude-text` produces an AR/TAE component artifact; load it with
`bundle_kwargs={"load_text_model": False}` when explicitly inspecting components.

## Numerical evidence

- [Original released weights, generation, losses and gradient hashes](../architecture/motionstreamer-source.json).
- [Real SentenceT5 raw text and complete T2M](../architecture/motionstreamer-text-source.json).
- [Dataset sampling, padding, normalization and random-state checks](../architecture/motionstreamer-data-source.json).
- [Original optimizer/LR trajectories and native resume](../architecture/motionstreamer-schedule.json).
- [Modified complete artifact and offline cold public loading](../architecture/motionstreamer-checkpoint.json).
- [Two-GPU native AR training and 187 synchronized parameter tensors](../architecture/motionstreamer-distributed.json).
- [Actual shared training CLI, resume and current-weight export](../architecture/motionstreamer-native-checkpoint.json).

These records cover the stated artifacts and controlled cases. They establish
component equivalence, without claiming full-dataset convergence or benchmark
metrics. The upstream continuation helper reads an uninitialized
`use_out_proj` attribute; the native model exposes that mode explicitly.
The original T2M launcher also passes conflicting positional/keyword dataset
arguments. Native configuration supplies explicit dataset paths, while source
comparisons execute the original network, data-reader and objective code.
