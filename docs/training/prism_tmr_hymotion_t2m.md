# PRISM, TMR, and HYMotion T2M Training

Motius releases the training entry points for PRISM, TMR, and HYMotion T2M.
The recipes deliberately contain no dataset download locations or private
storage layout. Bring data you are licensed to use and expose it through the
neutral formats below.

## Text-motion manifest

PRISM and HYMotion T2M use `ManifestTextMotionDataset`. Set
`MOTIUS_DATA_ROOT` to a local directory and `MOTIUS_TRAIN_MANIFEST` to a JSON or
JSONL manifest. Relative motion and feature paths are resolved under the data
root.

```json
{
  "samples": [
    {
      "motion": "motions/000001.npy",
      "caption": ["A person walks forward.", "Someone takes several steps."],
      "text_features": "text_features/000001.npz"
    }
  ]
}
```

Motion arrays have shape `[frames, features]`. PRISM expects its native 138D
body representation; HYMotion T2M expects the released 201D representation.
The loader randomly crops long training samples and pads short samples to the
configured canvas while preserving `num_frames`/`tgt_length`, so padded frames
are excluded by both trainers.

PRISM can encode captions online. Cached PRISM feature files may instead store
`t5_text_embeds` and `t5_text_mask`; when prompt dropout is enabled, set
`MOTIUS_NULL_TEXT_FEATURE` to the cached empty-prompt feature. HYMotion T2M's
public recipe uses cached files containing `text_vec_raw`, `text_ctxt_raw`, and
`text_ctxt_raw_length`.

## TMR materialized format

The TMR recipe follows the public TMR materialized layout:

```text
data/training/tmr/
  annotations.json
  splits/train.txt
  motions/*.npy
  stats/mean.pt
  stats/std.pt
  token_embeddings/<encoder>.npy
  token_embeddings/<encoder>_slice.npy
  token_embeddings/<encoder>_index.json
  sent_embeddings/<encoder>.npy
  sent_embeddings/<encoder>_index.json
```

`annotations.json` maps sample IDs to a motion path and one or more timed text
annotations. The split file contains one sample ID per line.

## Launch

```bash
# PRISM: frozen fp32 VAE and text encoder, bf16 transformer
MOTIUS_DATA_ROOT=/path/to/training-data \
MOTIUS_TRAIN_MANIFEST=train.json \
bash tools/dist_train.sh configs/prism/train_prism.py 8

# TMR: fp32 training by default
MOTIUS_DATA_ROOT=/path/to/tmr-data \
bash tools/dist_train.sh configs/tmr/train_tmr_smpl22.py 8

# HYMotion T2M: cached text features and optional public warm start
MOTIUS_DATA_ROOT=/path/to/training-data \
MOTIUS_TRAIN_MANIFEST=train.json \
MOTIUS_MOTION_STATS=/path/to/stats \
MOTIUS_PRETRAINED_WEIGHTS=/path/to/motion_transformer.safetensors \
bash tools/dist_train.sh configs/hymotion_t2m/train_hymotion_t2m.py 8
```

Use `MOTIUS_WORK_DIR` to redirect checkpoints. Batch size, worker count, and
epoch count can be changed with MMEngine `--cfg-options`. PRISM keeps VAE
encoding in fp32 even when the transformer runs under bf16 mixed precision.

## Privacy audit

Run the release audit before publishing changes to these recipes:

```bash
python tools/audit_training_release.py
```

The audit rejects internal filesystem roots, cluster identifiers, credentials,
private data-source labels, and remote data URLs in the released training
surface.
