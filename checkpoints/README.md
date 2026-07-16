# Checkpoints And Runtime Assets

This directory is part of the public Motius repository. It keeps stable local
paths and setup instructions for runtime assets that cannot all be bundled in
the Python package.

| Directory | Contents | How it is provided |
| --------- | -------- | ------------------ |
| `body_models/` | SMPL, SMPL-H, and SMPL-X parameter files | Download manually under the original license |
| `characters/` | User-provided rigged characters, including Mixamo FBX files | Download manually under the source license |
| `models/` | Optional local snapshots of Model Zoo artifacts | Prefer `Pipeline.from_pretrained(...)` |
| `evaluators/` | Optional local snapshots of Evaluator Zoo artifacts | Prefer evaluator `from_pretrained(...)` |
| `fbxsdk/` | Optional Autodesk FBX SDK Python runtime | Install the licensed wheel locally |

Small, redistributable support files may be committed here. Large model
weights, caches, and license-controlled assets remain ignored; their target
directory and download procedure must be documented in the nearest README.
When adding a new artifact, commit only files needed by users and never commit
Hugging Face caches such as `.cache/`, `hub/`, or `xet/`.

Generated motion, FBX, video, and evaluation artifacts belong under
`outputs/`, not under `checkpoints/`.

## Environment Variables

The following variables can redirect common local assets:

```bash
export MOTIUS_BODY_MODEL_DIR="$PWD/checkpoints/body_models"
export HF_HOME="$PWD/checkpoints/.hf_cache"
export MOTIUS_FBXSDK_PYTHON="$(command -v python3.10)"
export MOTIUS_FBXSDK_PYTHONPATH="$PWD/checkpoints/fbxsdk/cp310"
```

`HF_HOME` is optional. Its contents are local cache data and are not committed.
