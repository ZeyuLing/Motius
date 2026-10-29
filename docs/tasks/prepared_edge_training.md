# Native EDGE training

`configs/edge/prepared_dance.py` runs the native `EDGETrainer` through the shared
`AccelerateRunner`. The Trainer owns the original reconstruction, velocity, FK
and foot-skate objectives, Adan optimization and source-order EMA. Its Bundle
contains the dance decoder and, while training, a frozen EMA copy.

Prepare a JSON list or JSONL manifest containing `{"path":"window.npz"}` records.
Each NPZ supplies float tensors `motion [150,151]`, already normalized by the
selected checkpoint's range scaler, and `music_features [150,4800]`, using the
original 30-fps Jukebox feature contract. Paths are relative to the manifest.
Raw audio extraction, corpus slicing and fitting the original scaler happen
before this boundary.

```bash
export MOTIUS_EDGE_ARTIFACT=/path/to/edge/artifact
export MOTIUS_EDGE_MANIFEST=/path/to/prepared/train.jsonl
export MOTIUS_EDGE_WORK_DIR=work_dirs/edge_native
export MOTIUS_EDGE_BATCH_SIZE=32
export MOTIUS_EDGE_STEPS=100000

torchrun --standalone --nproc_per_node=2 tools/train.py configs/edge/prepared_dance.py
torchrun --standalone --nproc_per_node=2 tools/train.py configs/edge/prepared_dance.py --auto-resume
```

The entry point uses original Adan (`lr=4e-4`, `weight_decay=0.02`) and EMA
decay `0.9999` after every successful update. It uses accumulation one, controlled
FP32 and math attention. Both online and EMA neural roots and the complete
optimizer are saved for resume. This prepared-data integration does not claim
the original raw-corpus distribution, complete epoch/validation protocol or FP16.

`bundle.save_pretrained(directory)` exports the active online network;
`bundle.save_pretrained(directory, use_ema=True)` exports the current EMA network.
Both preserve the scaler and original notices and support offline loading with
`motius.Pipeline.from_pretrained(directory)`. The independent source certificate
pins PyTorch3D v0.7.2 rotation math because the original README leaves its version
unspecified. Native stitching/FK preserves the original axis-angle/quaternion
operation order; public joints apply the documented Z-up to Y-up coordinate
permutation. `edge_motion` and `motion_135` are additional unified representations.
