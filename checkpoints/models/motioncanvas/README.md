# MotionCanvas Checkpoint

The complete release is hosted at
[`ZeyuLing/Motius-MotionCanvas-0.46B`](https://huggingface.co/ZeyuLing/Motius-MotionCanvas-0.46B).
It includes the motion transformer, learned M2M condition parameters,
MotionCanvas-198 statistics, SMPL-22 rest offsets, Qwen3, and CLIP.

`Mean.npy`, `Std.npy`, `normalization_provenance.md`, and
`bone_offsets_22.pt` are deliberately tracked here because they are small and
required by the public training recipe. The Hub artifact contains the same
tensors for inference.

No manual download is required:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionCanvas-0.46B",
    cache_dir="/path/to/large/huggingface-cache",
    bundle_kwargs={"device": "cuda", "text_dtype": "bf16"},
)
```

The complete artifact is about 20 GB. Pass the top-level `cache_dir` argument
or set `HF_HOME`; placing `cache_dir` inside `bundle_kwargs` does not control
the initial snapshot download. When using Hugging Face Xet on a small system
disk, also point `HF_XET_CACHE` at the same large filesystem.
