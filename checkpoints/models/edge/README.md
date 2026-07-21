# EDGE

The released EDGE artifact is loaded through:

```python
from motius.pipelines.edge import EDGEPipeline

pipeline = EDGEPipeline.from_pretrained(
    "ZeyuLing/Motius-EDGE-AISTPP",
    device="cuda",
)
```

Raw-audio inference additionally uses the frozen OpenAI Jukebox 5B frontend
through `jukemirlib`. Its two files live outside the 1.19 GB EDGE diffusion
checkpoint and are cached as:

| File | SHA-256 |
| ---- | ------- |
| `vqvae.pth.tar` | `69745413a48e887f8a3fe91b972a6f7f434021a1ce911a99187b331eb48c059a` |
| `prior_level_2.pth.tar` | `89a1dd14f5b2f9b16b3e73b53fa2138cc89fd96bb13249b4267fea471de92672` |

Pass their directory as `jukebox_cache_dir` to the pipeline. Omitting it uses
the standard `~/.cache/jukemirlib` cache and lets `jukemirlib` fetch the same
official OpenAI files. The recommended repository-local cache path is
`checkpoints/models/edge/jukebox_cache/`; its downloaded contents remain
ignored by Git.

For provenance checks, download the public mirror of the official
`checkpoint.pt` from
[`edge-dance-generation/EDGE`](https://huggingface.co/edge-dance-generation/EDGE)
and verify SHA256
`28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7`.
Convert it without importing the upstream checkout:

```bash
python tools/convert_edge_checkpoint.py \
  --checkpoint checkpoints/models/edge/official_checkpoint.pt \
  --output checkpoints/models/edge/hf_artifact
```

The source Google Drive link currently returns an access page instead of the
1.19 GB checkpoint; the mirror hash above is pinned so substitution is
detectable.
