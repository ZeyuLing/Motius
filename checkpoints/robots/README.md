# Robot Assets

Runtime robot descriptions live here when their upstream distribution terms or
size make them unsuitable for Git.

## BeyondMimic Unitree G1

BeyondMimic uses the exact `unitree_description` archive named by the official
training repository. Download and verify it with:

```bash
python tools/download_beyondmimic_assets.py
```

The command verifies SHA256
`b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674`
and extracts only:

```text
checkpoints/robots/unitree_description/
  urdf/g1/
  meshes/g1/
```

To keep the assets elsewhere, pass `--output PATH` to the downloader and
`--asset-root PATH` to the BeyondMimic trainer, or set
`MOTIUS_BEYONDMIMIC_ASSET_ROOT`.
