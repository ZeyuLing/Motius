# SMPL-Family Body Models

Motius uses the public SMPL body joints as its human-motion bridge, but the
official parameter files have separate licenses and cannot be redistributed in
this repository. Download the models you need from their official pages:

- [SMPL](https://smpl.is.tue.mpg.de/download.php)
- [SMPL+H and MANO](https://mano.is.tue.mpg.de/download.php)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php)

After accepting the relevant license, extract files into this layout:

```text
checkpoints/body_models/
├── smpl/
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smplh/
│   ├── SMPLH_FEMALE.pkl
│   └── SMPLH_MALE.pkl
└── smplx/
    ├── SMPLX_FEMALE.npz
    ├── SMPLX_MALE.npz
    └── SMPLX_NEUTRAL.npz
```

Motius also accepts the standard uppercase `.pkl` or `.npz` names directly
under each model-type directory, as well as `smplh/female/model.npz` and
`smplh/male/model.npz`. The SMPL-H files used by DART include the MANO hand PCA
parameters; these can be merged using the official
[SMPL-X conversion tool](https://github.com/vchoutas/smplx/blob/main/tools/merge_smplh_mano.py).
Pass this directory root or a model file:

```python
from motius.motion.skeleton import resolve_smpl_model_path

model = resolve_smpl_model_path(
    "checkpoints/body_models",
    model_type="smplh",
    gender="female",
)
print(model)
```

The command must print the file you downloaded. Use the source motion's actual
`gender` and `betas` whenever they are available.

The locally installed SMPL-H pair is accompanied by `smplh_manifest.json`,
which records each file's SHA-256 checksum, preparation and source, plus the
original `LICENSE_SMPLH.txt` and `LICENSE_MANO.txt`. These local assets are
ignored by Git and are not included in the Motius wheel.

The SMPL-H pair, original licenses and checksum manifest are backed up in
the private [ZeyuLing/body_models](https://huggingface.co/ZeyuLing/body_models)
repository. Restore into this directory after logging in with an account
that has access to the backup:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZeyuLing/body_models",
    revision="eca860440a2d3535050bfb526e1b688af2055203",
    local_dir="checkpoints/body_models",
    allow_patterns=["smplh/*", "smplh_manifest.json", "LICENSE_*.txt"],
)
```

The [official MANO/SMPL-H license](https://mano.is.tue.mpg.de/license.html)
restricts distribution to third parties. Access to a private backup does not
grant additional model rights; public redistribution requires the licensor's
prior written permission.
