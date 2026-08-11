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
│   ├── female/model.npz
│   ├── male/model.npz
│   └── neutral/model.npz
└── smplx/
    ├── SMPLX_FEMALE.npz
    ├── SMPLX_MALE.npz
    └── SMPLX_NEUTRAL.npz
```

Motius also accepts the standard uppercase `.pkl` or `.npz` names directly
under each model-type directory. Pass this directory root or a model file:

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
