# Getting Started

Motius requires Python 3.10 or newer. Clone the repository with Git LFS
enabled so the robot meshes and simulator assets are materialized:

```bash
git lfs install
git clone https://github.com/ZeyuLing/Motius.git
cd Motius
git lfs pull
python -m pip install -e ".[dev]"
```

## Verify the core framework

Verify that the core package imports and registers its framework modules:

```bash
python - <<'PY'
import motius

motius.register_all_modules()
print("Motius core import OK")
PY
```

Run the framework and public-artifact smoke tests:

```bash
python -m pytest -q \
  tests/test_pipeline_pretrained.py \
  tests/test_motion_tracking_artifacts.py
```

Inspect the complete motion-tracking reproduction gate, which accepts explicit
released-artifact, replay, viewer-manifest, and trainer-evidence paths:

```bash
python tools/verify_motion_tracking_reproduction.py --help
```

The release invocation uses `--require-complete`; see the
[Motion Tracking validation contract](tasks/motion_tracking.md#validation).

## Train a model

The standard training command writes its artifacts below `outputs/`:

```bash
python tools/train.py path/to/config.py --work-dir outputs/my_experiment
```

Distributed jobs can be launched through Accelerate:

```bash
accelerate launch tools/train.py path/to/config.py \
  --work-dir outputs/my_experiment
```

Use `--auto-resume` to continue from the latest checkpoint in the configured
work directory.

Simulator-native motion-tracking training requires a matching Isaac Lab
installation. Install the method-specific dependencies and follow the selected
Model Card:

```bash
python -m pip install -e ".[motion-tracking-train-sonic]"
python -m pip install -e ".[motion-tracking-train-protomotions]"
python -m pip install -e ".[motion-tracking-train-beyondmimic]"
```

- [SONIC](model_zoo/sonic.md)
- [ProtoMotions](model_zoo/protomotions.md)
- [BeyondMimic](model_zoo/beyondmimic.md)
- [Motion Tracking task and validation contract](tasks/motion_tracking.md)

All generated files belong under `outputs/`. Checkpoints and manually
downloaded assets belong under `checkpoints/`.
