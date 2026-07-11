# Getting Started

## Install

Create an environment with Python 3.9 or newer, then install the repository in
editable mode:

```bash
python -m pip install -e ".[dev]"
```

## Smoke Test

Verify that the core package imports and registers its framework modules:

```bash
python - <<'PY'
import motius

motius.register_all_modules()
print("Motius core import OK")
PY
```

Run the test suite:

```bash
pytest -q
```

## Train Entry Point

The standard training command is:

```bash
python tools/train.py path/to/config.py --work-dir outputs/my_experiment
```

Distributed jobs can be launched through Accelerate:

```bash
accelerate launch tools/train.py path/to/config.py --work-dir outputs/my_experiment
```

Use `--auto-resume` when training should resume from the latest checkpoint in
the configured work directory.

## Output Location

Place generated files under `outputs/`, including:

- checkpoints
- logs
- evaluation tables
- visualizations
- temporary exported artifacts

Keeping generated files in `outputs/` makes the repository root stable and
keeps public commits reviewable.
