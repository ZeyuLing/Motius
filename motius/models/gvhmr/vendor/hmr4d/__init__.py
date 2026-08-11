import os
import sys
from pathlib import Path

sys.modules["hmr4d"] = sys.modules[__name__]

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path(
    os.environ.get(
        "MOTIUS_GVHMR_ARTIFACT_ROOT",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()
BODY_MODELS_ROOT = Path(
    os.environ.get(
        "MOTIUS_GVHMR_BODY_MODELS_ROOT",
        PROJ_ROOT / "inputs/checkpoints/body_models",
    )
).expanduser().resolve()


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)
