"""HumanML3D matching artifact metadata and non-neural inference resources."""

import json
from pathlib import Path
import shutil


def export_matching_resources(bundle, directory, weights_file):
    target = Path(directory)
    source = next(
        (
            bundle.artifact_dir / name
            for name in ["glove", "vocab"]
            if (bundle.artifact_dir / name).is_dir()
        ),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            f"Missing evaluator vocabulary in {bundle.artifact_dir}"
        )
    (target / "glove").mkdir(exist_ok=True)
    for name in ["our_vab_data.npy", "our_vab_words.pkl", "our_vab_idx.pkl"]:
        destination = target / "glove" / name
        if (source / name).resolve() != destination.resolve():
            shutil.copy2(source / name, destination)
    properties = {
        name: getattr(bundle, name)
        for name in [
            "spacy_model",
            "max_text_length",
            "max_motion_length",
            "unit_length",
        ]
    }
    (target / "matching_config.json").write_text(
        json.dumps(properties, indent=2) + "\n"
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "motius-humanml3d-263-evaluator",
                "embedding_dim": 512,
                "weights_file": weights_file,
                "format_version": 1,
            },
            indent=2,
        )
        + "\n"
    )
    notices = Path(__file__).resolve().parents[1] / "licenses" / "humanml3d_evaluator"
    for path in notices.iterdir():
        if path.is_file() and path.name != "__init__.py":
            shutil.copy2(path, target / path.name)
