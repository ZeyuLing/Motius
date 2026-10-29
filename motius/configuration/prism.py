"""PRISM artifact identity and saved runtime defaults."""

import json
from pathlib import Path


def runtime_options(directory):
    config = json.loads((Path(directory) / "motius_config.json").read_text())
    allowed = {
        "transformer_dtype",
        "text_dtype",
        "default_kafs_mode",
        "latent_sample_method",
    }
    return {
        name: value
        for name, value in config.get("runtime_options", {}).items()
        if name in allowed
    }


def write_pipeline_index(directory):
    metadata = dict(
        _class_name="PRISMPipeline",
        _library_name="motius",
        model_type="prism",
        pipeline_class="motius.pipelines.prism.PRISMPipeline",
        bundle_class="motius.models.prism.PRISMBundle",
        tasks=[
            "text_to_motion",
            "temporal_motion_completion",
            "sequential_text_to_motion",
        ],
    )
    (Path(directory) / "model_index.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
