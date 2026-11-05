"""Self-describing metadata for the GenTrack specialization of HYMotion."""

import json
from pathlib import Path


def specialize_artifact(directory):
    directory = Path(directory)
    classes = dict(
        pipeline_class="motius.pipelines.gentrack.GenTrackPipeline",
        bundle_class="motius.models.gentrack.GenTrackG1Bundle",
    )
    for filename in ["hymotion_t2m_config.json", "model_index.json"]:
        path = directory / filename
        metadata = json.loads(path.read_text())
        metadata.update(classes)
        metadata.update(
            _class_name="GenTrackPipeline",
            tasks=["text_to_motion"],
            specialization="gentrack_g1",
        )
        if "api" in metadata:
            metadata["api"] = {
                key: classes["bundle_class"] + "." + key
                for key in ["from_pretrained", "from_config"]
            }
        path.write_text(json.dumps(metadata, indent=2) + "\n")
