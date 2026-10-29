"""Non-neural resources for the MotionStreamer matching encoders."""

import copy
import json
from pathlib import Path
import shutil


def export_resources(bundle, directory):
    target = Path(directory)
    preprocessor = copy.deepcopy(bundle.preprocessor_config)
    source = bundle.artifact_dir / preprocessor.get("tokenizer", "tokenizer")
    tokenizer = target / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    if source.resolve() != tokenizer.resolve():
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".json", ".txt", ".model"}:
                shutil.copy2(path, tokenizer / path.name)
    bundle.text_encoder.text_model.config.save_pretrained(tokenizer)
    preprocessor.update(
        mean="stats/mean.npy",
        std="stats/std.npy",
        tokenizer="tokenizer",
        max_motion_length=bundle.max_motion_length,
        unit_length=bundle.unit_length,
    )
    config = copy.deepcopy(bundle.evaluator_config)
    config.update(
        weights_file="model.safetensors",
        text_attn_implementation=bundle.text_encoder.text_model.config._attn_implementation,
        neural_dtype={
            name: str(next(getattr(bundle, name).parameters()).dtype).removeprefix(
                "torch."
            )
            for name in ["text_encoder", "motion_encoder"]
        },
    )
    for filename, data in [
        ("config.json", config),
        ("preprocessor_config.json", preprocessor),
    ]:
        (target / filename).write_text(json.dumps(data, indent=2) + "\n")
    notices = Path(__file__).resolve().parents[1] / "licenses/motionstreamer"
    for path in notices.iterdir():
        if path.is_file() and path.name != "__init__.py":
            shutil.copy2(path, target / path.name)
