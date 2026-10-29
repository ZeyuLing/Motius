"""Export current neural tensors and selected tokenizer metadata."""

from pathlib import Path
import shutil


def copy_motion_licenses(directory):
    """Retain both original notices with the shared MLD/MotionLCM networks."""
    destination = Path(directory)
    source = Path(__file__).resolve().parents[1] / "licenses"
    files = []
    for method in ["mld", "motionlcm"]:
        target = destination / "licenses" / method
        target.mkdir(parents=True, exist_ok=True)
        for path in (source / method).iterdir():
            if path.is_file() and path.suffix != ".py":
                shutil.copy2(path, target / path.name)
                files.append((target / path.name).relative_to(destination).as_posix())
    return sorted(files)


def save_text_encoder(network, directory):
    if network.name == "t5":
        from motius.configuration.motionstreamer.text import save_text_network

        return save_text_network(network.text_model, network.modelpath, directory)
    destination = Path(directory)
    network.text_model.save_pretrained(destination, safe_serialization=True)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(network.modelpath).save_pretrained(destination)
