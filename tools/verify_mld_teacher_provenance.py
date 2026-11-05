"""Check published T5 MLD tensors against the author's MotionLCM V1 archive."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "teacher", "artifact", "upstream", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file

    entry = "experiments_t2m/mld_humanml/mld_humanml_v1.ckpt"
    with zipfile.ZipFile(args.archive) as archive:
        digest = hashlib.sha256(archive.read(entry)).hexdigest()
    assert hashlib.sha256(args.teacher.read_bytes()).hexdigest() == digest
    state = torch.load(args.teacher, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    counts = {}
    for name in ["vae", "denoiser"]:
        native = load_file(str(args.artifact / (name + ".safetensors")))
        original = {
            k[len(name) + 1 :]: v for k, v in state.items() if k.startswith(name + ".")
        }
        assert native.keys() == original.keys()
        assert all(torch.equal(native[key], original[key]) for key in native), name
        counts[name] = len(native)
    report = dict(
        complete=True,
        scope="Every published Motius MLD VAE/denoiser tensor versus author MotionLCM V1 teacher extracted from the author's current prepare/download_pretrained_models.sh Google Drive archive. Establishes this artifact's neural provenance; text weights, supplied statistics, official CLIP MLD and full corpus training are separate certificates.",
        download_url="https://drive.google.com/uc?id=1U7homKobR2gaDLfL5flS3N0g7e0a_AQd",
        download_script_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        download_script_sha256=hashlib.sha256(
            (args.upstream / "prepare/download_pretrained_models.sh").read_bytes()
        ).hexdigest(),
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        archive_entry=entry,
        teacher_sha256=digest,
        exact_neural_tensors=counts,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
