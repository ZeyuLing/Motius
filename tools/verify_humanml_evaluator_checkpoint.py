"""Public HumanML3D matching inference with current weights and cold exports."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import numpy as np
    import torch
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors
    from motius.pipelines.humanml3d_matching import HumanML3DMatchingPipeline

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    serialization, action = args.worker.split("-")
    exported = args.output_dir / ("exported_" + serialization)
    path = args.artifact if action == "export" else exported
    pipeline = HumanML3DMatchingPipeline.from_pretrained(str(path))
    bundle = pipeline.bundle
    names = ["movement_encoder", "text_encoder", "motion_encoder"]
    if action == "export":
        with torch.no_grad():
            bundle.movement_encoder.out_net.bias[0].add_(0.0625)
            bundle.text_encoder.output_net[-1].bias[0].add_(0.125)
            bundle.motion_encoder.output_net[-1].bias[0].add_(0.03125)
            bundle.mean[0].add_(1e-8)
            bundle.std[-1].add_(1e-10)
        bundle.max_text_length = 17
        bundle.max_motion_length = 192
    captions = ["A Person walks forward.", "", "the person walks " * 20]
    motions = [
        np.sin(np.arange(length * 263).reshape(length, 263) * 0.017) * 0.3
        for length in [40, 80, 196]
    ]
    values = tensors(
        dict(
            neural={name: hashes(getattr(bundle, name)) for name in names},
            mean=bundle.mean,
            std=bundle.std,
            properties={
                name: getattr(bundle, name)
                for name in [
                    "spacy_model",
                    "max_text_length",
                    "max_motion_length",
                    "unit_length",
                ]
            },
            text=pipeline.encode_texts(captions),
            motion=pipeline.encode_motions(motions),
        )
    )
    torch.save(values, args.output_dir / (args.worker + ".pt"))
    if action == "export":
        bundle.save_pretrained(exported, safe_serialization=serialization == "safe")
    assert not any(
        name in sys.modules
        for name in [
            "networks.evaluator_wrapper",
            "data.dataset",
            "utils.word_vectorizer",
        ]
    )
    print(args.worker, "public matching outputs saved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--worker", choices=["safe-export", "safe-cold", "torch-export", "torch-cold"]
    )
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        scope="Actual released HumanML3D matching encoders with modified current weights, sub-FP32 statistics changes and non-default text/motion limits. Public text/ragged physical inference and every neural hash/statistic/config value match through offline fresh-process safetensors AND PyTorch artifact loading. Source parity is separately certified; this is current-weight export evidence, not training/convergence.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for serialization in ["safe", "torch"]:
            for action in ["export", "cold"]:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--artifact",
                        str(args.artifact),
                        "--output-dir",
                        str(args.output_dir),
                        "--worker",
                        serialization + "-" + action,
                    ],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    },
                    check=True,
                )
            compare_full(
                torch.load(
                    args.output_dir / (serialization + "-export.pt"), weights_only=True
                ),
                torch.load(
                    args.output_dir / (serialization + "-cold.pt"), weights_only=True
                ),
                "/" + serialization,
                report["tests"],
            )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Current HumanML3D evaluator safetensors/PyTorch cold exports exact", flush=True
    )


if __name__ == "__main__":
    main()
