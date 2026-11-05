"""Verify actual current-weight ARDY exports and cold Auto Pipeline inference."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_prompthmr_video_parity import tensors, compare_full


def predict(pipeline, features):
    return pipeline.generate(
        "prepared text",
        pipeline.bundle.horizon + 8,
        text_feat=features,
        num_denoising_steps=10,
        seed=91,
        return_numpy=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cold:
        from motius import Pipeline

        pipeline = Pipeline.from_pretrained(str(args.cold))
        features = torch.load(args.output_dir / "features.pt", weights_only=True).cuda()
        result = predict(pipeline, features)
        assert not any(
            name == "ardy" or name.startswith("ardy.") for name in sys.modules
        )
        torch.save(
            tensors(dict(output=result, weights=pipeline.bundle.state_dict())),
            args.output_dir / "cold.pt",
        )
        return
    from motius.models.ardy import ARDYBundle
    from motius.pipelines.ardy import ARDYPipeline

    report = dict(
        complete=False,
        scope="Real released Core-H40 and G1-H52 architectures, deliberate changes to both denoiser stages and tokenizer decoder, current-weight save and fresh Auto Pipeline reload. Complete public two-window inference outputs, including decoded geometry and G1 qpos. Not an independent original-source checkpoint comparison.",
        tests=[],
    )
    report_path = args.output_dir / "checkpoint_parity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    torch.manual_seed(50)
    features = torch.randn(1, 1, 4096, device="cuda") * 0.1
    torch.save(features.cpu(), args.output_dir / "features.pt")
    for name in ["ARDY-Core-RP-20FPS-Horizon40", "ARDY-G1-RP-25FPS-Horizon52"]:
        pipeline = ARDYPipeline(
            ARDYBundle(
                checkpoint_path=str(args.artifacts / name),
                device="cuda",
                text_encoder=False,
            )
        )
        before = predict(pipeline, features)
        with torch.no_grad():
            pipeline.bundle.denoiser.root_model.output_linear.bias.add_(0.01)
            pipeline.bundle.denoiser.body_model.output_linear.bias.add_(0.02)
            pipeline.bundle.autoencoder.decoder.output_proj.bias.add_(0.005)
        expected = predict(pipeline, features)
        assert not torch.equal(before["features"], expected["features"]), name
        artifact = args.output_dir / name
        pipeline.save_pretrained(artifact)
        (args.output_dir / "cold.pt").unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                __file__,
                "--artifacts",
                str(args.artifacts),
                "--output-dir",
                str(args.output_dir),
                "--cold",
                str(artifact),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
        )
        cold = torch.load(args.output_dir / "cold.pt", weights_only=True)
        compare_full(
            tensors(dict(output=expected, weights=pipeline.bundle.state_dict())),
            cold,
            name,
            report["tests"],
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        del pipeline, before, expected, cold
        torch.cuda.empty_cache()
    report["complete"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Current-weight cold export comparisons:", len(report["tests"]))


if __name__ == "__main__":
    main()
