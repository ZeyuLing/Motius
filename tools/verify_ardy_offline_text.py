"""Verify a portable current-weight ARDY/LLM2Vec artifact without source paths."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_prompthmr_video_parity import tensors, compare_full


def hash_state(bundle):
    return {
        name: hashlib.sha256(
            value.detach()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()
        for name, value in bundle.state_dict().items()
    }


def run(pipeline):
    pipeline._ensure_text_encoder()
    with torch.no_grad():
        features, lengths = pipeline.model.text_encoder(["a person walks forward"])
    output = pipeline.generate(
        "a person walks forward",
        52,
        num_denoising_steps=10,
        seed=71,
        return_numpy=False,
    )
    return tensors(
        dict(features=features, lengths=torch.as_tensor(lengths), output=output)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-encoders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    artifact = args.output_dir / "artifact"
    if args.cold:
        assert (
            not args.text_encoders.exists()
        ), "Original text directory must be absent in this process"
        from motius import Pipeline

        pipeline = Pipeline.from_pretrained(str(artifact))
        result = run(pipeline)
        result["weights"] = hash_state(pipeline.bundle)
        assert not any(
            name == "ardy" or name.startswith("ardy.") for name in sys.modules
        )
        torch.save(result, args.output_dir / "cold.pt")
        return
    from motius.models.ardy import ARDYBundle
    from motius.pipelines.ardy import ARDYPipeline

    bundle = ARDYBundle(
        checkpoint_path=str(args.checkpoint),
        text_encoders_dir=str(args.text_encoders),
        device="cuda",
    )
    pipeline = ARDYPipeline(bundle)
    before = run(pipeline)
    with torch.no_grad():
        bundle.text_model.base_model.model.norm.weight[0].add_(0.03125)
    expected = run(pipeline)
    assert not torch.equal(before["features"], expected["features"])
    expected["weights"] = hash_state(bundle)
    pipeline.save_pretrained(artifact, include_text_encoder=True, copy_mode="hardlink")
    assert not list(
        (artifact / "text_encoders").rglob("*.py")
    ), "Portable artifact must not copy source applications"
    # Cold loading must succeed with the original text path unavailable.
    hidden = args.text_encoders.with_name(
        args.text_encoders.name + "-hidden-for-cold-check"
    )
    assert not hidden.exists()
    args.text_encoders.rename(hidden)
    try:
        subprocess.run(
            [
                sys.executable,
                "-u",
                __file__,
                "--checkpoint",
                str(args.checkpoint),
                "--text-encoders",
                str(args.text_encoders),
                "--output-dir",
                str(args.output_dir),
                "--cold",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
        )
    finally:
        hidden.rename(args.text_encoders)
    actual = torch.load(args.output_dir / "cold.pt", weights_only=True)
    assert expected.pop("weights") == actual.pop(
        "weights"
    ), "All current NN tensor hashes must match"
    report = dict(
        complete=False,
        scope="Actual 8B LLM2Vec/Llama-3 backbone and supervised adapter, deliberately modified final normalization weight, portable native NN/config/tokenizer export and cold Auto Pipeline reconstruction with original text directory absent. Full raw-prompt generation outputs exact; original-source raw-text certificate is separate.",
        tests=[],
    )
    compare_full(expected, actual, "", report["tests"])
    report.update(complete=True, all_current_neural_tensors_sha256_equal=True)
    (args.output_dir / "source_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Portable real-text artifact output comparisons:", len(report["tests"]))


if __name__ == "__main__":
    main()
