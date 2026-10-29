"""Verify complete current-weight MotionStreamer export and cold public loading."""

from pathlib import Path
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(pipeline):
    import torch
    from verify_prompthmr_video_parity import tensors

    torch.manual_seed(71)
    torch.cuda.manual_seed_all(71)
    features = pipeline.text_frontend.encode(["a person walks forward", ""])
    motion = pipeline.infer_t2m(["a person walks forward"], [12])[0]
    return tensors(dict(features=features, motion=motion))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args()
    args.artifact, args.output_dir = args.artifact.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from motius import Pipeline
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import compare_full

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    source_text = args.output_dir / "source-text"
    exported = args.output_dir / "exported"
    if args.cold:
        assert not source_text.exists()
        pipeline = Pipeline.from_pretrained(exported, device="cuda")
        output = run(pipeline)
        output["weights"] = hashes(pipeline.bundle)
        assert not any(
            name == "models" or name.startswith("models.") for name in sys.modules
        )
        torch.save(output, args.output_dir / "cold.pt")
        return
    from motius.models.motionstreamer import MotionStreamerBundle
    from motius.pipelines.motionstreamer import MotionStreamerPipeline

    if not source_text.exists():
        shutil.copytree(
            args.artifact / "text_encoder/sentence-t5-xxl",
            source_text,
            copy_function=os.link,
        )
    pipeline = MotionStreamerPipeline(
        MotionStreamerBundle.from_pretrained(
            args.artifact, text_model_name=str(source_text), device="cuda"
        )
    )
    baseline = run(pipeline)
    with torch.no_grad():
        pipeline.bundle.text_model.projection.weight[0, 0].add_(0.125)
        pipeline.bundle.ar.out_proj.bias.add_(0.03125)
        list(pipeline.bundle.tae.parameters())[-1].view(-1)[0].add_(0.0078125)
    expected = run(pipeline)
    assert not torch.equal(baseline["features"], expected["features"])
    assert not torch.equal(baseline["motion"], expected["motion"])
    expected["weights"] = hashes(pipeline.bundle)
    pipeline.save_pretrained(exported)
    assert not list(exported.rglob("*.py"))
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    hidden = source_text.with_name("source-text-hidden")
    assert not hidden.exists()
    source_text.rename(hidden)
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--artifact",
                str(args.artifact),
                "--output-dir",
                str(args.output_dir),
                "--cold",
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
        )
    finally:
        hidden.rename(source_text)
    actual = torch.load(args.output_dir / "cold.pt", weights_only=True)
    assert expected.pop("weights") == actual.pop("weights")
    report = dict(
        complete=False,
        scope="Complete native MotionStreamer current-weight artifact: modified actual SentenceT5 projection, autoregressive output bias and TAE decoder bias, every neural tensor SHA256 plus text features and generated motion checked after cold public loading with original text directory hidden and HF offline. Separate certificates establish original-source numerical equivalence.",
        tests=[],
    )
    compare_full(expected, actual, "", report["tests"])
    report.update(complete=True, all_current_neural_tensors_sha256_equal=True)
    (args.output_dir / "source_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Complete current-weight MotionStreamer cold export exact")


if __name__ == "__main__":
    main()
