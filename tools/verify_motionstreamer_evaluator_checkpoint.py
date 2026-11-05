"""Current matching encoders, Float64 statistics and tokenizer survive cold export."""

from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import numpy as np
    import torch
    from motius.pipelines.motionstreamer_matching import MotionStreamerMatchingPipeline
    from verify_motionstreamer_source import hashes

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    path = args.artifact if args.worker == "export" else args.output_dir / "exported"
    pipe = MotionStreamerMatchingPipeline.from_pretrained(
        str(path), bundle_kwargs={"device": "cuda"}
    )
    bundle = pipe.bundle
    if args.worker == "export":
        with torch.no_grad():
            bundle.motion_encoder.skel_embedding.bias[0].add_(0.0625)
            bundle.text_encoder.projection[-1].bias[0].add_(0.125)
            bundle.text_encoder.text_model.embeddings.word_embeddings.weight[
                101, 0
            ].add_(0.03125)
            bundle.mean[0].add_(1e-8)
            bundle.std[1].add_(1e-9)
        bundle.max_motion_length = 192
        bundle.unit_length = 8
    captions = ["A person walks.", "", "Someone turns and waves."]
    tokens = pipe.tokenizer(captions, return_tensors="pt", padding=True).to(bundle.device)
    with torch.no_grad():
        backbone = bundle.text_encoder.text_model(**tokens).last_hidden_state
    motions = [
        np.random.default_rng(81 + i).normal(size=(length, 272)).astype(np.float64)
        * 0.1
        for i, length in enumerate([20, 72, 300])
    ]
    values = dict(
        tokenizer_class=type(pipe.tokenizer).__name__,
        attention_class=type(bundle.text_encoder.text_model.transformer.layer[0].attention).__name__,
        tokens={name:value.cpu() for name,value in tokens.items()},
        backbone=backbone.cpu(),
        neural={
            name: hashes(getattr(bundle, name))
            for name in ["text_encoder", "motion_encoder"]
        },
        mean=bundle.mean.cpu(),
        std=bundle.std.cpu(),
        max_motion_length=bundle.max_motion_length,
        unit_length=bundle.unit_length,
        text=torch.from_numpy(pipe.encode_texts(captions)),
        motion=torch.from_numpy(pipe.encode_motions(motions)),
    )
    torch.save(values, args.output_dir / (args.worker + ".pt"))
    if args.worker == "export":
        pipe.save_pretrained(args.output_dir / "exported")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=["export", "cold"])
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
        scope="Actual released matching encoders with modified motion/text/BERT neural weights, sub-Float32 Float64 mean/std changes and nondefault max192/unit8 preprocessing. All neural hashes, statistics, raw-caption and ragged physical-motion embeddings match after offline fresh-process export/loading.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for kind in ["export", "cold"]:
            subprocess.run(
                [sys.executable, __file__, *sys.argv[1:], "--worker", kind],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "HF_HUB_OFFLINE": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
                check=True,
            )
        compare_full(
            torch.load(args.output_dir / "export.pt", weights_only=False),
            torch.load(args.output_dir / "cold.pt", weights_only=False),
            "",
            report["tests"],
        )
        report["complete"] = True
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print("MotionStreamer matching current checkpoint exact")


if __name__ == "__main__":
    main()
