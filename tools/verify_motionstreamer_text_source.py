"""Compare native raw-text generation with original SentenceT5/MotionStreamer execution."""

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
    from safetensors.torch import load_file
    from verify_motionstreamer_source import digest
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    text_path = args.artifact / "text_encoder/sentence-t5-xxl"
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from sentence_transformers import SentenceTransformer
        from models.llama_model import LLaMAHF, LLaMAHFConfig
        from models.tae import Causal_HumanTAE

        # Actual original demo constructor: default FP32 despite FP16 storage.
        frontend = SentenceTransformer(str(text_path), device="cuda").eval()
        ar = (
            LLaMAHF(LLaMAHFConfig.from_name("Normal_size"), device=torch.device("cuda"))
            .cuda()
            .eval()
        )
        tae = (
            Causal_HumanTAE(
                hidden_size=1024,
                down_t=2,
                stride_t=2,
                depth=3,
                dilation_growth_rate=3,
                activation="relu",
                latent_dim=16,
                clip_range=[-30, 20],
            )
            .cuda()
            .eval()
        )
        ar.load_state_dict(
            load_file(str(args.artifact / "ar.safetensors")), strict=True
        )
        tae.load_state_dict(
            load_file(str(args.artifact / "tae.safetensors")), strict=True
        )
        weights = {}
        for name, value in frontend.state_dict().items():
            if name.startswith("0.auto_model."):
                name = "backbone." + name.removeprefix("0.auto_model.")
            elif name.startswith("2.linear."):
                name = "projection." + name.removeprefix("2.linear.")
            else:
                raise ValueError("Unexpected actual SentenceT5 neural state: " + name)
            weights[name] = digest(value)
    else:
        from motius.models.motionstreamer import MotionStreamerBundle
        from motius.pipelines.motionstreamer import MotionStreamerPipeline
        from verify_motionstreamer_source import hashes

        pipeline = MotionStreamerPipeline(
            MotionStreamerBundle.from_pretrained(args.artifact, device="cuda")
        )
        frontend = pipeline.text_frontend
        weights = hashes(pipeline.bundle.text_model)
    cases = [
        "a person walks forward",
        [
            "  a person turns around  ",
            "",
            "the person slowly raises both arms above the head",
        ],
    ]
    result = dict(
        weights=weights, features=[tensors(frontend.encode(case)) for case in cases]
    )
    torch.manual_seed(817)
    torch.cuda.manual_seed_all(817)
    with torch.no_grad():
        if args.worker == "upstream":
            latent = ar.sample_for_eval_CFG(
                [cases[0]],
                length=12,
                tokenize_model=frontend,
                device=torch.device("cuda"),
                cfg=4.0,
            )
            value = tae.forward_decoder(latent)[0]
            mean = torch.from_numpy(np.load(args.artifact / "Mean.npy")).to("cuda")
            std = torch.from_numpy(np.load(args.artifact / "Std.npy")).to("cuda")
            motion = (value * std + mean)[:12].cpu().numpy().astype(np.float32)
        else:
            motion = pipeline.infer_t2m([cases[0]], [12])[0]
    result["motion"] = tensors(motion)
    result["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.worker:
        worker(args)
        return
    report = dict(
        complete=False,
        scope="Actual original SentenceTransformer 3.2.1/SentenceT5-XXL default-FP32 construction versus native tensor-only text backbone/pooling/projection/normalization with separate tokenizer/batching processor. Single/ragged/empty/whitespace prompts and complete native public T2M versus original 50-step-per-token AR/TAE graph. Every text neural tensor SHA256 checked. Published model weights are verified separately against original HF LFS hashes.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--upstream",
                str(args.upstream),
                "--artifact",
                str(args.artifact),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            check=True,
        )
    import torch
    from verify_prompthmr_video_parity import compare_full

    try:
        compare_full(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
        report["complete"] = True
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Original/native real SentenceT5 neural weights, raw text and full generation exact"
    )


if __name__ == "__main__":
    main()
