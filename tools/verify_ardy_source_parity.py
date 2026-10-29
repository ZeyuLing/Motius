"""Independent pinned ARDY source/released-weight and full generation comparison."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_prompthmr_video_parity import tensors, compare_full


def worker(args):
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from ardy.model.load_model import load_model
    else:
        from motius.configuration.ardy.loader import load_model
    result = {}
    text_encoder = None
    if args.text_encoders is not None:
        os.environ["TEXT_ENCODERS_DIR"] = str(args.text_encoders)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for folder in ["ARDY-Core-RP-20FPS-Horizon40", "ARDY-G1-RP-25FPS-Horizon52"]:
        torch.manual_seed(826)
        native = args.worker == "native"
        if native:
            from motius.models.ardy import ARDYBundle
            from motius.pipelines.ardy import ARDYPipeline

            bundle = ARDYBundle(
                checkpoint_path=str(args.artifacts / folder),
                device="cuda",
                text_encoder=None if args.text_encoders else False,
                text_encoders_dir=(
                    str(args.text_encoders) if args.text_encoders else None
                ),
            )
            pipeline = ARDYPipeline(bundle, text_encoder=text_encoder)
            if args.text_encoders is not None:
                pipeline._ensure_text_encoder()
            model = pipeline.model
        else:
            model = load_model(
                folder,
                checkpoints_dir=str(args.artifacts),
                device="cuda",
                text_encoder=(
                    text_encoder
                    if text_encoder is not None
                    else (None if args.text_encoders else False)
                ),
                text_encoder_mode="local",
            )
        if args.text_encoders is not None:
            text_encoder = model.text_encoder
            result[folder + "/text_features"] = tensors(
                text_encoder(["a person walks forward", "a person raises both arms"])
            )
            if native:
                assert bundle.text_model is text_encoder.model.model
        # Compare every actual trained network tensor using source key names.
        result[folder + "/weights"] = tensors(
            dict(
                denoiser=(
                    model.denoiser.model.core if native else model.denoiser.model
                ).state_dict(),
                tokenizer=(
                    model.autoencoder.core if native else model.autoencoder
                ).state_dict(),
            )
        )
        for case, batch, length in [
            ("unconstrained", 1, model.gen_horizon_len + 12),
            ("constrained_ragged", 2, model.gen_horizon_len + 12),
        ]:
            torch.manual_seed(827)
            features = (
                None
                if args.text_encoders
                else torch.randn(batch, 1, 4096, device="cuda") * 0.1
            )
            text_mask = (
                None
                if args.text_encoders
                else torch.ones(batch, 1, dtype=torch.bool, device="cuda")
            )
            lengths = torch.full((batch,), length, device="cuda", dtype=torch.long)
            if batch > 1:
                lengths[-1] -= 7
            pad = torch.arange(length, device="cuda")[None] < lengths[:, None]
            motion_mask = torch.zeros(
                batch, length, model.motion_rep.motion_rep_dim, device="cuda"
            )
            observed = torch.zeros_like(motion_mask)
            if case.startswith("constrained"):
                rotations = (
                    torch.eye(3, device="cuda")
                    .reshape(1, 1, 3, 3)
                    .repeat(length, model.skeleton.nbjoints, 1, 1)
                )
                roots = torch.zeros(length, 3, device="cuda")
                roots[:, 0] = torch.arange(length, device="cuda") * 0.002
                physical = model.motion_rep(rotations, roots, to_normalize=True)
                observed[:] = physical[None]
                motion_mask[:, ::13, :5] = 1
                motion_mask[:, length // 2, -4:] = 1
            with torch.no_grad():
                output = model(
                    texts=["a person walks forward", "a person raises both arms"][
                        :batch
                    ],
                    num_frames=length,
                    num_denoising_steps=10,
                    pad_mask=pad,
                    first_heading_angle=torch.zeros(batch, device="cuda"),
                    motion_mask=motion_mask,
                    observed_motion=observed,
                    cfg_weight=(2.0, 1.5),
                    text_feat=features,
                    text_pad_mask=text_mask,
                    progress_bar=lambda values: values,
                )
            result[folder + "/" + case] = tensors(
                dict(
                    output=output,
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                )
            )
        del model
        torch.cuda.empty_cache()
    if args.worker == "native":
        assert not any(
            name == "ardy" or name.startswith("ardy.") for name in sys.modules
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifacts", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--text-encoders", type=Path)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            worker(args)
        except BaseException:
            import traceback

            (args.output_dir / (args.worker + "_error.txt")).write_text(
                traceback.format_exc()
            )
            raise
        return
    report = dict(
        complete=False,
        scope="Pinned original ARDY repository, released Core-H40 and G1-H52 weights and actual original full two-window generation. Synthetic precomputed 4096-D text tensors (not LLM text-encoder certification), unconstrained and two-sample ragged constrained inputs. Shared FP32 eager runtime; not original training protocol.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    if args.text_encoders is not None:
        report["scope"] = report["scope"].replace(
            "Synthetic precomputed 4096-D text tensors (not LLM text-encoder certification)",
            "Actual original 8B LLM2Vec Llama-3 tokenizer, bidirectional backbone, MNTP and supervised adapters; real text features and raw-prompt generation",
        )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        output = args.output_dir / (kind + ".pt")
        output.unlink(missing_ok=True)
        (args.output_dir / (kind + "_error.txt")).unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, "-u", __file__, *sys.argv[1:], "--worker", kind],
            cwd=args.upstream if kind == "upstream" else ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            check=True,
        )
        assert output.is_file(), kind
    import torch

    try:
        compare_full(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Exact ARDY source comparisons:", len(report["tests"]))


if __name__ == "__main__":
    main()
