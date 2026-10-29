"""Released PRISM inference versus the original Motius owner boundaries."""

from pathlib import Path
import argparse
import ast
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def previous_class(relative, name):
    source = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "HEAD:" + relative], text=True
    )
    source = source.replace(
        "from motius.models.prism.processor import",
        "from motius.motion.representation.prism import",
    )
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            node.decorator_list = []
    scope = {"__name__": "prism_original_ownership"}
    exec(compile(tree, "HEAD:" + relative, "exec"), scope)
    return scope[name]


def worker(args):
    import numpy as np
    import torch
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.worker == "reference":
        print("Loading original PRISM facade and released neural weights", flush=True)
        Bundle = previous_class("motius/models/prism/bundle.py", "PRISMBundle")
        Pipeline = previous_class("motius/pipelines/prism/pipeline.py", "PRISMPipeline")
        bundle = Bundle(
            checkpoint_path=str(args.artifact), device="cuda", training=args.training
        )
        pipe = Pipeline(bundle)
    else:
        print("Loading native PRISM facade and released neural weights", flush=True)
        from motius.pipelines.prism import PRISMPipeline

        pipe = PRISMPipeline.from_pretrained(
            str(args.artifact),
            bundle_kwargs={"device": "cuda", "training": args.training},
        )
        bundle = pipe.bundle
    print("All three PRISM neural roots loaded", flush=True)
    if args.training:
        import hashlib

        if args.worker == "reference":
            Trainer = previous_class(
                "motius/trainers/prism/prism_trainer.py", "PrismTrainer"
            )
        else:
            from motius.trainers.prism import PrismTrainer as Trainer
        output = {}
        for case, (sample, condition) in enumerate([("mode", 0.0), ("sample", 1.0)]):
            bundle.latent_sample_method = sample
            trainer = Trainer(
                bundle,
                frame_condition_rate=condition,
                prompt_drop_rate=0.1,
                max_text_length=16,
            )
            bundle.train()
            bundle.zero_grad(set_to_none=True)
            torch.manual_seed(117 + case)
            batch = dict(
                motion=torch.randn(2, 17, 138, device="cuda") * 0.01,
                caption=["A person walks forward.", "A person turns."],
                num_frames=[17, 13],
            )
            torch.manual_seed(817 + case)
            terms = trainer.train_step(batch)
            terms["loss"].backward()
            output["training" + str(case)] = dict(
                terms=tensors(terms),
                gradients={
                    name: hashlib.sha256(
                        parameter.grad.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    ).hexdigest()
                    for name, parameter in bundle.transformer.named_parameters()
                    if parameter.grad is not None
                },
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            )
        torch.save(output, args.output_dir / (args.worker + ".pt"))
        return
    prefix = args.output_dir / "prefix.npz"
    np.savez(
        prefix,
        global_orient=np.zeros((9, 3), np.float32),
        body_pose=np.zeros((9, 63), np.float32),
        transl=np.tile(np.array([[0.0, 0.0, 0.9]], np.float32), (9, 1)),
    )
    output = {
        "neural": {
            name: hashes(getattr(bundle, name))
            for name in ["transformer", "vae", "text_encoder"]
        }
    }
    output["t2m"] = tensors(
        pipe.infer_text_to_motion(
            ["A person walks forward.", "A person turns."],
            [17, 13],
            seed=81,
            num_inference_steps=50,
        )
    )
    output["tp2m"] = tensors(
        pipe.infer_temporal_motion_completion(
            "A person waves.",
            str(prefix),
            num_frames=21,
            condition_num_frames=5,
            seed=91,
            num_inference_steps=50,
        )
    )
    output["sequential"] = tensors(
        pipe.infer_sequential_text_to_motion(
            ["walk forward", "turn left"],
            [17, 13],
            pad_to_frames=33,
            ar_condition_frames=5,
            seed=93,
            num_inference_steps=50,
        )
    )
    output["rng"] = torch.cuda.get_rng_state()
    torch.save(output, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=["reference", "native"])
    parser.add_argument("--training", action="store_true")
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
        scope=(
            "PRISM original Motius training graph at HEAD versus native Trainer, actual full released PRISM-KT/UMT5/VAE networks. Mode/stochastic latent encoding, prefix condition off/on, actual raw captions/dropout, all weighted loss/diagnostic tensors, active gradient hashes and CPU/CUDA RNG exact. Explicit prepared physical-motion fixtures; full corpus/optimizer/AMP protocol excluded."
            if args.training
            else "PRISM is an original Motius method. Full released PRISM-KT neural weights, actual UMT5 captions and all three public tasks at 50 steps compared with the original Bundle/Pipeline source at HEAD. Only the relocated processor import changes in the preserved facade; neural arithmetic and backend are shared unchanged. Native ownership migration evidence, not an external-paper source certificate. Two short ragged text motions, real SMPL prefix and two-segment fixed-canvas generation."
        ),
        reference_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        artifact_revision="d51a75d176b493fe777215c827d23ef22e03fe01",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for kind in ["reference", "native"]:
            subprocess.run(
                [sys.executable, __file__, *sys.argv[1:], "--worker", kind],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
                check=True,
            )
        compare_full(
            torch.load(args.output_dir / "reference.pt", weights_only=False),
            torch.load(args.output_dir / "native.pt", weights_only=False),
            "",
            report["tests"],
        )
        report["complete"] = True
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print("PRISM full released ownership parity exact")


if __name__ == "__main__":
    main()
