"""Official HY-Motion Lite checkpoint and full original generate/decode parity."""

from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import faulthandler
    import signal
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    import torch, numpy as np, yaml
    from verify_motionstreamer_source import hashes, digest
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.worker in {"features", "upstream"}:
        sys.path.insert(0, str(args.upstream))
        os.chdir(args.upstream)
        from hymotion.pipeline.motion_diffusion import MotionFlowMatching
        from hymotion.network.text_encoders import text_encoder as text_module

        text_module.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] = str(args.qwen)
        text_module.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] = str(args.clip)
        config = yaml.safe_load(args.config.read_text())
        options = config["train_pipeline_args"]
        options["test_cfg"]["mean_std_dir"] = str(args.upstream / "stats")
        runtime = (
            MotionFlowMatching(
                network_module=config["network_module"],
                network_module_args=config["network_module_args"],
                **options,
            )
            .cuda()
            .eval()
        )
        state = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )["model_state_dict"]
        incompatible = runtime.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys
        assert all(
            key.startswith("body_model.") for key in incompatible.missing_keys
        ), incompatible
        if args.worker == "features":
            captions = [
                "A person walks forward.",
                "A person turns and raises both arms.",
            ]
            torch.save(
                tensors(runtime.encode_text({"text": captions})),
                args.output_dir / "text_features.pt",
            )
            print("Official original 128-token raw-caption features saved", flush=True)
            return
        network = runtime.motion_transformer
        conditions = {
            name: digest(getattr(runtime, name))
            for name in [
                "null_vtxt_feat",
                "null_ctxt_input",
                "special_game_vtxt_feat",
                "special_game_ctxt_feat",
            ]
        }
        capture = {}
        decoder = runtime.decode_motion_from_latent

        def recorded_decode(latent, **kwargs):
            capture["latent"] = latent.detach().cpu()
            return decoder(latent, **kwargs)

        runtime.decode_motion_from_latent = recorded_decode
    else:
        from motius.models.hymotion_t2m import HyMotionT2MBundle
        from motius.pipelines.hymotion_t2m import HyMotionT2MPipeline

        print("Native official artifact load begins", flush=True)
        bundle = HyMotionT2MBundle.from_pretrained(
            str(args.artifact), device="cuda"
        ).eval()
        print("Native official artifact loaded", flush=True)
        network = bundle.motion_transformer
        conditions = {
            name: digest(getattr(bundle, name))
            for name in [
                "null_vtxt_feat",
                "null_ctxt_input",
                "special_game_vtxt_feat",
                "special_game_ctxt_feat",
            ]
        }
    result = dict(neural=hashes(network), conditions=conditions, cases=[])
    features = torch.load(args.output_dir / "text_features.pt", weights_only=True)
    captions = ["A person walks forward.", "A person turns and raises both arms."]
    if args.worker == "native":
        from motius.motion.conditioning.hymotion_text import encode_hymotion_text

        encoded = encode_hymotion_text(bundle, captions)
        for key, value in features.items():
            assert torch.equal(encoded[key].cpu(), value), key
        # Both full pipelines below encode their actual per-case batches.
        print("Native raw-caption frontend equals original features", flush=True)
    for case, (batch, length, scale, special, gpu_noise) in enumerate(
        [
            (1, 24, 5.0, False, True),
            (2, 40, 1.0, False, True),
            (2, 40, 5.0, True, True),
            (2, 40, 5.0, False, False),
        ]
    ):
        text = {key: value[:batch].cuda() for key, value in features.items()}
        seeds = [71, 923][:batch]
        torch.manual_seed(491 + case)
        if args.worker == "upstream":
            runtime.random_generator_on_gpu = gpu_noise
            values = runtime.generate(
                captions[:batch],
                seed_input=seeds,
                duration_slider=length / 30.0,
                cfg_scale=scale,
                use_special_game_feat=special,
                length=length,
            )
            values["latent"] = capture["latent"]
        else:
            pipeline = HyMotionT2MPipeline(
                bundle, num_steps=50, text_guidance_scale=scale
            )
            values = pipeline.infer_text_to_motion(
                captions[:batch],
                num_frames=length,
                seed_input=seeds,
                use_special_game_feat=special,
                random_generator_on_gpu=gpu_noise,
            )
        result["cases"].append(
            tensors(
                {
                    key: values[key]
                    for key in [
                        "latent",
                        "latent_denorm",
                        "rot6d",
                        "transl",
                        "root_rotations_mat",
                        "keypoints3d",
                    ]
                }
            )
        )
        print(
            args.worker,
            "official full 50-step HY-Motion inference",
            case,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "upstream",
        "artifact",
        "checkpoint",
        "config",
        "qwen",
        "clip",
        "output-dir",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["features", "upstream", "native"])
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    for name in [
        "upstream",
        "artifact",
        "checkpoint",
        "config",
        "qwen",
        "clip",
        "output_dir",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        scope="Original Tencent HY-Motion Lite network, full MotionFlowMatching constructor/generate/decode and original WoodenMesh assets, original max128 Qwen3/CLIP raw-caption features and primary checkpoint. Four full50Euler cases: seeded single/batch, unguided, source-token guidance and CPU/GPU per-sample noise; every used motion neural/condition hash and decoded output. Controlled FP32/math attention, source text mixed BF16/FP32. Original full training/corpus metrics excluded.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in (
        ["upstream", "native"]
        if args.reuse_features
        else ["features", "upstream", "native"]
    ):
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in [
            "upstream",
            "artifact",
            "checkpoint",
            "config",
            "qwen",
            "clip",
            "output_dir",
        ]:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            command + ["--worker", kind],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            check=True,
        )
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
    print(
        "Official HY-Motion all neural states/full generation/decode exact", flush=True
    )


if __name__ == "__main__":
    main()
