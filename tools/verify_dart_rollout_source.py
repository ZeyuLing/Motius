"""Original DART FlowMDM rollout with actual licensed bodies and text networks."""

from pathlib import Path
import argparse
import ast
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import copy
    import enum
    import math
    import random
    import numpy as np
    import torch
    from types import SimpleNamespace
    from verify_dart_source_parity import source_definitions
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors
    from motius.configuration.dart import read_arguments

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["MOTIUS_BODY_MODEL_DIR"] = str(args.body_models)
    denoiser_file = (
        args.artifact / "mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt"
    )
    denoiser_args = read_arguments(denoiser_file.parent / "args.yaml").denoiser_args
    vae_file = args.artifact / denoiser_args.mvae_path
    vae_args = read_arguments(vae_file.parent / "args.yaml")
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        # Resolve only the original data-path configuration. All body/data/NN
        # classes and the complete rollout body remain original source.
        import config_files.data_paths as data_paths

        data_paths.body_model_dir = args.body_models
        from utils import smpl_utils, misc_util
        from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
        from model.mld_denoiser import DenoiserTransformer
        from model.mld_vae import AutoMldVae
        from tqdm import tqdm

        denoiser = DenoiserTransformer(**vars(denoiser_args.model_args)).cuda().eval()
        denoiser.load_state_dict(
            torch.load(denoiser_file, map_location="cpu", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
        vae = AutoMldVae(**vars(vae_args.model_args)).cuda().eval()
        state = torch.load(vae_file, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        vae.load_state_dict(state, strict=True)
        vae.latent_mean, vae.latent_std = (
            state["latent_mean"].cuda(),
            state["latent_std"].cuda(),
        )
        scope = dict(
            torch=torch,
            th=torch,
            F=torch.nn.functional,
            np=np,
            enum=enum,
            math=math,
            deepcopy=copy.deepcopy,
        )
        source_definitions(
            args.upstream / "diffusion/nn.py", scope, {"mean_flat", "sum_flat"}
        )
        source_definitions(args.upstream / "diffusion/losses.py", scope)
        source_definitions(args.upstream / "diffusion/gaussian_diffusion.py", scope)
        source_definitions(args.upstream / "diffusion/respace.py", scope)
        scope["gd"] = SimpleNamespace(**scope)
        source_definitions(
            args.upstream / "mld/train_mld.py", scope, {"create_gaussian_diffusion"}
        )
        make_diffusion = scope["create_gaussian_diffusion"]
        source_definitions(
            args.upstream / "mld/rollout_mld.py",
            {"nn": torch.nn},
            {"ClassifierFreeWrapper"},
        )
        wrapper_scope = {"nn": torch.nn}
        source_definitions(
            args.upstream / "mld/rollout_mld.py",
            wrapper_scope,
            {"ClassifierFreeWrapper"},
        )
        guided = wrapper_scope["ClassifierFreeWrapper"](denoiser)
        rollout_scope = dict(
            torch=torch,
            np=np,
            **{
                name: getattr(smpl_utils, name)
                for name in [
                    "PrimitiveUtility",
                    "get_new_coordinate",
                    "FOOT_JOINTS_IDX",
                ]
            },
            encode_text=misc_util.encode_text,
        )
        # AST extraction skips the app's Python-3.10-only dataclass defaults
        # and unrelated training imports; it does not modify any statement.
        source_definitions(
            args.upstream / "mld/sample_flowmdm_mld.py", rollout_scope, {"rollout"}
        )
        rollout = rollout_scope["rollout"]
    else:
        from motius.pipelines.dart import DARTPipeline
        from motius.models.dart import DARTBundle

        bundle = DARTBundle.from_pretrained(str(args.artifact), device="cuda")
        denoiser, vae = bundle.denoiser_model, bundle.vae_model
    result = dict(neural={"denoiser": hashes(denoiser), "vae": hashes(vae)}, cases=[])
    capture = {}

    def capture_first(module, values):
        if capture:
            return
        x, timestep, condition = values
        capture.update(
            tensors(
                dict(
                    noise=x,
                    timestep=timestep,
                    text=condition["text_embedding"],
                    history=condition["history_motion_normalized"],
                )
            )
        )

    denoiser.register_forward_pre_hook(capture_first)
    cases = [
        ("A person walks forward.", 17, "", 5.0, False, False),
        ("", 21, "ddim5", 2.5, False, False),
        ("A person turns left.", 16, "", 1.0, True, False),
        ("A person raises both arms.", 9, "ddim5", 5.0, False, True),
    ]
    for index, (caption, length, respacing, guidance, predicted, floor) in enumerate(
        cases
    ):
        capture.clear()
        if args.worker == "upstream":
            dataset = SinglePrimitiveDataset(
                cfg_path=str(args.artifact / vae_args.data_args.cfg_path),
                sequence_path=str(args.artifact / "data/stand_20fps.pkl"),
                dataset_path=str(args.artifact / vae_args.data_args.data_dir),
                device="cuda",
                batch_size=1,
                body_type=vae_args.data_args.body_type,
                enforce_gender="male",
                enforce_zero_beta=1,
            )
            options = SimpleNamespace(
                device="cuda",
                batch_size=1,
                respacing=respacing,
                guidance_param=guidance,
                zero_noise=False,
                use_predicted_joints=predicted,
                fix_floor=floor,
            )
            config = SimpleNamespace(**vars(denoiser_args.diffusion_args))
            config.respacing = respacing
            diffusion = make_diffusion(config)
            random.seed(71 + index)
            np.random.seed(71 + index)
            torch.manual_seed(71 + index)
            with torch.no_grad():
                raw = rollout(
                    {"text": [caption], "lengths": [length]},
                    denoiser_args,
                    guided,
                    vae_args,
                    vae,
                    diffusion,
                    dataset,
                    options,
                )
            output = {
                name: raw[name][0]
                for name in ["betas", "transl", "global_orient", "body_pose", "joints"]
            }
            from pytorch3d.transforms import matrix_to_rotation_6d

            transform = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            )
            translation = torch.einsum("ij,tj->ti", transform, output["transl"].cpu())
            root_rotation = torch.einsum(
                "ij,tjk->tik", transform, output["global_orient"].cpu()
            )
            rotations = torch.cat(
                [root_rotation[:, None], output["body_pose"].cpu()], dim=1
            )
            canonical_motion = torch.cat(
                [translation, matrix_to_rotation_6d(rotations).reshape(length, 132)],
                dim=-1,
            )
            text_network = dataset.clip_model
        else:
            pipeline = DARTPipeline(
                bundle,
                respacing=respacing,
                guidance_param=guidance,
                use_predicted_joints=predicted,
                fix_floor=floor,
            )
            raw = pipeline.infer_t2m_smpl([caption], [length], seed=71 + index)[0]
            output = {
                name: raw[name]
                for name in ["betas", "transl", "global_orient", "body_pose", "joints"]
            }
            canonical_motion = pipeline.infer_text_to_motion(
                [caption], [length], seed=71 + index
            )[0]
            text_network = bundle.clip_model
        result["cases"].append(
            tensors(
                dict(
                    output=output,
                    motion135=canonical_motion,
                    first_network_input=capture.copy(),
                    cuda_rng=torch.cuda.get_rng_state(),
                )
            )
        )
        result["text_neural"] = hashes(text_network)
        print(
            args.worker,
            "complete original DART body rollout",
            index,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "body-models", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "body_models", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        scope="Original DART released HumanML3D denoiser/VAE/CLIP weights, original complete FlowMDM rollout and actual SinglePrimitiveDataset seed/text/FK. Actual licensed official merged SMPL-H seed and original default SMPL-X rollout reconstruction. Four DDPM/DDIM/ragged/predicted-joint/floor cases, raw world SMPL parameters/joints, independently converted public Motion135 outputs, all neural hashes and CUDA RNG. Original rollout function extracted without statement changes to omit app dataclass/framework imports; only asset paths configured. Ordinary rollout_mld's seed-inclusive output, training and corpus benchmarks excluded.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for kind in ["upstream", "native"]:
            command = [sys.executable, str(Path(__file__).resolve())]
            for name in ["upstream", "artifact", "body_models", "output_dir"]:
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            subprocess.run(
                command + ["--worker", kind],
                cwd=ROOT,
                env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                check=True,
            )
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
    print("Original complete DART licensed-body rollout exact", flush=True)


if __name__ == "__main__":
    main()
