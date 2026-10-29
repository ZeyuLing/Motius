"""Independent EDGE decoder, DDIM, pose stitching, FK and loss comparison."""

from pathlib import Path
from types import SimpleNamespace as NS
from functools import partial
from typing import Optional, Union
import argparse, ast, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compile_nodes(path, namespace, predicate):
    nodes = [node for node in ast.parse(path.read_text()).body if predicate(node)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def source_runtime(args, config, weights, scale, offset):
    import torch, numpy as np
    import torch.nn.functional as F
    from einops import reduce
    from tqdm import tqdm

    sys.path.insert(0, str(args.upstream))
    from model.model import DanceDecoder
    from model.utils import extract, make_beta_schedule
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler

    rotations = dict(
        torch=torch, F=F, Optional=Optional, Device=Union[str, torch.device]
    )
    compile_nodes(
        args.rotation_reference,
        rotations,
        lambda node: isinstance(node, ast.FunctionDef),
    )
    namespace = {**rotations, "np": np, "torch": torch}
    compile_nodes(
        args.upstream / "dataset/quaternion.py",
        namespace,
        lambda node: isinstance(node, ast.FunctionDef),
    )
    compile_nodes(
        args.upstream / "vis.py",
        namespace,
        lambda node: (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in {"smpl_offsets", "smpl_parents"}
                for t in node.targets
            )
        )
        or (isinstance(node, ast.ClassDef) and node.name == "SMPLSkeleton"),
    )
    import copy

    namespace.update(
        copy=copy,
        nn=torch.nn,
        F=F,
        partial=partial,
        reduce=reduce,
        tqdm=tqdm,
        extract=extract,
        make_beta_schedule=make_beta_schedule,
    )
    compile_nodes(
        args.upstream / "model/diffusion.py",
        namespace,
        lambda node: isinstance(node, (ast.FunctionDef, ast.ClassDef)),
    )
    network = DanceDecoder(**config["network"]).cuda()
    network.load_state_dict(weights, strict=True)
    skeleton = namespace["SMPLSkeleton"]("cuda")
    diffusion = namespace["GaussianDiffusion"](
        network,
        150,
        151,
        skeleton,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=2,
    ).cuda()
    normalizer = Normalizer.__new__(Normalizer)
    normalizer.scaler = MinMaxScaler((-1, 1), clip=True)
    normalizer.scaler.scale_, normalizer.scaler.min_ = scale.cpu(), offset.cpu()
    tree = ast.parse((args.upstream / "model/diffusion.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GaussianDiffusion"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "render_sample"
    )
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.value) == "normalizer.unnormalize(samples)"
    )
    end = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "mode == 'long'"
    )
    # The original long-mode branch contains FK/rendering after its pose merge.
    branch = method.body[end]
    cutoff = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "full_pose" for t in n.targets)
    )
    body = method.body[start:end] + branch.body[:cutoff]
    body += [
        ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id="full_q", ctx=ast.Load()),
                    ast.Name(id="full_pos", ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            )
        )
    ]
    fn = ast.fix_missing_locations(
        ast.FunctionDef(
            name="source_pose",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=x) for x in ["samples", "cond", "normalizer"]],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
        )
    )
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(args.upstream / "model/diffusion.py"),
            "exec",
        ),
        namespace,
    )
    return network, diffusion, skeleton, normalizer, namespace["source_pose"]


def worker(args):
    import torch, numpy as np
    from safetensors.torch import load_file
    from verify_motionstreamer_source import hashes
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
    config = json.loads((args.artifact / "edge_config.json").read_text())["config"]
    weights = load_file(str(args.artifact / "model.safetensors"))
    stats = np.load(args.artifact / "normalizer.npz")
    scale, offset = [torch.from_numpy(stats[name]).cuda() for name in ["scale", "min"]]
    if args.worker == "upstream":
        network, diffusion, skeleton, normalizer, pose = source_runtime(
            args, config, weights, scale, offset
        )
    else:
        from motius.models.edge import EDGEBundle
        from motius.pipelines.edge import EDGEPipeline
        from motius.pipelines.edge.sampler import edge_ddim_sample, stitch_edge_windows
        from motius.motion.representation.normalization import range_denormalize
        from motius.motion.representation.edge import edge_axis_angle_forward_kinematics
        from motius.trainers.edge import EDGETrainer

        bundle = EDGEBundle.from_pretrained(str(args.artifact)).cuda()
        network = bundle.network
        trainer = EDGETrainer(bundle)
    result = dict(neural=hashes(network), inference=[], training=[])
    network.eval()
    for case, batch in enumerate([1, 2]):
        torch.manual_seed(118 + case)
        condition = torch.randn(batch, 150, 4800, device="cuda") * 0.03
        torch.manual_seed(851 + case)
        with torch.no_grad():
            if args.worker == "upstream":
                normalized = diffusion.long_ddim_sample((batch, 150, 151), condition)
                angles, root = pose(normalized.clone(), condition, normalizer)
                joints = skeleton.forward(angles, root)
                public_joints = torch.stack(
                    (joints[..., 0], joints[..., 2], -joints[..., 1]), dim=-1
                )[0]
            else:
                normalized = edge_ddim_sample(network, condition)
                physical = range_denormalize(normalized, offset, scale)
                _, angles, root = stitch_edge_windows(physical, return_pose=True)
                angles, root = angles.unsqueeze(0), root.unsqueeze(0)
                joints = edge_axis_angle_forward_kinematics(angles, root)
                public_joints = torch.stack(
                    (joints[..., 0], joints[..., 2], -joints[..., 1]), dim=-1
                )[0]
                public = EDGEPipeline(bundle).infer_music_to_dance(
                    music_features=condition, seed=851 + case
                )
                assert np.array_equal(
                    public.joints, public_joints.cpu().numpy()
                ), "Public pipeline drift"
            result["inference"].append(
                tensors(
                    dict(
                        normalized=normalized,
                        angles=angles,
                        root=root,
                        joints=joints,
                        public_joints=public_joints,
                    )
                )
            )
        print(
            args.worker,
            "full 50-step EDGE inference",
            batch,
            "windows saved",
            flush=True,
        )
    network.train()
    if args.worker == "upstream":
        from model.adan import Adan

        ema_network = diffusion.master_model
    else:
        from motius.optimizers.adan import Adan

        ema_network = bundle.ema_network
    optimizer = Adan(network.parameters(), lr=4e-4, weight_decay=0.02)
    from verify_motionstreamer_source import digest

    def fingerprint(value):
        if isinstance(value, torch.Tensor):
            return digest(value)
        if isinstance(value, dict):
            return {str(key): fingerprint(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [fingerprint(item) for item in value]
        return value

    for case in range(3):
        torch.manual_seed(190 + case)
        motion = torch.randn(2, 150, 151, device="cuda") * 0.15
        condition = torch.randn(2, 150, 4800, device="cuda") * 0.03
        network.zero_grad(set_to_none=True)
        torch.manual_seed(739 + case)
        if args.worker == "upstream":
            loss, losses = diffusion(motion, condition)
        else:
            trainer.train()
            values = trainer.train_step(dict(motion=motion, music_features=condition))
            loss = values["loss"]
            losses = [
                values[key]
                for key in [
                    "reconstruction_loss",
                    "velocity_loss",
                    "fk_loss",
                    "foot_loss",
                ]
            ]
        loss.backward()
        gradients = hashes(network, gradients=True)
        optimizer.step()
        if args.worker == "upstream":
            diffusion.ema.update_model_average(ema_network, network)
        else:
            trainer.update_ema()
        result["training"].append(
            tensors(
                dict(
                    loss=loss,
                    components=losses,
                    gradients=gradients,
                    online=hashes(network),
                    ema=hashes(ema_network),
                    optimizer=fingerprint(optimizer.state_dict()),
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                )
            )
        )
        print(args.worker, "EDGE all four losses/gradients", case, "saved", flush=True)
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "rotation-reference", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "rotation_reference", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        scope="Original EDGE network/full GaussianDiffusion methods/SMPLSkeleton and untouched original pose-stitching statements. Full 50-step DDIM for one/two 150-frame windows, physical pose/FK and public Pipeline joints, three original four-term objectives/all gradient hashes/Adan optimizer states/online and source EMA updates/RNG. Published released EMA neural weights and supplied normalizer; prepared Jukebox fixtures, controlled FP32/math attention, independent PyTorch3D v0.7.2 pure rotation source. Original raw audio/data distribution, full epoch training/validation schedule and FP16 are separate scopes.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in ["upstream", "artifact", "rotation_reference", "output_dir"]:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            command + ["--worker", kind],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
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
    print("Original EDGE inference/losses/gradients exact", flush=True)


if __name__ == "__main__":
    main()
