"""Verify MotionCanvas condition ownership against immutable original methods."""

from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
import argparse, ast, hashlib, json, os, subprocess, sys, types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from motius.models.animatecanvas import AnimateCanvasBundle
    from motius.motion.conditioning.hymotion_conditions import (
        prepare_padding,
        prepare_condition_context,
        mask_text_cond,
        predict_conditioned_flow,
    )
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
    bundle = AnimateCanvasBundle.from_pretrained(str(args.artifact), device="cuda")
    if args.worker == "snapshot":
        tree = ast.parse(args.source_bundle.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MotionCanvasBundle"
        )
        names = {
            "prepare_padding",
            "prepare_condition_context",
            "mask_text_cond",
            "maybe_inject_source_token",
            "predict_flow",
        }
        methods = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in methods} == names
        helper = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_length_to_mask"
        )
        namespace = dict(
            torch=torch,
            F=F,
            Tensor=Tensor,
            List=List,
            Optional=Optional,
            Set=Set,
            Tuple=Tuple,
            Union=Union,
        )
        exec(
            compile(
                ast.Module(body=[helper, *methods], type_ignores=[]),
                str(args.source_bundle),
                "exec",
            ),
            namespace,
        )
        for name in names:
            setattr(bundle, name, types.MethodType(namespace[name], bundle))
    result = dict(neural=hashes(bundle.motion_transformer), cases=[])
    for case in range(4):
        bundle.train(case != 0)
        torch.manual_seed(713 + case)
        source = torch.randn(2, 12, 198, device="cuda", requires_grad=True) * 0.03
        target = torch.randn(2, 10, 198, device="cuda") * 0.03
        mask = (torch.rand_like(source) > 0.4).float()
        reference = torch.randn(2, 2, 198, device="cuda") * 0.03
        vtxt = torch.randn(2, 1, 768, device="cuda", requires_grad=True) * 0.03
        ctxt = torch.randn(2, 4, 4096, device="cuda", requires_grad=True) * 0.03
        text_mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0]], device="cuda", dtype=torch.bool
        )
        sources = ["Other", "Other"] if case == 3 else ["Game", "taObao"]
        bundle.zero_grad(set_to_none=True)
        torch.manual_seed(942 + case)
        if args.worker == "snapshot":
            padded = bundle.prepare_padding(
                source, target, [10, 8], src_mask=mask, ref_pose=reference
            )
            text = bundle.mask_text_cond(
                vtxt,
                ctxt,
                force_mask=case == 2,
                cond_mask_prob=0.35,
                return_text_available=True,
            )
            context = bundle.prepare_condition_context(
                padded[0], ref_pose=reference, src_mask=padded[1]
            )
        else:
            padded = prepare_padding(
                source, target, [10, 8], src_mask=mask, ref_pose=reference
            )
            text = mask_text_cond(
                bundle,
                vtxt,
                ctxt,
                force_mask=case == 2,
                cond_mask_prob=0.35,
                return_text_available=True,
            )
            context = prepare_condition_context(
                padded[0], ref_pose=reference, src_mask=padded[1]
            )
        noisy = torch.randn(2, context.shape[1], 198, device="cuda") * 0.03
        inputs = dict(
            x_input=torch.cat([noisy, context], dim=-1),
            ctxt_input=text[1],
            vtxt_input=text[0],
            timesteps=torch.tensor([0.25, 0.75], device="cuda"),
            x_mask_temporal=padded[-1],
            ctxt_mask_temporal=text_mask,
            sources=sources,
            trigger_sources={"Taobao", "Game"},
            special_game_prob=1.0 if case in [0, 2] else 0.5,
        )
        prediction = (
            bundle.predict_flow(**inputs)
            if args.worker == "snapshot"
            else predict_conditioned_flow(bundle, **inputs)
        )
        loss = prediction.square().mean()
        loss.backward()
        result["cases"].append(
            tensors(
                dict(
                    padded=padded,
                    context=context,
                    text=text,
                    prediction=prediction,
                    loss=loss,
                    network_gradients=hashes(bundle.motion_transformer, gradients=True),
                    bundle_gradients={
                        name: None if parameter.grad is None else parameter.grad
                        for name, parameter in bundle.named_parameters()
                        if not name.startswith("motion_transformer.")
                    },
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                )
            )
        )
        print(
            args.worker,
            "MotionCanvas real 0.46B conditioning/forward/gradient",
            case,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["artifact", "source-bundle", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["snapshot", "native"])
    args = parser.parse_args()
    for name in ["artifact", "source_bundle", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        scope="MotionCanvas condition ownership migration: unchanged original Motius Bundle methods executed on the real released 0.46B MMDiT versus external padding/dropout/source-token processors and atomic Bundle forward. Four train/eval/forced-null/source-selection cases, every neural state/gradient hash, orphan condition gradients and CPU/CUDA RNG. Prepared text-feature fixture; this is original-Motius-method migration evidence, not an independent external training or raw text/inference benchmark certificate.",
        source_bundle_sha256=hashlib.sha256(
            args.source_bundle.read_bytes()
        ).hexdigest(),
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["snapshot", "native"]:
        cmd = [sys.executable, str(Path(__file__).resolve())]
        for name in ["artifact", "source_bundle", "output_dir"]:
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            cmd + ["--worker", kind],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
            check=True,
        )
    try:
        compare_full(
            torch.load(args.output_dir / "snapshot.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "MotionCanvas original condition ownership/forward/gradients exact", flush=True
    )


if __name__ == "__main__":
    main()
