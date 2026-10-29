"""Compare complete released-weight MotionCanvas inference before/after ownership moves."""

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
    from motius.motion.conditioning.hymotion_text import encode_hymotion_text
    import motius.pipelines.animatecanvas.animatecanvas_pipeline as current
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
    bundle = AnimateCanvasBundle.from_pretrained(
        str(args.artifact), device="cuda"
    ).eval()
    if args.worker == "features":
        # Preserve this method's existing lazy text-tower device/precision.
        captions = ["A person walks forward.", "A person turns and raises both arms."]
        values = encode_hymotion_text(bundle, captions)
        torch.save(tensors(values), args.output_dir / "text_features.pt")
        print(
            "Actual released text weights/own MotionCanvas frontend features saved",
            flush=True,
        )
        return
    if args.worker == "snapshot":
        source_bundle = args.snapshot_root / "motius/models/motioncanvas/bundle.py"
        tree = ast.parse(source_bundle.read_text())
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
                str(source_bundle),
                "exec",
            ),
            namespace,
        )
        for name in names:
            setattr(bundle, name, types.MethodType(namespace[name], bundle))
        source_pipeline = (
            args.snapshot_root
            / "motius/pipelines/motioncanvas/motioncanvas_pipeline.py"
        )
        cls = next(
            n
            for n in ast.parse(source_pipeline.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "MotionCanvasPipeline"
        )
        cls.decorator_list = []  # Registration is unrelated to the numerical class.
        pipeline_namespace = dict(current.__dict__)
        exec(
            compile(
                ast.Module(body=[cls], type_ignores=[]), str(source_pipeline), "exec"
            ),
            pipeline_namespace,
        )
        pipeline_type = pipeline_namespace["MotionCanvasPipeline"]
    else:
        pipeline_type = current.AnimateCanvasPipeline
    pipeline = pipeline_type(bundle, num_steps=50)
    text = torch.load(args.output_dir / "text_features.pt", weights_only=True)
    length = 24
    generator = torch.Generator(device="cuda").manual_seed(391)
    source = (
        torch.randn(2, length, 198, device="cuda", generator=generator)
        * 0.03
        * bundle.std
        + bundle.mean
    )
    target = source + bundle.std * 0.01
    mask = torch.ones(2, length, device="cuda")
    mask[:, :4] = 0
    mask[:, -2:] = 0
    keymask = torch.ones_like(mask)
    keymask[:, ::8] = 0
    common = dict(num_frames=[24, 20], data_src=["Game", "Taobao"], **text)
    result = {}
    for case in range(4):
        if case == 0:
            values = pipeline.infer_m2m(seed=351, **common)
        elif case == 1:
            values = pipeline.infer_temporal_motion_completion(
                source, mask, seed=352, **common
            )
        elif case == 2:
            values = pipeline.infer_motion_editing(
                source, mask, edit_source_motion=target, seed=353, **common
            )
        else:
            values = pipeline.infer_keyframe_motion_control(
                source, keymask, seed=354, **common
            )
        result[str(case)] = tensors(values)
        print(
            args.worker,
            "real 0.46B full 50-step public MotionCanvas",
            case,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["artifact", "snapshot-root", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["features", "snapshot", "native"])
    args = parser.parse_args()
    for name in ["artifact", "snapshot_root", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    source_pipeline = (
        args.snapshot_root / "motius/pipelines/motioncanvas/motioncanvas_pipeline.py"
    )
    report = dict(
        complete=False,
        scope="Actual published MotionCanvas 0.46B and hash-verified Qwen3/CLIP weights: four full 50-step public inference paths (T2M, temporal completion, editing, keyframes), decoded outputs and known-coordinate projection versus the preserved pre-ownership original Motius pipeline class/Bundle methods. Uses paired real raw-caption features from this artifact's own existing lazy-device frontend; prepared physical-motion fixtures. This certifies ownership migration, not an independent external benchmark/whole original training protocol.",
        snapshot_pipeline_sha256=hashlib.sha256(
            source_pipeline.read_bytes()
        ).hexdigest(),
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["features", "snapshot", "native"]:
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in ["artifact", "snapshot_root", "output_dir"]:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            command + ["--worker", kind],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
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
        "MotionCanvas complete ownership-migration public inference exact", flush=True
    )


if __name__ == "__main__":
    main()
