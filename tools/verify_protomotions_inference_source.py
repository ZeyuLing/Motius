"""Original G1 export composition versus native raw-observation inference."""

from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors, compare_full


def worker(args):
    import torch

    torch.set_num_threads(4)
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        file = args.upstream / "deployment/export_bm_tracker_onnx.py"
        spec = importlib.util.spec_from_file_location("original_proto_export", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured = {}
        original_export = torch.onnx.export

        def capture(model, inputs, *positional, **kwargs):
            captured.update(
                model=model, inputs=inputs, input_names=kwargs["input_names"]
            )
            return original_export(model, inputs, *positional, **kwargs)

        # Observe the original exporter without replacing any neural/geometry computation.
        torch.onnx.export = capture
        try:
            module.export_tracker(
                str(args.artifact / "last.ckpt"),
                str(args.output_dir / "original_export"),
                validate=True,
            )
        finally:
            torch.onnx.export = original_export
        model = captured["model"]
        fixtures = []
        outputs = []
        for batch_size in [1, 3]:
            torch.manual_seed(1181 + batch_size)
            inputs = []
            for sample in captured["inputs"]:
                shape = (batch_size, *sample.shape[1:])
                value = torch.randn(shape) * 0.17
                if shape[-1] == 4:
                    value = torch.nn.functional.normalize(value, dim=-1)
                inputs.append(value)
            fixture = dict(zip(captured["input_names"], inputs))
            fixtures.append(fixture)
            with torch.no_grad():
                values = model(*inputs)
            outputs.append(
                tensors(
                    dict(
                        zip(
                            [
                                "actions",
                                "joint_pos_targets",
                                "stiffness_targets",
                                "damping_targets",
                            ],
                            values,
                        )
                    )
                )
            )
        torch.save(fixtures, args.output_dir / "fixtures.pt")
    else:
        from motius.pipelines.protomotions import ProtoMotionsPipeline
        from motius import Pipeline

        pipeline = ProtoMotionsPipeline.from_original_checkpoint(
            args.artifact / "last.ckpt"
        )
        fixtures = torch.load(args.output_dir / "fixtures.pt", weights_only=True)
        outputs = []
        for fixture in fixtures:
            result = pipeline(fixture)
            outputs.append(
                tensors(
                    {
                        name: result[name]
                        for name in [
                            "actions",
                            "joint_pos_targets",
                            "stiffness_targets",
                            "damping_targets",
                        ]
                    }
                )
            )
        # Independently exported graphs must agree on nonzero raw robot inputs.
        from motius.pipelines.protomotions.export import export_tracker
        import onnxruntime as ort
        import numpy as np
        import yaml

        export_tracker(args.artifact / "last.ckpt", args.output_dir / "native_export")
        sessions = [
            ort.InferenceSession(
                str(args.output_dir / name / "unified_pipeline.onnx"),
                providers=["CPUExecutionProvider"],
            )
            for name in ["original_export", "native_export"]
        ]
        for fixture in fixtures:
            graphs = [
                session.run(
                    None,
                    {
                        item.name: fixture[item.name].numpy()
                        for item in session.get_inputs()
                    },
                )
                for session in sessions
            ]
            for left, right in zip(*graphs):
                np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-5)
        metadata = [
            yaml.safe_load(
                (args.output_dir / name / "unified_pipeline.yaml").read_text()
            )
            for name in ["original_export", "native_export"]
        ]
        assert (
            metadata[0] == metadata[1]
        ), "Native deployment metadata differs from actual original exporter"
        # Verify complete current neural state, processing metadata and a cold public load.
        with torch.no_grad():
            pipeline.bundle.actor.mean.mlp[-1].bias.add_(0.03125)
        pipeline.save_pretrained(args.output_dir / "exported")
        restored = Pipeline.from_pretrained(args.output_dir / "exported")
        for name, value in pipeline.bundle.state_dict().items():
            assert torch.equal(value, restored.bundle.state_dict()[name]), name
        assert restored.deployment == pipeline.deployment
        for fixture in fixtures:
            left = pipeline(fixture)
            right = restored(fixture)
            compare_full(tensors(left), tensors(right), "artifact", [])
    torch.save(outputs, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "source_parity.json"
    report = dict(
        complete=False,
        scope="Released G1 checkpoint, pristine original export_tracker and its actual UnifiedPipelineModule versus native raw-observation Pipeline; real observation/quaternion encoding, actor, normalization and PD conversion. Synthetic robot-state inputs, not physical simulation or PPO/AMP training. Independently exported source/native ONNX graphs checked on nonzero B=1/B=3 inputs (rtol=atol=2e-5); deployment YAML metadata exactly equal.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        artifact=json.loads((args.artifact / "provenance.json").read_text()),
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        env = {
            **os.environ,
            "PYTHONPATH": str(root),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MUJOCO_GL": "egl",
        }
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
            cwd=args.upstream if kind == "upstream" else root,
            env=env,
            check=True,
        )
    import torch

    torch.set_num_threads(4)
    try:
        compare_full(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Original complete G1 deployment composition and native inference are exact")


if __name__ == "__main__":
    main()
