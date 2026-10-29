"""Compare the original GenTrack MuJoCo judge with its native dependencies."""

from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from verify_prompthmr_video_parity import tensors

    torch.set_num_threads(4)
    if args.worker == "upstream":
        import motius.configuration.gentrack

        sys.path.insert(0, str(args.upstream))
        spec = importlib.util.spec_from_file_location(
            "original_gentrack_judge", args.original_judge
        )
        module = importlib.util.module_from_spec(spec)
        source = args.original_judge.read_text()
        source = source.replace(
            "from motius.models.gentrack.tracker_paths import (",
            "from motius.configuration.gentrack import (",
        )
        exec(compile(source, str(args.original_judge), "exec"), module.__dict__)
        graph = args.graph_root / "original_export/unified_pipeline.onnx"
    else:
        from motius.simulators.gentrack import protomotions as module

        graph = args.graph_root / "native_export/unified_pipeline.onnx"
    player = module.MotionPlayer(str(args.motion), control_dt=0.02)
    playback = [
        dict(
            state=player.get_state_at_frame(index),
            future=player.get_future_references(index, step_indices=[1, 2, 4, 8]),
        )
        for index in [0, 17, player.total_frames - 1]
    ]
    mapping = module.parse_body_mesh_mapping(args.mjcf)
    result = module.simulate_and_export(
        str(graph),
        str(args.motion),
        str(args.output_dir / (args.worker + ".json")),
        str(args.mjcf),
        mapping,
        terminate_on_unexpected_fall=False,
        export_frames=True,
        collect_tracking_trajectories=True,
    )
    result.pop("elapsed_seconds")
    frames = json.loads((args.output_dir / (args.worker + ".json")).read_text())
    if args.worker == "native":
        assert not any(
            name == "protomotions" or name.startswith(("protomotions.", "deployment."))
            for name in sys.modules
        )
    torch.save(
        tensors(dict(playback=playback, rollout=result, frames=frames)),
        args.output_dir / (args.worker + ".pt"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "upstream",
        "original-judge",
        "graph-root",
        "motion",
        "mjcf",
        "output-dir",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in [
        "upstream",
        "original_judge",
        "graph_root",
        "motion",
        "mjcf",
        "output_dir",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    report = dict(
        complete=False,
        scope="Original checked-in GenTrack MuJoCo judge (configuration import location supplied explicitly, numerical code unchanged) with pristine original ProtoMotions MotionPlayer/state helpers and independently exported source ONNX versus native judge/data components and independently exported native ONNX. Full playback, physical trajectory, tracking metrics and rendered-frame numeric data; elapsed timing excluded. One synthetic reference, not full co-evolution or dataset metrics.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in [
            "upstream",
            "original_judge",
            "graph_root",
            "motion",
            "mjcf",
            "output_dir",
        ]:
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        subprocess.run(
            [*command, "--worker", kind],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "MUJOCO_GL": "egl",
                "PHYSFLOW_ORT_INTRA_OP_THREADS": "4",
                "PHYSFLOW_ORT_INTER_OP_THREADS": "1",
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
    print("Original/native MuJoCo judge tensors exact:", len(report["tests"]))


if __name__ == "__main__":
    main()
