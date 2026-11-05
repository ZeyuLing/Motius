"""Compare native G1 replay conversion and packaging with original implementations."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    if args.worker == "upstream":
        sys.path[:0] = [str(args.upstream), str(args.converter.parent)]
        spec = importlib.util.spec_from_file_location(
            "original_converter", args.converter
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        convert = original.main
        names = original.CANONICAL_G1_DOF_NAMES
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        working = args.output_dir / "source-working"
        (working / "protomotions/data").mkdir(parents=True, exist_ok=True)
        link = working / "protomotions/data/assets"
        if not link.exists():
            link.symlink_to(args.assets, target_is_directory=True)
        os.chdir(working)
    else:
        from motius.motion.representation.protomotions.qpos_conversion import (
            convert_directory as convert,
            CANONICAL_G1_DOF_NAMES as names,
        )
        from motius.datasets.protomotions_motion import MotionLib, MotionLibConfig

    source = args.output_dir / "fixtures"
    source.mkdir(exist_ok=True)
    # Cover reordered/missing joints, quaternion interpolation, contact labels,
    # and a source rate equal to the target rate as well as 30 -> 50 Hz.
    for index, fps in enumerate([30, 50]):
        selected = names if index == 1 else names[::2][::-1]
        t = np.arange(61, dtype=np.float32) / fps
        qpos = np.zeros((len(t), 7 + len(selected)), dtype=np.float32)
        qpos[:, 0] = t * 0.15
        qpos[:, 2] = 0.78 + 0.02 * np.sin(t)
        qpos[:, 3] = np.cos(t * 0.08)
        qpos[:, 6] = np.sin(t * 0.08)
        qpos[:, 7:] = 0.025 * np.sin(t[:, None] + np.arange(len(selected)))
        np.savez(
            source / f"clip_{index}.npz",
            qpos=qpos,
            frequency=fps,
            joint_names=np.asarray(["root", *selected]),
        )
    directory = args.output_dir / args.worker
    options = dict(
        input_dir=source,
        output_dir=directory,
        output_fps=50,
        robot_type="g1",
        force_remake=True,
        max_files=None,
        num_rank=1,
        slurm_rank=0,
        manifest=None,
    )
    if args.worker == "native":
        options["asset_root"] = args.assets
    convert(**options)
    import inspect

    csv_source = args.converter.with_name("convert_g1_csv_to_proto.py")
    if args.worker == "upstream":
        spec = importlib.util.spec_from_file_location(
            "original_csv_converter", csv_source
        )
        original_csv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_csv)
        csv_convert = original_csv.main
    else:
        from motius.motion.representation.protomotions.csv_conversion import (
            convert_csv_directory as csv_convert,
        )
    for index, fps in enumerate([100, 50]):
        csv_input = args.output_dir / ("csv-fixture-" + str(index))
        csv_input.mkdir(exist_ok=True)
        t = np.arange(101, dtype=np.float32) / fps
        qpos = np.zeros((len(t), 7 + len(names)), dtype=np.float32)
        qpos[:, 0] = 0.1 * t
        qpos[:, 2] = 0.78 + 0.02 * np.sin(t)
        qpos[:, 3] = np.cos(0.07 * t)
        qpos[:, 6] = np.sin(0.07 * t)
        qpos[:, 7:] = 0.025 * np.sin(t[:, None] + np.arange(len(names)))
        np.savetxt(csv_input / ("csv-clip-" + str(index) + ".csv"), qpos, delimiter=",")
        csv_options = {}
        for name, parameter in inspect.signature(csv_convert).parameters.items():
            default = parameter.default
            csv_options[name] = getattr(default, "default", default)
        csv_options.update(
            input_dir=csv_input,
            output_dir=directory,
            input_fps=fps,
            output_fps=50,
            pos_units="m",
            rot_format="quat_wxyz",
            joint_units="rad",
            has_header=False,
            has_frame_column=False,
            force_remake=True,
        )
        if args.worker == "native":
            csv_options["asset_root"] = args.assets
        csv_convert(**csv_options)
    if args.worker == "upstream":
        file = (
            args.upstream
            / "data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py"
        )
        spec = importlib.util.spec_from_file_location(
            "original_retargeted_converter", file
        )
        original_retargeted = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_retargeted)
        retarget_convert = original_retargeted.main
    else:
        from motius.motion.representation.protomotions.retargeted_conversion import (
            convert_retargeted_directory as retarget_convert,
        )
    retarget_input = args.output_dir / "retargeted-fixtures"
    retarget_input.mkdir(exist_ok=True)
    for index, fps in enumerate([60, 30]):
        t = np.arange(121, dtype=np.float32) / fps
        position = np.stack(
            [0.1 * t, 0.02 * np.cos(t), 0.78 + 0.02 * np.sin(t)], axis=-1
        )
        quaternion = np.zeros((len(t), 4), dtype=np.float32)
        quaternion[:, 0] = np.cos(0.07 * t)
        quaternion[:, 3] = np.sin(0.07 * t)
        joints = 0.025 * np.sin(t[:, None] + np.arange(len(names)))
        np.savez(
            retarget_input / ("retarget-clip-" + str(index) + ".npz"),
            base_frame_pos=position,
            base_frame_wxyz=quaternion,
            joint_angles=joints,
            fps=fps,
        )
    retarget_options = {
        name: getattr(parameter.default, "default", parameter.default)
        for name, parameter in inspect.signature(retarget_convert).parameters.items()
    }
    retarget_options.update(
        retargeted_motion_dir=retarget_input,
        output_dir=directory,
        input_fps=30,
        output_fps=30,
        force_remake=True,
    )
    if args.worker == "native":
        retarget_options["asset_root"] = args.assets
    retarget_convert(**retarget_options)
    motions = {}
    for file in sorted(directory.glob("*.motion")):
        value = torch.load(file, map_location="cpu", weights_only=False)
        conversion = value.pop("state_conversion", None)
        if conversion is not None:
            assert conversion.value == "common"
        motions[file.name] = value
    torch.manual_seed(301)
    library = MotionLib(MotionLibConfig(motion_file=str(directory)), device="cpu")
    library.save_to_file(directory / "packaged.pt")
    library = MotionLib(
        MotionLibConfig(motion_file=str(directory / "packaged.pt")), device="cpu"
    )
    ids = torch.tensor([0, 1, 0, 1])
    sampled = library.get_motion_state(
        ids, torch.tensor([0.0, 0.1, 0.9, 1.1])
    ).to_dict()
    sampled.pop("state_conversion", None)
    from verify_prompthmr_video_parity import tensors

    package = torch.load(
        directory / "packaged.pt", map_location="cpu", weights_only=False
    )
    # Absolute provenance roots differ between independent worker outputs.
    package["motion_files"] = [Path(name).name for name in package["motion_files"]]
    torch.save(
        tensors(dict(motions=motions, package=package, sampled=sampled)),
        args.output_dir / (args.worker + ".pt"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "converter", "assets", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "converter", "assets", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    report = dict(
        complete=False,
        tests=[],
        scope="Original checked-in GenTrack G1 NPZ/CSV converters, original PyRoki-output converter and pristine ProtoMotions FK/contact/MotionLib versus native conversion, portable COMMON data and package/reload/interpolation. Package provenance compared as relative filenames because worker output roots differ. Deterministic synthetic trajectories; not dataset coverage or full co-evolution training.",
        original_converter_sha256=hashlib.sha256(
            args.converter.read_bytes()
        ).hexdigest(),
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--worker", kind],
            env={**os.environ, "PYTHONPATH": str(ROOT), "MUJOCO_GL": "egl"},
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
        "Original/native replay conversion and package tensors exact:",
        len(report["tests"]),
    )


if __name__ == "__main__":
    main()
