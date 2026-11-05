"""Compare native G1 trajectory IK with the original ProtoMotions/PyRoki solver."""

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
    import jax
    import jax.numpy as jnp
    import jaxlie
    import numpy as np
    import pyroki as pk
    import yourdfpy

    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        spec = importlib.util.spec_from_file_location(
            "original_pyroki_solver",
            args.upstream / "pyroki/batch_retarget_to_g1_from_keypoints.py",
        )
        solver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solver)
    else:
        from motius.pipelines.gentrack import pyroki_retarget as solver

    urdf = yourdfpy.URDF.load(
        str(args.assets / "urdf/for_retargeting/g1.urdf"),
        mesh_dir=str(args.assets / "mesh/G1"),
    )
    robot = pk.Robot.from_urdf(urdf)
    solver.G1_LINK_NAMES = list(robot.links.names)
    human, indices = solver.get_humanoid_retarget_indices()
    solver.human_retarget_names, solver.g1_joint_retarget_indices = human, indices
    frames = 4
    joints = jnp.zeros((frames, robot.joints.num_actuated_joints))
    joints = joints + 0.008 * jnp.sin(
        jnp.arange(joints.size).reshape(joints.shape) * 0.13
    )
    transforms = jaxlie.SE3(robot.forward_kinematics(cfg=joints))
    positions = transforms.translation()[:, indices]
    rotations = transforms.rotation().as_matrix()[:, indices]
    root = jnp.stack(
        [jnp.arange(frames) * 0.003, jnp.zeros(frames), jnp.ones(frames) * 0.8], axis=-1
    )
    positions = positions + root[:, None]
    auxiliary = positions[:, jnp.array([13, 14, 0])] + jnp.array([0.05, 0.02, 0.01])
    positions = jnp.concatenate([positions, auxiliary], axis=1)
    rotations = jnp.concatenate(
        [rotations, rotations[:, jnp.array([13, 14, 0])]], axis=1
    )
    mask = jnp.zeros((len(indices), len(indices)))
    for left, right, weight in solver.direct_pairs:
        a, b = human.index(left), human.index(right)
        mask = mask.at[a, b].set(weight).at[b, a].set(weight)
    weights = solver.RetargetingWeights(
        local_alignment=1.0,
        global_alignment=4.0,
        root_smoothness=1.0,
        joint_smoothness=4.0,
        self_collision=0.0,
        joint_rest_penalty=1.0,
        joint_vel_limit=50.0,
        foot_contact=30.0,
        foot_tilt=1.0,
    )
    contact = jnp.zeros((frames, 1))
    root, joints = solver.solve_retargeting(
        robot=robot,
        robot_coll=None,
        target_keypoints=positions,
        target_orientations=rotations,
        left_foot_contact=contact,
        right_foot_contact=contact,
        g1_joint_retarget_indices=indices,
        g1_retarget_mask=mask,
        weights=weights,
        subsample_factor=1,
        input_fps=30.0,
    )
    np.savez(
        args.output_dir / (args.worker + ".npz"),
        root=np.asarray(root.wxyz_xyz),
        joints=np.asarray(joints),
        fk=np.asarray(robot.forward_kinematics(cfg=joints)),
    )
    (args.output_dir / (args.worker + "-runtime.json")).write_text(
        json.dumps(
            dict(jax=jax.__version__, backend=jax.default_backend(), frames=frames),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "assets", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "assets", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    report = dict(
        complete=False,
        scope="Actual pinned PyRoki G1 robot and complete original ProtoMotions trajectory least-squares solver versus native workflow. Four nonconstant frames, original objectives/initialization/default 800-iteration termination, CPU JAX backend. No SMPL frontend or full-dataset retargeting claim.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--upstream",
                str(args.upstream),
                "--assets",
                str(args.assets),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "JAX_PLATFORMS": "cpu",
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "LD_LIBRARY_PATH": str(Path(sys.prefix) / "lib/python3.12/site-packages/torch/lib"),
            },
            check=True,
        )
    import numpy as np

    original, native = [
        np.load(args.output_dir / (kind + ".npz")) for kind in ["upstream", "native"]
    ]
    for name in original.files:
        np.testing.assert_array_equal(original[name], native[name], err_msg=name)
        report["tests"].append(
            dict(tensor=name, shape=list(original[name].shape), exact=True)
        )
    report.update(
        complete=True,
        runtime=json.loads((args.output_dir / "native-runtime.json").read_text()),
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Original/native complete G1 trajectory solver exact")


if __name__ == "__main__":
    main()
