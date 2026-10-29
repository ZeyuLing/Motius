"""Export native observation/policy/action composition for deployment clients."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from motius.motion.representation import protomotions as representation
from .export_metadata import build_deployment_metadata


class TrackerDeployment(nn.Module):
    def __init__(self, bundle, deployment):
        super().__init__()
        self.actor = bundle.actor
        self.deployment = deployment
        self.paths = sorted(
            {
                path
                for config in deployment["observations"].values()
                for path in config["dynamic_vars"].values()
            }
        )
        self.input_names = [path.replace(".", "_") for path in self.paths]
        for name, value in deployment["action"].items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, *inputs):
        raw = dict(zip(self.paths, inputs))
        prepared = {}
        for key, config in self.deployment["observations"].items():
            parameters = {
                name: value
                for name, value in config.items()
                if name not in {"compute_func", "dynamic_vars"}
            }
            parameters.update(
                {name: raw[path] for name, path in config["dynamic_vars"].items()}
            )
            prepared[key] = getattr(representation, config["compute_func"])(
                **parameters
            )
        actions = self.actor(prepared)["mean"]
        result = representation.bm_pd_action(
            actions, **{name: getattr(self, name) for name in self.deployment["action"]}
        )
        return (
            actions,
            result["processed_action"],
            result["stiffness_targets"],
            result["damping_targets"],
        )


def sample_inputs(module, num_dofs, num_bodies, *, batch=1, history_steps=1):
    future = module.deployment["future_steps"]
    future = len(future) if isinstance(future, list) else int(future)
    output = []
    for path in module.paths:
        if path.startswith("historical."):
            shape = (batch, history_steps, num_dofs)
        elif path.endswith(("dof_pos", "dof_vel")):
            shape = (
                (batch, future, num_dofs)
                if path.startswith("mimic.")
                else (batch, num_dofs)
            )
        elif path.endswith(("future_pos", "future_vel", "future_ang_vel")):
            shape = (batch, future, num_bodies, 3)
        elif path.endswith("future_rot"):
            shape = (batch, future, num_bodies, 4)
        elif "anchor_rot" in path or path.endswith("root_rot"):
            shape = (batch, future, 4) if "future" in path else (batch, 4)
        elif "ground_height" in path:
            shape = (batch, 1)
        elif "anchor_" in path or path.startswith("current."):
            shape = (batch, future, 3) if "future" in path else (batch, 3)
        else:
            raise ValueError("Unsupported G1 deployment input: " + path)
        value = torch.zeros(shape)
        if "rot" in path:
            w_last = next(
                (
                    cfg.get("w_last", True)
                    for cfg in module.deployment["observations"].values()
                    if "w_last" in cfg
                ),
                True,
            )
            value[..., 3 if w_last else 0] = 1
        output.append(value)
    return tuple(output)


def export_tracker(checkpoint, output_dir, *, configuration=None, validate=True):
    from motius.pipelines.protomotions import ProtoMotionsPipeline
    from motius.configuration.protomotions.environment import (
        load_environment_configuration,
    )

    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = Path(
        configuration or checkpoint.parent / "resolved_configs_inference.yaml"
    )
    typed = load_environment_configuration(configuration.with_suffix(".pt"))
    pipeline = ProtoMotionsPipeline.from_original_checkpoint(
        checkpoint, configuration=configuration
    )
    module = TrackerDeployment(pipeline.bundle, pipeline.deployment).eval()
    robot = typed["robot"]
    body_names = list(robot.kinematic_info.body_names)
    joint_names = list(robot.kinematic_info.dof_names)
    history = 1
    for config in pipeline.deployment["observations"].values():
        steps = config.get("history_steps")
        if steps is not None:
            history = max(
                history, max(steps) if isinstance(steps, list) else int(steps)
            )
    inputs = sample_inputs(
        module, len(joint_names), len(body_names), history_steps=history
    )
    outputs = ["actions", "joint_pos_targets", "stiffness_targets", "damping_targets"]
    destination = output_dir / "unified_pipeline.onnx"
    torch.onnx.export(
        module,
        inputs,
        str(destination),
        input_names=module.input_names,
        output_names=outputs,
        dynamic_axes={name: {0: "batch_size"} for name in module.input_names + outputs},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    actual_names = [item.name for item in session.get_inputs()]
    if validate:
        for batch in [1, 3]:
            values = sample_inputs(
                module,
                len(joint_names),
                len(body_names),
                batch=batch,
                history_steps=history,
            )
            with torch.no_grad():
                expected = module(*values)
            actual = session.run(
                outputs,
                {
                    name: value.numpy()
                    for name, value in zip(module.input_names, values)
                    if name in actual_names
                },
            )
            for left, right in zip(expected, actual):
                np.testing.assert_allclose(left.numpy(), right, rtol=2e-5, atol=2e-5)
    # Match the original simulator switch to MuJoCo for physical rollout timing.
    from motius.configuration.protomotions.compatibility_mujoco import (
        MujocoSimParams,
    )

    simulator = typed.get("simulator")
    simulation = None
    if simulator is not None:
        simulation = getattr(robot.simulation_params, "mujoco", MujocoSimParams())
        if simulation is None:
            simulation = getattr(simulator, "sim", None)
    physics_dt, decimation = 0.001, 20
    if simulation is not None:
        fps = getattr(simulation, "fps", None)
        steps = getattr(simulation, "decimation", None)
        if fps and steps:
            physics_dt, decimation = 1 / fps, steps
    control_dt = physics_dt * decimation
    info = robot.control.control_info
    gains = lambda name: [float(getattr(info[joint], name)) for joint in joint_names]
    try:
        effort_limits = gains("effort")
    except (AttributeError, KeyError):
        effort_limits = None
    meta = build_deployment_metadata(
        onnx_in_names=actual_names,
        onnx_out_names=outputs,
        onnx_name_to_key={
            name: path
            for name, path in zip(module.input_names, module.paths)
            if name in actual_names
        },
        input_shapes={
            path: list(value.shape) for path, value in zip(module.paths, inputs)
        },
        obs_input_keys=module.paths,
        actor_obs_configs=typed["env"].observation_components,
        joint_names=joint_names,
        body_names=body_names,
        stiffness=gains("stiffness"),
        damping=gains("damping"),
        effort_limits=effort_limits,
        pd_target_max_accel=getattr(typed["simulator"], "pd_target_max_accel", None),
        anchor_body_name=robot.anchor_body_name,
        anchor_body_index=robot.anchor_body_index,
        root_body_index=0,
        num_bodies=len(body_names),
        num_dofs=len(joint_names),
        mjcf_path=robot.asset.asset_file_name,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        future_step_indices=pipeline.deployment["future_steps"],
        checkpoint=str(checkpoint),
        control_type="BUILT_IN_PD",
    )
    import yaml

    (output_dir / "unified_pipeline.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False)
    )
    return destination
