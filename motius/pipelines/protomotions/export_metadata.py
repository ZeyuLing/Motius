# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Source-compatible deployment contract metadata, without runtime imports."""


def build_deployment_metadata(
    *,
    onnx_in_names,
    onnx_out_names,
    onnx_name_to_key,
    input_shapes,
    obs_input_keys,
    actor_obs_configs,
    joint_names,
    body_names,
    stiffness,
    damping,
    effort_limits,
    pd_target_max_accel,
    anchor_body_name,
    anchor_body_index,
    root_body_index,
    num_bodies,
    num_dofs,
    mjcf_path,
    control_dt,
    physics_dt,
    decimation,
    future_step_indices,
    checkpoint,
    control_type,
) -> dict:
    """Build the rich YAML metadata dict."""

    # Build per-input descriptors with element names
    policy_inputs = []
    for onnx_name in onnx_in_names:
        key = onnx_name_to_key.get(onnx_name, onnx_name)
        shape = input_shapes.get(key, "unknown")
        entry: dict = {
            "name": onnx_name,
            "key": key,
            "shape": shape,
        }

        # Determine kind and element names
        if "dof_pos" in key and "future" not in key:
            entry["kind"] = "joint_pos"
            entry["element_names"] = [joint_names]
        elif "dof_vel" in key and "future" not in key:
            entry["kind"] = "joint_vel"
            entry["element_names"] = [joint_names]
        elif "anchor_rot" in key and "future" not in key and "mimic" not in key:
            entry["kind"] = "anchor_rot"
            entry["element_names"] = [["x", "y", "z", "w"]]
        elif "root_local_ang_vel" in key:
            entry["kind"] = "local_root_ang_vel"
        elif "processed_actions" in key or "historical" in key:
            entry["kind"] = "last_actions"
            # The ONNX output key this feeds back from
            entry["output_key"] = "robot_action"
        elif "mimic" in key and "anchor_rot" in key:
            entry["kind"] = "reference_motion_body_rot"
            entry["future_steps"] = len(future_step_indices)
            entry["element_names"] = [[anchor_body_name], ["x", "y", "z", "w"]]
        elif "mimic" in key and "dof_pos" in key:
            entry["kind"] = "reference_motion_joint_pos"
            entry["future_steps"] = len(future_step_indices)
            entry["element_names"] = [joint_names]
        elif "mimic" in key and "dof_vel" in key:
            entry["kind"] = "reference_motion_joint_vel"
            entry["future_steps"] = len(future_step_indices)
            entry["element_names"] = [joint_names]

        policy_inputs.append(entry)

    # Build output descriptors
    policy_outputs = [
        {
            "name": "actions",
            "kind": "actions",
            "key": "actions",
            "shape": [1, num_dofs],
        },
        {
            "name": "joint_pos_targets",
            "kind": "joint_pos_targets",
            "key": "joint_pos_targets",
            "shape": [1, num_dofs],
            "joint_names": joint_names,
        },
        {
            "name": "stiffness_targets",
            "kind": "stiffness_targets",
            "key": "stiffness_targets",
            "shape": [1, num_dofs],
            "joint_names": joint_names,
        },
        {
            "name": "damping_targets",
            "kind": "damping_targets",
            "key": "damping_targets",
            "shape": [1, num_dofs],
            "joint_names": joint_names,
        },
    ]

    # Passthrough keys: obs_input_keys that aren't ONNX inputs
    # (resolved to constants during tracing)
    passthrough_keys = [k for k in obs_input_keys if k not in onnx_name_to_key.values()]

    content = {
        "type": "unified_pipeline",
        "dt": control_dt,
        "joint_names": joint_names,
        "body_names": body_names,
        "default_joint_stiffness": stiffness,
        "default_joint_damping": damping,
        "policy_inputs": policy_inputs,
        "policy_outputs": policy_outputs,
        "_runtime": {
            "onnx_in_names": onnx_in_names,
            "onnx_out_names": onnx_out_names,
            "onnx_name_to_in_key": onnx_name_to_key,
            "passthrough_keys": passthrough_keys,
            "obs_context_keys": obs_input_keys,
        },
        "metadata": {
            "checkpoint": checkpoint,
            "control_type": control_type,
        },
        # Full deployment contract fields (for robojudo integration)
        "robot": {
            "mjcf_path": mjcf_path,
            "num_bodies": num_bodies,
            "num_dofs": num_dofs,
            "anchor_body_name": anchor_body_name,
            "anchor_body_index": anchor_body_index,
            "root_body_name": body_names[root_body_index],
            "root_body_index": root_body_index,
            "body_names": body_names,
            "joint_names": joint_names,
        },
        "control": {
            "stiffness": stiffness,
            "damping": damping,
            "effort_limits": effort_limits,
            "pd_target_max_accel": pd_target_max_accel,
            "action_ema_alpha": 1.0,
        },
        "timing": {
            "control_dt": control_dt,
            "physics_dt": physics_dt,
            "decimation": decimation,
        },
        "motion": {
            "future_step_indices": future_step_indices,
            "future_dt_seconds": [
                round(s * control_dt, 6) for s in future_step_indices
            ],
        },
    }
    return content
