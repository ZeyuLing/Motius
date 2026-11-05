# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Generate Unitree G1 motion with the native Motius PyTorch pipeline."""

from __future__ import annotations

import warp as wp

from motius.configuration.motionbricks_runtime import MotionBricksInferenceConfig
from motius.configuration.motionbricks_runtime_contract import (
    MODEL_MAX_VALID_FRAMES,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
    UNITREE_G1_N1_CONTRACT,
)
from motius.motion.runtime.motionbricks import unitree_g1_kernels as kernels
from motius.motion.runtime.motionbricks.dof_animation_buffer import DOFAnimationBuffer
from motius.motion.runtime.motionbricks.dof_utils import ROOT_COORDINATE_COUNT
from motius.motion.runtime.motionbricks.foot_contacts import FOOT_CONTACT_COUNT
from motius.motion.runtime.motionbricks.unitree_g1_buffers import DOFToken
from motius.pipelines.motionbricks.runtime_engine import MotionBricksEngine
from motius.pipelines.motionbricks.runtime_generation import (
    check_generation_buffers,
    write_conditions,
)


class MotionBricksUnitreeG1Engine(MotionBricksEngine):
    """Generate G1 motion from recent motion and target poses.

    Each pose contains 36 joint coordinates: root position, an XYZW quaternion,
    and 29 joint angles. The root joint is the pelvis. Positions are in meters.
    Joint angles are in radians. TF32 stays off unless the build options enable it,
    because it moves joint angles by several degrees for under 2 percent faster
    inference.
    """

    contract = UNITREE_G1_N1_CONTRACT
    default_enable_tf32 = False

    def _initialize_pose_layout(self) -> None:
        self._root_position_filter = wp.array(
            kernels.ROOT_POSITION_FILTER, dtype=wp.float32, device=self.device
        )
        self._num_coordinates = ROOT_COORDINATE_COUNT + len(self.contract.joint_names)
        self._endpoint_deltas = wp.empty(
            (2, self._num_coordinates), dtype=wp.float32, device=self.device
        )

    def create_input(self) -> DOFToken:
        """Create a context or target token with room for four poses."""
        self._ensure_open()
        return DOFToken(len(self.contract.joint_names), device=self.device)

    def create_output(self) -> DOFAnimationBuffer:
        """Create an empty animation buffer with room for the longest generated motion."""
        self._ensure_open()
        return DOFAnimationBuffer(
            self.contract.joint_names,
            MODEL_SAMPLE_RATE,
            wp.empty(
                (MODEL_MAX_VALID_FRAMES, self._num_coordinates),
                dtype=wp.float32,
                device=self.device,
            ),
            num_frames=0,
            foot_contacts=wp.zeros(
                (MODEL_MAX_VALID_FRAMES, FOOT_CONTACT_COUNT),
                dtype=wp.float32,
                device=self.device,
            ),
        )

    def _prepare_generation(
        self, context: DOFToken, target: DOFToken, output: DOFAnimationBuffer
    ) -> None:
        num_dofs = len(self.contract.joint_names)
        for name, token in (("Context", context), ("Target", target)):
            if token.num_dofs != num_dofs:
                raise ValueError(
                    f"{name} token must contain {num_dofs} joint angles per pose."
                )
        if output.num_dofs != num_dofs:
            raise ValueError(f"Output must contain {num_dofs} joint angles per frame.")
        check_generation_buffers(context, target, output)

    def _encode_guide_poses(
        self, context: DOFToken, target: DOFToken, config: MotionBricksInferenceConfig
    ) -> None:
        wp.launch(
            kernels.compute_guide_alignment_kernel,
            dim=1,
            inputs=[
                wp.vec2(*config.heading_alignment),
                self._runtime_to_model,
                context.poses,
            ],
            outputs=[self._alignment],
            stream=self.stream,
        )
        for token, tensor_name in ((context, "context_qpos"), (target, "target_qpos")):
            wp.launch(
                kernels.encode_motionbricks_inputs_kernel,
                dim=MODEL_POSES_PER_TOKEN,
                inputs=[self._runtime_to_model, token.poses, self._alignment],
                outputs=[self.device_tensor(tensor_name)[0]],
                stream=self.stream,
            )
        write_conditions(self, target.num_poses, config)

    def _decode_generated_poses(self, output: DOFAnimationBuffer) -> None:
        wp.launch(
            kernels.decode_motionbricks_output_kernel,
            dim=MODEL_MAX_VALID_FRAMES,
            inputs=[
                self._alignment,
                self._model_to_runtime,
                self.device_tensor("pred_qpos")[0],
                self.device_tensor("pred_foot_contacts")[0],
                self.device_tensor("pred_num_tokens"),
                self._root_position_filter,
            ],
            outputs=[output.poses, output.foot_contacts, self._generated_frame_count],
            stream=self.stream,
        )

    def _post_process_generated_poses(
        self, context: DOFToken, target: DOFToken, output: DOFAnimationBuffer
    ) -> None:
        """Match the generated motion's endpoints to the last context pose and first target pose.

        Root corrections span the generated motion. Joint angle corrections fade over
        JOINT_ANGLE_FADE_SECONDS at each end with a quadratic ease in and out.
        """
        wp.launch(
            kernels.compute_endpoint_deltas_kernel,
            dim=2,
            inputs=[
                self._generated_frame_count,
                context.poses,
                target.poses,
                output.poses,
            ],
            outputs=[self._endpoint_deltas],
            stream=self.stream,
        )
        wp.launch(
            kernels.apply_endpoint_deltas_kernel,
            dim=MODEL_MAX_VALID_FRAMES,
            inputs=[
                self._generated_frame_count,
                self._endpoint_deltas,
                kernels.JOINT_ANGLE_FADE_SECONDS * MODEL_SAMPLE_RATE,
            ],
            outputs=[output.poses],
            stream=self.stream,
        )
