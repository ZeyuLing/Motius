# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Generate SOMA motion with the native Motius PyTorch pipeline."""

from __future__ import annotations

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime import MotionBricksInferenceConfig
from motius.configuration.motionbricks_runtime_contract import (
    MODEL_CONTEXT_AND_TARGET_POSES,
    MODEL_MAX_VALID_FRAMES,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
    SOMA_N1_CONTRACT,
    SOMA_N1_HEADING_JOINTS,
)
from motius.motion.runtime.motionbricks import soma_kernels as kernels
from motius.motion.runtime.motionbricks.animation_buffer import AnimationBuffer
from motius.motion.runtime.motionbricks.foot_contacts import FOOT_CONTACT_COUNT
from motius.motion.runtime.motionbricks.pose_utils import compute_world_poses
from motius.motion.runtime.motionbricks.soma_buffers import PoseToken
from motius.pipelines.motionbricks.runtime_engine import MotionBricksEngine
from motius.pipelines.motionbricks.runtime_generation import (
    check_generation_buffers,
    write_conditions,
)


class MotionBricksSOMAEngine(MotionBricksEngine):
    """Generate SOMA motion from recent motion and target poses.

    Each pose holds one local transform per skeleton joint. The root joint is
    Hips.
    """

    contract = SOMA_N1_CONTRACT

    def _initialize_pose_layout(self) -> None:
        self._pose_skeleton = None
        self._model_to_pose = None
        self._pose_to_model = None
        self._guide_world = None
        self._start_deltas = None
        self._end_deltas = None
        self._right_leg_index = -1
        self._left_leg_index = -1

    def create_input(self, skeleton) -> PoseToken:
        """Create a context or target token with room for four poses.

        Args:
            skeleton: Skeleton of the poses. It must contain the model's joint names.
                Missing joints raise ValueError.
        """
        self._ensure_open()
        self._bind_pose_skeleton(skeleton)
        return PoseToken(skeleton.num_joints, device=self.device)

    def create_output(self, skeleton) -> AnimationBuffer:
        """Create an empty animation buffer with room for the longest generated motion.

        Args:
            skeleton: Skeleton of the generated motion. It must contain the model's joint
                names. Missing joints raise ValueError.
        """
        self._ensure_open()
        self._bind_pose_skeleton(skeleton)
        return AnimationBuffer(
            skeleton,
            MODEL_SAMPLE_RATE,
            wp.empty(
                (MODEL_MAX_VALID_FRAMES, skeleton.num_joints),
                dtype=wp.transform,
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
        self, context: PoseToken, target: PoseToken, output: AnimationBuffer
    ) -> None:
        self._bind_pose_skeleton(output.skeleton)
        num_joints = output.skeleton.num_joints
        for name, token in (("Context", context), ("Target", target)):
            if token.num_joints != num_joints:
                raise ValueError(
                    f"{name} token must contain {num_joints} joints per pose."
                )
        check_generation_buffers(context, target, output)

    def _bind_pose_skeleton(self, skeleton) -> None:
        if skeleton is self._pose_skeleton:
            return
        model_to_pose = [
            skeleton.joint_index(name) for name in self.contract.joint_names
        ]
        if -1 in model_to_pose:
            raise ValueError(
                "The skeleton is missing model joints: "
                + ", ".join(
                    (
                        name
                        for name, index in zip(self.contract.joint_names, model_to_pose)
                        if index == -1
                    )
                )
            )
        assert model_to_pose[0] == 0, "The skeleton must store the root at index zero."
        pose_to_model = np.full(skeleton.num_joints, -1, dtype=np.int32)
        for model_joint, pose_joint in enumerate(model_to_pose):
            parent = skeleton.joint_parent(pose_joint)
            if parent == -1 or pose_to_model[parent] >= 0:
                pose_to_model[pose_joint] = model_joint
        self._pose_skeleton = None
        right_leg, left_leg = SOMA_N1_HEADING_JOINTS
        self._right_leg_index = skeleton.joint_index(right_leg)
        self._left_leg_index = skeleton.joint_index(left_leg)
        self._model_to_pose = wp.array(
            model_to_pose, dtype=wp.int32, device=self.device
        )
        self._pose_to_model = wp.array(
            pose_to_model, dtype=wp.int32, device=self.device
        )
        self._guide_world = wp.empty(
            (MODEL_CONTEXT_AND_TARGET_POSES, skeleton.num_joints),
            dtype=wp.transform,
            device=self.device,
        )
        self._start_deltas = wp.empty(
            skeleton.num_joints, dtype=wp.transform, device=self.device
        )
        self._end_deltas = wp.empty(
            skeleton.num_joints, dtype=wp.transform, device=self.device
        )
        self._pose_skeleton = skeleton

    def _encode_guide_poses(
        self, context: PoseToken, target: PoseToken, config: MotionBricksInferenceConfig
    ) -> None:
        for token, rows in (
            (context, self._guide_world[:MODEL_POSES_PER_TOKEN]),
            (target, self._guide_world[MODEL_POSES_PER_TOKEN:]),
        ):
            compute_world_poses(
                token.poses,
                self._pose_skeleton.parent_indices,
                rows,
                self._runtime_to_model,
                stream=self.stream,
            )
        wp.launch(
            kernels.compute_guide_alignment_kernel,
            dim=1,
            inputs=[
                self._right_leg_index,
                self._left_leg_index,
                wp.vec2(*config.heading_alignment),
                self._guide_world,
            ],
            outputs=[self._alignment],
            stream=self.stream,
        )
        wp.launch(
            kernels.encode_motionbricks_inputs_kernel,
            dim=MODEL_CONTEXT_AND_TARGET_POSES,
            inputs=[
                len(self.contract.joint_names),
                self._right_leg_index,
                self._left_leg_index,
                self._model_to_pose,
                self._guide_world,
                self._alignment,
            ],
            outputs=[
                self.device_tensor("global_root_values"),
                self.device_tensor("local_root_values"),
                self.device_tensor("local_poses"),
            ],
            stream=self.stream,
        )
        write_conditions(self, target.num_poses, config)

    def _decode_generated_poses(self, output: AnimationBuffer) -> None:
        wp.launch(
            kernels.decode_motionbricks_output_kernel,
            dim=MODEL_MAX_VALID_FRAMES,
            inputs=[
                len(self.contract.joint_names),
                self._pose_skeleton.num_joints,
                self._pose_skeleton.parent_indices,
                self._pose_to_model,
                self._pose_skeleton.reference_local_transforms,
                self._alignment,
                self._model_to_runtime,
                self.device_tensor("pred_global_motions"),
                self.device_tensor("pred_num_tokens"),
            ],
            outputs=[output.poses, output.foot_contacts, self._generated_frame_count],
            stream=self.stream,
        )

    def _post_process_generated_poses(
        self, context: PoseToken, target: PoseToken, output: AnimationBuffer
    ) -> None:
        """Match the generated motion's endpoints to the last context pose and first target pose.

        The corrections span the generated motion.
        """
        wp.launch(
            kernels.compute_endpoint_deltas_kernel,
            dim=self._pose_skeleton.num_joints,
            inputs=[
                self._generated_frame_count,
                context.poses,
                target.poses,
                output.poses,
            ],
            outputs=[self._start_deltas, self._end_deltas],
            stream=self.stream,
        )
        wp.launch(
            kernels.apply_endpoint_deltas_kernel,
            dim=(MODEL_MAX_VALID_FRAMES, self._pose_skeleton.num_joints),
            inputs=[self._generated_frame_count, self._start_deltas, self._end_deltas],
            outputs=[output.poses],
            stream=self.stream,
        )
