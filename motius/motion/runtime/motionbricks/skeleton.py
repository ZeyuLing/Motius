# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Describe a joint hierarchy and its reference pose."""

import numpy as np
import warp as wp

from motius.motion.runtime.motionbricks import pose_utils


class Skeleton:
    """Joint hierarchy and reference pose."""

    def __init__(self, joint_names, parent_indices, local_transforms) -> None:
        """Create a skeleton from its hierarchy and reference pose.

        Args:
            joint_names: Name of each joint. There must be at least one.
            parent_indices: Parent index of each joint, or -1 for the root.
            local_transforms: Reference pose, as one local transform per joint.
        """
        self.joint_names = list(joint_names)
        self._num_joints = len(self.joint_names)
        if self._num_joints == 0:
            raise ValueError("A skeleton needs at least one joint.")
        self._np_parent_indices = np.asarray(parent_indices, dtype=np.int32)
        if self._np_parent_indices.shape != (self._num_joints,):
            raise ValueError(
                f"Expected {self._num_joints} parent indices, got {self._np_parent_indices.size}."
            )
        if len(local_transforms) != self._num_joints:
            raise ValueError(
                f"Expected {self._num_joints} local transforms, got {len(local_transforms)}."
            )
        self.parent_indices = wp.array(self._np_parent_indices, dtype=wp.int32)
        self._reference_local_transforms = wp.array(
            local_transforms, dtype=wp.transform
        )
        self.up_axis = wp.vec3(0.0, 0.0, 1.0)
        self.forward_axis = wp.vec3(0.0, -1.0, 0.0)

    @property
    def num_joints(self) -> int:
        """Number of joints in the hierarchy."""
        return self._num_joints

    @property
    def device(self):
        """Device of the skeleton's arrays."""
        return self.parent_indices.device

    @property
    def reference_local_transforms(self) -> wp.array:
        """Reference pose, as one local joint transform per joint."""
        return self._reference_local_transforms

    def joint_index(self, joint_name: str) -> int:
        """Return the index of a joint, or -1 when no joint has that name.

        Args:
            joint_name: Name of the joint.
        """
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            return -1

    def joint_name(self, index: int) -> str:
        """Return the name of a joint.

        Args:
            index: Joint index. An index out of range raises ValueError.
        """
        if not 0 <= index < self._num_joints:
            raise ValueError(f"Joint index {index} must be in [0, {self._num_joints}).")
        return self.joint_names[index]

    def joint_parent(self, index: int) -> int:
        """Return the parent index of a joint, or -1 for the root.

        Args:
            index: Joint index. An index out of range raises ValueError.
        """
        if not 0 <= index < self._num_joints:
            raise ValueError(f"Joint index {index} must be in [0, {self._num_joints}).")
        return int(self._np_parent_indices[index])

    def joint_mask(self, joint_names, weight: float = 1.0) -> wp.array:
        """Return one weight per joint: weight on the named joints, zero elsewhere.

        Args:
            joint_names: Joints that get the weight. Names the skeleton lacks are skipped.
            weight: Weight of the named joints, clamped to 0 to 1.
        """
        weight = wp.clamp(float(weight), 0.0, 1.0)
        weights = np.zeros(self._num_joints, dtype=np.float32)
        for joint_name in joint_names:
            index = self.joint_index(joint_name)
            if index >= 0:
                weights[index] = weight
        return wp.array(weights, dtype=wp.float32, device=self.device)

    def compute_world_transforms(
        self,
        local_transforms: wp.array,
        world_transforms: wp.array,
        root_tx=wp.transform_identity(),
        stream: wp.Stream | None = None,
    ) -> None:
        """Compute the world transforms of a pose into world_transforms.

        Args:
            local_transforms: Local joint transforms of one pose.
            world_transforms: Array that receives one world transform per joint.
            root_tx: Transform applied to the root. The default leaves the root
                unchanged.
            stream: Stream for the work. None uses the current stream.
        """
        pose_utils.compute_world_poses(
            local_transforms,
            self.parent_indices,
            world_transforms,
            root_tx,
            stream=stream,
        )
