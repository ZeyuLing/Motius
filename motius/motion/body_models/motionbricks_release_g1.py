# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"G1 forward kinematics and conversion between joint coordinates and motion features."

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def wrap(value):
    """Wrap angles in radians to [-pi, pi)."""
    return (value + np.pi) % (2 * np.pi) - np.pi


def quat_matrix(q):
    """Convert WXYZ quaternions to rotation matrices."""
    r, i, j, k = q.unbind(-1)
    s = 2 / (q * q).sum(-1)
    return torch.stack(
        (
            1 - s * (j * j + k * k),
            s * (i * j - k * r),
            s * (i * k + j * r),
            s * (i * j + k * r),
            1 - s * (i * i + k * k),
            s * (j * k - i * r),
            s * (i * k - j * r),
            s * (j * k + i * r),
            1 - s * (i * i + j * j),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_quat(m):
    """Convert rotation matrices to WXYZ quaternions."""
    m00, m01, m02 = (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2])
    m10, m11, m12 = (m[..., 1, 0], m[..., 1, 1], m[..., 1, 2])
    m20, m21, m22 = (m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])
    a = (
        torch.stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            -1,
        )
        .clamp_min(0)
        .sqrt()
    )
    rows = torch.stack(
        (
            torch.stack((a[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, a[..., 1] ** 2, m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, a[..., 2] ** 2, m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, a[..., 3] ** 2), -1),
        ),
        -2,
    ) / (2 * a[..., :, None].clamp_min(0.1))
    indices = a.argmax(-1)[..., None, None].expand(a.shape[:-1] + (1, 4))
    return rows.gather(-2, indices).squeeze(-2)


def rotate(m, p):
    """Rotate points by matrices."""
    return (m @ p[..., None])[..., 0]


def six_matrix(value):
    """Convert 6D rotations to rotation matrices by orthonormalizing their two vectors."""
    x = value[..., :3] / value[..., :3].norm(dim=-1, keepdim=True)
    z = torch.cross(x, value[..., 3:], dim=-1)
    z = z / z.norm(dim=-1, keepdim=True)
    return torch.stack((x, torch.cross(z, x, dim=-1), z), dim=-1)


class G1Rig(nn.Module):
    """G1 forward kinematics, and decoding of motion features into joint coordinates."""

    def __init__(self, arrays, indices):
        super().__init__()
        for key, value in {
            "basis": arrays["mujoco_to_motion_matrix"],
            "axes": arrays["mujoco_joint_axis_values_mujoco_space"],
            "model_axes": arrays["mujoco_joint_axis_values_motion_space"],
            "q2t": arrays["rot_offsets_q2t"],
            "f2q": arrays["rot_offsets_f2q"],
            "rest": arrays["capture_neutral_joints_mujoco"],
            "neutral": arrays["neutral_joints"],
            "horizontal_robot": np.array([1, 1, 0], np.float32),
            "horizontal_model": np.array([1, 0, 1], np.float32),
            "identity": np.eye(3, dtype=np.float32),
        }.items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))
        self.parents = arrays["mujoco_joint_including_root_parent_list"].tolist()
        self.model_parents = arrays["joint_parents"].tolist()
        self.reverse = arrays["motion_indices_to_mujoco_indices"].tolist()
        self.register_buffer(
            "mapping",
            torch.tensor(arrays["mujoco_indices_to_motion_indices"], dtype=torch.long),
        )
        self.register_buffer(
            "rotation_indices",
            torch.tensor(indices["global_rot_data"], dtype=torch.long),
        )
        self.register_buffer(
            "root_indices", torch.tensor(indices["global_root_pos"], dtype=torch.long)
        )
        self.register_buffer(
            "contact_indices", torch.tensor(indices["foot_contacts"], dtype=torch.long)
        )
        self.register_buffer(
            "parent_indices", torch.tensor(self.model_parents, dtype=torch.long)
        )

    def fk(self, qpos):
        """Compute joint positions and rotations from joint coordinates."""
        dofs = qpos[..., 7:]
        quats = torch.cat(
            (
                torch.cos(dofs / 2)[..., None],
                torch.sin(dofs / 2)[..., None] * self.axes,
            ),
            -1,
        )
        local = self.q2t[self.mapping] @ quat_matrix(quats)
        rotations = [quat_matrix(qpos[..., 3:7])]
        positions = [qpos[..., :3] + self.rest[0]]
        for joint in range(1, 30):
            parent = self.parents[joint]
            positions.append(
                positions[parent]
                + rotate(rotations[parent], self.rest[joint] - self.rest[parent])
            )
            rotations.append(rotations[parent] @ local[..., joint - 1, :, :])
        p = [rotate(self.basis, value) for value in positions]
        r = [self.basis @ value @ self.basis.T for value in rotations]
        model_p, model_r = ([], [])
        for joint, robot_index in enumerate(self.reverse):
            if robot_index >= 0:
                model_p.append(p[robot_index])
                model_r.append(r[robot_index])
            else:
                parent = self.model_parents[joint]
                model_p.append(
                    model_p[parent]
                    + rotate(
                        model_r[parent], self.neutral[joint] - self.neutral[parent]
                    )
                )
                model_r.append(self.identity.expand_as(model_r[parent]))
        return (torch.stack(model_p, -2), torch.stack(model_r, -3))

    def decode(self, features):
        """Decode motion features into the root position, rotation, and joint angles."""
        r = six_matrix(
            features[..., self.rotation_indices].reshape(features.shape[0], 64, 34, 6)
        )
        parents = torch.cat(
            (
                self.identity.expand_as(r[..., :1, :, :]),
                r[..., self.parent_indices[1:], :, :],
            ),
            -3,
        )
        local = self.f2q @ (parents.transpose(-2, -1) @ r)
        root_p = rotate(self.basis.T, features[..., self.root_indices])
        root_r = self.basis.T @ local[..., 0, :, :] @ self.basis
        hinge = local[..., self.mapping, :, :]
        angles = torch.stack(
            (
                torch.atan2(hinge[..., 2, 1], hinge[..., 2, 2]),
                torch.atan2(hinge[..., 0, 2], hinge[..., 0, 0]),
                torch.atan2(hinge[..., 1, 0], hinge[..., 1, 1]),
            ),
            -1,
        )
        return (root_p, root_r, (angles * self.model_axes).sum(-1))


class G1PlacedPosePreprocess(G1Rig):
    """Encode placed G1 context and target poses into the model's root and pose inputs."""

    def forward(
        self,
        context_qpos,
        target_qpos,
        has_global_root_values,
        has_local_root_values,
        has_local_poses,
        allowed_pred_num_tokens,
    ):
        poses = torch.cat((context_qpos, target_qpos), 1)
        poses = torch.cat(
            (poses[..., :3], poses[..., [6, 3, 4, 5]], poses[..., 7:]), -1
        )
        p, r = self.fk(poses)
        root = p[:, :, 0]
        heading = torch.atan2(r[:, :, 0, 0, 2], r[:, :, 0, 2, 2])
        global_root = torch.cat(
            (root, heading.cos()[..., None], heading.sin()[..., None]), -1
        )
        windows = []
        for start in (0, 4):
            yaw_vel = (
                wrap(heading[:, start + 1 : start + 4] - heading[:, start : start + 3])
                * 30
            )
            velocity = (
                root[:, start + 1 : start + 4, [0, 2]]
                - root[:, start : start + 3, [0, 2]]
            ) * 30
            first = torch.cat(
                (yaw_vel[..., None], velocity, root[:, start : start + 3, 1:2]), -1
            )
            last = torch.cat(
                (
                    torch.zeros_like(first[:, :1, :3]),
                    root[:, start + 3 : start + 4, 1:2],
                ),
                -1,
            )
            windows.append(torch.cat((first, last), 1))
        nonroot = p[:, :, 1:] - root[:, :, None] * self.horizontal_model
        six = torch.cat((r[..., :, 0], r[..., :, 1]), -1)
        local_poses = torch.cat(
            (
                nonroot.reshape(nonroot.shape[0], 8, 99),
                six.reshape(six.shape[0], 8, 204),
            ),
            -1,
        )
        return (
            global_root,
            has_global_root_values,
            torch.cat(windows, 1),
            has_local_root_values,
            local_poses,
            has_local_poses,
            allowed_pred_num_tokens,
        )


class G1PlacedPosePostprocess(G1Rig):
    """Decode motion features into joint coordinates, token counts, and foot contacts."""

    def forward(self, pred_global_motions, pred_num_tokens):
        root_p, root_r, hinges = self.decode(pred_global_motions)
        quaternion = matrix_quat(root_r)[..., [1, 2, 3, 0]]
        return (
            torch.cat((root_p, quaternion, hinges), -1),
            pred_num_tokens,
            pred_global_motions[..., self.contact_indices],
        )
