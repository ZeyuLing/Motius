"""HY-Motion wooden character geometry, outside neural model bundles.

Linear blend skinning preserves Tencent HY-Motion's operation order and its
untranslated posed joints. Only the five inference geometry assets are needed.
See motius/licenses/hymotion_t2m/UPSTREAM_LICENSE.md.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from motius.motion.representation.hymotion_runtime.geometry import (
    rot6d_to_rotation_matrix,
)


def simple_lbs(v_template, rot_mats, joints, parents, skin_weights, skin_indices):
    """
    Simple Linear Blend Skinning without shape blending.

    Args:
        v_template: (V, 3) template vertices
        rot_mats: (B, J, 3, 3) rotation matrices for each joint
        joints: (J, 3) joint positions in rest pose
        parents: (J,) parent indices for each joint
        skin_weights: (V, 4) skin weights for 4 bones per vertex
        skin_indices: (V, 4) bone indices for 4 bones per vertex

    Returns:
        vertices: (B, V, 3) transformed vertices
        posed_joints: (B, J, 3) transformed joint positions
    """
    batch_size = rot_mats.shape[0]
    num_joints = rot_mats.shape[1]
    num_verts = v_template.shape[0]
    device = rot_mats.device
    dtype = rot_mats.dtype

    # Compute relative joint positions
    rel_joints = joints.clone()
    rel_joints[1:] = joints[1:] - joints[parents[1:]]

    # Build transformation chain: transforms_mat (B, J, 4, 4)
    transforms_mat = torch.zeros(
        batch_size, num_joints, 4, 4, device=device, dtype=dtype
    )
    transforms_mat[..., :3, :3] = rot_mats
    transforms_mat[..., :3, 3] = rel_joints.unsqueeze(0).expand(batch_size, -1, -1)
    transforms_mat[..., 3, 3] = 1.0

    # Forward kinematics: accumulate transforms from root to each joint
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        parent_idx = parents[i].item()
        curr_transform = torch.bmm(transform_chain[parent_idx], transforms_mat[:, i])
        transform_chain.append(curr_transform)

    transforms = torch.stack(transform_chain, dim=1)  # (B, J, 4, 4)

    # Get posed joint positions
    posed_joints = transforms[..., :3, 3].clone()  # (B, J, 3)

    # Compute relative transforms (for skinning)
    # We need to subtract the rest pose joint positions from the transform
    rel_transforms = transforms.clone()
    joints_homo = F.pad(joints, [0, 1], value=0)  # (J, 4)
    transformed_rest = torch.einsum("bjcd,jd->bjc", transforms[..., :3, :], joints_homo)
    rel_transforms[..., :3, 3] = transforms[..., :3, 3] - transformed_rest[..., :3]

    # Apply skinning: gather transforms for each vertex's 4 bones
    # skin_indices: (V, 4), skin_weights: (V, 4)
    vertex_transforms = torch.zeros(
        batch_size, num_verts, 4, 4, 4, device=device, dtype=dtype
    )
    for k in range(4):
        bone_idx = skin_indices[:, k].long()  # (V,)
        vertex_transforms[:, :, k] = rel_transforms[:, bone_idx]  # (B, V, 4, 4)

    # Weight the transforms
    skin_weights_expanded = (
        skin_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )  # (1, V, 4, 1, 1)
    skin_weights_expanded = skin_weights_expanded.expand(
        batch_size, -1, -1, 4, 4
    )  # (B, V, 4, 4, 4)

    weighted_transforms = (vertex_transforms * skin_weights_expanded).sum(
        dim=2
    )  # (B, V, 4, 4)

    # Apply to vertices
    v_homo = F.pad(v_template, [0, 1], value=1.0)  # (V, 4)
    vertices = torch.einsum(
        "bvcd,vd->bvc", weighted_transforms[..., :3, :], v_homo
    )  # (B, V, 3)

    return vertices, posed_joints


class HYMotionWoodenBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        directory = Path(__file__).resolve().parents[1] / "assets" / "hymotion_wooden"
        for name, filename, dtype, shape in (
            ("v_template", "v_template.bin", np.float32, (-1, 3)),
            ("j_template", "j_template.bin", np.float32, (-1, 3)),
            ("skin_weights", "skinWeights.bin", np.float32, (-1, 4)),
            ("skin_indices", "skinIndice.bin", np.uint16, (-1, 4)),
            ("parents", "kintree.bin", np.int32, (-1,)),
        ):
            values = (
                np.frombuffer((directory / filename).read_bytes(), dtype=dtype)
                .copy()
                .reshape(shape)
            )
            target_dtype = (
                torch.long if name in {"skin_indices", "parents"} else torch.float32
            )
            self.register_buffer(name, torch.tensor(values, dtype=target_dtype))

    def forward(self, rot6d, translation):
        frames = rot6d.shape[0]
        rotations = rot6d_to_rotation_matrix(rot6d).view(frames, -1, 3, 3)
        if rotations.shape[1] == 22:
            identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)[
                None, None
            ].repeat(frames, 30, 1, 1)
            rotations = torch.cat([rotations, identity], dim=1)
        vertices, joints = simple_lbs(
            self.v_template,
            rotations,
            self.j_template,
            self.parents,
            self.skin_weights,
            self.skin_indices,
        )
        return vertices + translation[:, None, :], joints
