# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Barycentric interpolation for mesh deformation transfer.

Example usage:
    >>> import torch
    >>> from soma.geometry.barycentric_interp import BarycentricInterpolator
    >>>
    >>> # Create interpolator
    >>> interp = BarycentricInterpolator(V_src, F_src, V_dst)
    >>>
    >>> # Apply to deformed source mesh
    >>> V_dst_deformed = interp(V_src_deformed)
    >>>
    >>> # Works with batches too
    >>> V_dst_batch = interp(V_src_batch)  # (batch, n_verts, 3)
"""

import numpy as np
import torch
import trimesh


def fabricate_tet(p0, p1, p2):
    """
    Fabricate a tetrahedron from triangle vertices by adding a point
    perpendicular to the triangle plane.

    Args:
        p0, p1, p2: Triangle vertices, shape (..., 3)

    Returns:
        p3: Fourth point of tetrahedron, shape (..., 3)
    """
    n = np.cross(p1 - p0, p2 - p0, axis=-1)
    p3 = p0 + n
    return p3


def compute_barycentric_coords_3d(p, v0, v1, v2, v3):
    """
    Compute 3D barycentric coordinates for point p in tetrahedron (v0, v1, v2, v3).

    Args:
        p: Query points, shape (..., 3)
        v0, v1, v2, v3: Tetrahedron vertices, shape (..., 3)

    Returns:
        Barycentric coordinates, shape (..., 4)
    """
    # Build matrix [v1-v0, v2-v0, v3-v0]
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)  # (..., 3, 3)

    # Solve T @ [b1, b2, b3]^T = p - v0
    rhs = p - v0  # (..., 3)

    b123 = np.linalg.solve(T, rhs[..., None]).squeeze(-1)  # (..., 3)

    # b0 = 1 - b1 - b2 - b3
    b0 = 1.0 - b123.sum(axis=-1, keepdims=True)  # (..., 1)

    return np.concatenate([b0, b123], axis=-1)  # (..., 4)


def barycentric_interpolation(V_tet, F_tet, face_ids, bary_coords):
    """
    Interpolate vertices using precomputed barycentric coordinates.

    Args:
        V_tet: Tetrahedralized vertices, shape (batch, n_verts + n_faces, 3) or (n_verts + n_faces, 3)
        F_tet: Tetrahedron face indices, shape (n_faces, 4)
        face_ids: Closest face indices for each target point, shape (n_target,)
        bary_coords: Barycentric coordinates, shape (n_target, 4)

    Returns:
        Interpolated vertices, shape (batch, n_target, 3) or (n_target, 3)
    """
    # Handle batched vs non-batched
    has_batch = V_tet.ndim == 3
    if not has_batch:
        V_tet = V_tet[None, ...]  # Add batch dimension

    # Get tetrahedron vertices: (batch, n_target, 4, 3)
    tet_indices = F_tet[face_ids]  # (n_target, 4)

    # Index into V_tet: (batch, n_target, 4, 3)
    v0 = V_tet[:, tet_indices[:, 0], :]  # (batch, n_target, 3)
    v1 = V_tet[:, tet_indices[:, 1], :]
    v2 = V_tet[:, tet_indices[:, 2], :]
    v3 = V_tet[:, tet_indices[:, 3], :]

    # Interpolate: sum(v_i * b_i)
    bc = bary_coords[None, :, :]  # (1, n_target, 4)
    result = (
        v0 * bc[..., 0:1] + v1 * bc[..., 1:2] + v2 * bc[..., 2:3] + v3 * bc[..., 3:4]
    )  # (batch, n_target, 3)

    if not has_batch:
        result = result[0]  # Remove batch dimension

    return result


class BarycentricInterpolator(torch.nn.Module):
    """
    Barycentric interpolation for transferring deformations from source to target mesh.
    Uses tetrahedral mesh construction and 3D barycentric coordinates.
    """

    def __init__(self, V_src, F_src, V_dst, correspondence_path=None):
        """
        Initialize barycentric interpolator.

        Args:
            V_src: Source mesh vertices, (n_src_verts, 3) - torch tensor
            F_src: Source mesh faces, (n_faces, 3) - torch tensor
            V_dst: Target mesh vertices, (n_dst_verts, 3) - torch tensor
            correspondence_path: Optional path to load precomputed correspondence
        """
        super().__init__()
        if not all(isinstance(x, torch.Tensor) for x in (V_src, F_src, V_dst)):
            raise TypeError("V_src, F_src, and V_dst must be torch.Tensor.")

        # Convert to numpy for trimesh operations
        self.V_src = V_src.detach().cpu().numpy()
        self.F_src = F_src.detach().cpu().numpy()
        self.V_dst = V_dst.detach().cpu().numpy()
        self.register_buffer(
            "V_src_torch", V_src.detach().to(dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "V_dst_torch", V_dst.detach().to(dtype=torch.float32), persistent=False
        )
        self.register_buffer("F_src_torch", F_src.detach().to(dtype=torch.long), persistent=False)
        self.register_buffer("face_ids", None, persistent=False)
        self.register_buffer("bary_coords", None, persistent=False)
        self.register_buffer("F_src_tet", None, persistent=False)

        if correspondence_path is not None:
            self.load_correspondence(correspondence_path)
        else:
            self.compute_correspondence()

    def compute_correspondence(self):
        """Compute correspondence using trimesh for closest point queries."""
        # Create trimesh object
        mesh_src = trimesh.Trimesh(vertices=self.V_src, faces=self.F_src)

        # Find closest points on source mesh for each target vertex
        closest_points, _, face_ids = mesh_src.nearest.on_surface(self.V_dst)
        face_ids = face_ids.astype(np.int64)

        # Fabricate tetrahedra from triangles
        V_src_P3 = fabricate_tet(
            self.V_src[self.F_src[:, 0]],
            self.V_src[self.F_src[:, 1]],
            self.V_src[self.F_src[:, 2]],
        )

        # Concatenate original vertices with fabricated points
        V_src_tet = np.concatenate([self.V_src, V_src_P3], axis=0)

        # Create tetrahedral face indices: original triangle + new vertex
        new_vert_indices = np.arange(self.F_src.shape[0])[:, None] + self.V_src.shape[0]
        F_src_tet = np.concatenate([self.F_src, new_vert_indices], axis=1)

        # Compute barycentric coordinates for each target point
        tet_indices = F_src_tet[face_ids]  # (n_dst, 4)
        v0 = V_src_tet[tet_indices[:, 0]]
        v1 = V_src_tet[tet_indices[:, 1]]
        v2 = V_src_tet[tet_indices[:, 2]]
        v3 = V_src_tet[tet_indices[:, 3]]

        bary_coords = compute_barycentric_coords_3d(self.V_dst, v0, v1, v2, v3)

        # Store as torch tensors (buffers)
        self.face_ids = torch.from_numpy(face_ids).to(self.device).long()
        self.bary_coords = torch.from_numpy(bary_coords).to(self.device)
        self.F_src_tet = torch.from_numpy(F_src_tet).to(self.device).long()

    def load_correspondence(self, path):
        """Load precomputed correspondence from file."""
        correspondence = np.load(path, allow_pickle=False)
        face_ids = correspondence["face_ids"]
        bary_coords = correspondence["bary_coords"]
        F_src_tet = correspondence["F_src_tet"]

        self.face_ids = torch.from_numpy(face_ids).to(self.device).long()
        self.bary_coords = torch.from_numpy(bary_coords).to(self.device).to(self.dtype)
        self.F_src_tet = torch.from_numpy(F_src_tet).to(self.device).long()

    def save_correspondence(self, path):
        """Save computed correspondence to file."""
        face_ids_np = self.face_ids.detach().cpu().numpy()
        bary_coords_np = self.bary_coords.detach().cpu().numpy()
        F_src_tet_np = self.F_src_tet.detach().cpu().numpy()

        np.savez(path, face_ids=face_ids_np, bary_coords=bary_coords_np, F_src_tet=F_src_tet_np)

    @property
    def device(self):
        return self.F_src_torch.device

    @property
    def dtype(self):
        return self.V_src_torch.dtype

    def forward(self, V_src_deformed):
        """
        Apply barycentric interpolation to transfer deformation.

        Args:
            V_src_deformed: Deformed source vertices, shape (batch, n_src_verts, 3) or (n_src_verts, 3)

        Returns:
            Deformed target vertices, shape (batch, n_dst_verts, 3) or (n_dst_verts, 3)
        """
        if not isinstance(V_src_deformed, torch.Tensor):
            raise TypeError("V_src_deformed must be a torch.Tensor.")

        # Handle batched vs non-batched
        has_batch = V_src_deformed.ndim == 3
        if not has_batch:
            V_src_deformed = V_src_deformed[None, ...]

        # Fabricate tetrahedra for deformed source
        # Get face vertices and fabricate tets
        face_indices = self.F_src_torch
        if face_indices.device != V_src_deformed.device:
            face_indices = face_indices.to(device=V_src_deformed.device)
        f0 = V_src_deformed[:, face_indices[:, 0], :]
        f1 = V_src_deformed[:, face_indices[:, 1], :]
        f2 = V_src_deformed[:, face_indices[:, 2], :]

        V_src_P3 = f0 + torch.cross(f1 - f0, f2 - f0, dim=-1)  # (batch, n_faces, 3)

        # Concatenate: (batch, n_src_verts + n_faces, 3)
        V_src_tet = torch.cat([V_src_deformed, V_src_P3], dim=1)

        # Interpolate using precomputed correspondence
        face_ids = self.face_ids
        bary_coords = self.bary_coords
        F_src_tet = self.F_src_tet
        if face_ids.device != V_src_deformed.device:
            face_ids = face_ids.to(device=V_src_deformed.device)
        if bary_coords.device != V_src_deformed.device or bary_coords.dtype != V_src_deformed.dtype:
            bary_coords = bary_coords.to(device=V_src_deformed.device, dtype=V_src_deformed.dtype)
        if F_src_tet.device != V_src_deformed.device:
            F_src_tet = F_src_tet.to(device=V_src_deformed.device)
        result = barycentric_interpolation(V_src_tet, F_src_tet, face_ids, bary_coords)

        if not has_batch:
            result = result[0]

        return result
