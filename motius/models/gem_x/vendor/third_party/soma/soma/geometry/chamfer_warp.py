# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import warp as wp

from soma.geometry._warp_init import ensure_warp_initialized


@wp.kernel
def chamfer_distance_batched_kernel(
    src_points: wp.array(dtype=wp.vec3),  # (B*N,) flattened batch of vec3
    mesh_ids: wp.array(dtype=wp.uint64),  # (B,) array of mesh handles
    batch_indices: wp.array(dtype=int),  # (B*N,) which batch each src point belongs to
    loss_output: wp.array(dtype=float),  # (B,) separate loss per batch
):
    tid = wp.tid()

    # Get which batch this point belongs to
    batch_idx = batch_indices[tid]
    mesh_id = mesh_ids[batch_idx]

    # Query the BVH for the closest point on the mesh (within max distance 1e6)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    p = src_points[tid]

    found = wp.mesh_query_point(mesh_id, p, 1e6, sign, face_index, face_u, face_v)

    if found:
        # Reconstruct the surface point from barycentric coords
        closest_p = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)

        # Compute squared distance
        diff = p - closest_p
        dist_sq = wp.dot(diff, diff)

        # Atomic add to the correct batch element's loss
        wp.atomic_add(loss_output, batch_idx, dist_sq)


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Cache the Warp mesh objects to avoid rebuilding topology
        self._cached_meshes = []
        self._cached_mesh_ids = None

    def clear_cache(self):
        self._cached_meshes = []
        self._cached_mesh_ids = None

    def forward(self, src_points, target_verts, target_faces=None, refit=False):
        """
        src_points: (B, N, 3) or (N, 3) - Source query points
        target_verts: (B, M, 3) or (M, 3) - Target vertices (supports batching!)
        target_faces: (F, 3) - Static topology (optional). If None, treats target as point cloud
                        by creating degenerate triangles (one per vertex)
        """
        # Ensure batched format
        src_is_batched = src_points.dim() == 3
        target_is_batched = target_verts.dim() == 3

        if src_is_batched and target_is_batched:
            assert src_points.shape[0] == target_verts.shape[0], (
                f"Batch size mismatch: src {src_points.shape[0]} vs target {target_verts.shape[0]}"
            )
            batch_size = src_points.shape[0]
        elif src_is_batched:
            batch_size = src_points.shape[0]
            target_verts = target_verts.unsqueeze(0).expand(batch_size, -1, -1)
        elif target_is_batched:
            batch_size = target_verts.shape[0]
            src_points = src_points.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Add batch dimension
            batch_size = 1
            src_points = src_points.unsqueeze(0)
            target_verts = target_verts.unsqueeze(0)

        num_src_per_batch = src_points.shape[1]

        # Flatten src_points to (B*N, 3)
        src_flat = src_points.reshape(-1, 3)
        src_wp = wp.from_torch(src_flat.contiguous(), dtype=wp.vec3)

        # Create batch indices array: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
        batch_indices = torch.arange(
            batch_size, device=src_points.device, dtype=torch.int32
        ).repeat_interleave(num_src_per_batch)
        batch_indices_wp = wp.from_torch(batch_indices.contiguous(), dtype=wp.int32)

        # Prepare faces (shared across all batches)
        if target_faces is None:
            # Point cloud mode - create degenerate triangles
            num_verts = target_verts.shape[1]
            target_faces = (
                torch.arange(num_verts, device=target_verts.device, dtype=torch.int32)
                .repeat(3, 1)
                .T
            )
            indices_flat = target_faces.reshape(-1)
            rebuild_cache = True  # Always rebuild in point cloud mode
        else:
            indices_flat = target_faces.to(dtype=torch.int32).contiguous().reshape(-1)
            rebuild_cache = len(self._cached_meshes) != batch_size

        indices_wp = wp.from_torch(indices_flat, dtype=wp.int32)

        # Three distinct paths based on cache state
        if rebuild_cache:
            # Path 1: Build new meshes from scratch
            self._cached_meshes = []
            mesh_ids = []

            for b in range(batch_size):
                verts_wp = wp.from_torch(target_verts[b].contiguous(), dtype=wp.vec3)
                mesh = wp.Mesh(points=verts_wp, indices=indices_wp)
                self._cached_meshes.append(mesh)
                mesh_ids.append(mesh.id)

            self._cached_mesh_ids = torch.tensor(
                mesh_ids, dtype=torch.int64, device=src_points.device
            )

        elif refit:
            # Path 2: Refit existing meshes with new vertex positions
            for b in range(batch_size):
                verts_wp = wp.from_torch(target_verts[b].contiguous(), dtype=wp.vec3)
                self._cached_meshes[b].points = verts_wp
                self._cached_meshes[b].refit()
            # mesh_ids unchanged, no need to rebuild

        # Path 3: Use cached meshes as-is (no loop needed!)
        # mesh_ids_wp is already set from previous build

        mesh_ids_wp = wp.from_torch(self._cached_mesh_ids.contiguous(), dtype=wp.uint64)

        # Single kernel launch for all batches
        result = ChamferBatchedFunction.apply(
            src_points,
            src_wp,
            mesh_ids_wp,
            batch_indices_wp,
            self._cached_meshes,
            batch_size,
            num_src_per_batch,
        )

        # Return scalar if input was unbatched
        return result.squeeze() if batch_size == 1 else result


class ChamferBatchedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        src_points,
        src_wp,
        mesh_ids_wp,
        batch_indices_wp,
        meshes,
        batch_size,
        num_src_per_batch,
    ):
        ensure_warp_initialized()
        ctx.save_for_backward(src_points)
        ctx.src_wp = src_wp
        ctx.mesh_ids_wp = mesh_ids_wp
        ctx.batch_indices_wp = batch_indices_wp
        ctx.meshes = meshes
        ctx.batch_size = batch_size
        ctx.num_src_per_batch = num_src_per_batch

        # Allocate output loss per batch element
        loss = torch.zeros(batch_size, dtype=torch.float32, device=src_points.device)
        loss_wp = wp.from_torch(loss)
        ctx.loss_wp = loss_wp

        # Run batched kernel - single launch for all batches!
        ctx.tape = wp.Tape()
        with ctx.tape:
            wp.launch(
                kernel=chamfer_distance_batched_kernel,
                dim=src_wp.shape[0],  # Total number of points across all batches
                inputs=[src_wp, mesh_ids_wp, batch_indices_wp, loss_wp],
                device=src_points.device.type,
            )

        # Return the averaged loss value per batch (mean instead of sum)
        return wp.to_torch(loss_wp) / num_src_per_batch

    @staticmethod
    def backward(ctx, grad_output):
        (src_points,) = ctx.saved_tensors

        # Scale the gradient by 1/num_points to match the forward pass averaging
        # grad_output is (B,) so we need to repeat it for each point in the batch
        scaled_grad = grad_output / ctx.num_src_per_batch
        scaled_grad_wp = wp.from_torch(scaled_grad.contiguous())

        # Warp's backward pass
        ctx.tape.backward(grads={ctx.loss_wp: scaled_grad_wp})

        # Extract gradient for source points and reshape to (B, N, 3)
        src_grad = None
        if ctx.src_wp.grad is not None:
            src_grad_flat = wp.to_torch(ctx.src_wp.grad)
            src_grad = src_grad_flat.reshape(ctx.batch_size, ctx.num_src_per_batch, 3)

        # Return gradients: (src_points, src_wp, mesh_ids_wp, batch_indices_wp, meshes, batch_size, num_src_per_batch)
        return src_grad, None, None, None, None, None, None
