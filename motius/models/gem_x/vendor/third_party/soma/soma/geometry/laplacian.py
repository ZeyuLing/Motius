# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def cotangent_weights(V, F):
    """
    Compute cotangent weights for mesh Laplacian.

    Args:
        V: Vertices (n_verts, 3)
        F: Faces (n_faces, 3)

    Returns:
        Sparse COO indices and values for cotangent Laplacian matrix
    """
    # Get triangle vertices
    v0 = V[F[:, 0]]  # (n_faces, 3)
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Edge vectors
    e0 = v2 - v1  # opposite to vertex 0
    e1 = v0 - v2  # opposite to vertex 1
    e2 = v1 - v0  # opposite to vertex 2

    # Compute cotangents using: cot(θ) = dot(a,b) / ||cross(a,b)||
    def compute_cot(a, b):
        dot_prod = (a * b).sum(dim=-1)
        cross_prod = torch.cross(a, b, dim=-1)
        cross_norm = torch.norm(cross_prod, dim=-1)
        # Add small epsilon to avoid division by zero
        return dot_prod / (cross_norm + 1e-8)

    # Cotangents at each vertex of each face
    cot0 = compute_cot(e1, e2)  # at vertex 0
    cot1 = compute_cot(e2, e0)  # at vertex 1
    cot2 = compute_cot(e0, e1)  # at vertex 2

    # Build sparse matrix entries
    # Each edge (i,j) gets weight from both adjacent triangles
    i_indices = []
    j_indices = []
    values = []

    # Edge (1,2) opposite to vertex 0
    i_indices.extend([F[:, 1], F[:, 2]])
    j_indices.extend([F[:, 2], F[:, 1]])
    values.extend([cot0, cot0])

    # Edge (2,0) opposite to vertex 1
    i_indices.extend([F[:, 2], F[:, 0]])
    j_indices.extend([F[:, 0], F[:, 2]])
    values.extend([cot1, cot1])

    # Edge (0,1) opposite to vertex 2
    i_indices.extend([F[:, 0], F[:, 1]])
    j_indices.extend([F[:, 1], F[:, 0]])
    values.extend([cot2, cot2])

    # Concatenate all indices and values
    i_indices = torch.cat(i_indices)
    j_indices = torch.cat(j_indices)
    values = torch.cat(values)

    return i_indices, j_indices, values


def _weights_to_laplacian(W):
    """
    Convert a sparse COO weight matrix into a graph Laplacian L = D - W.

    Args:
        W: Coalesced sparse COO tensor (n, n) of non-negative off-diagonal
           edge weights.

    Returns:
        Sparse CSR tensor of shape (n, n)
    """
    n = W.shape[0]
    device = W.device
    dtype = W.dtype

    row_sums = torch.sparse.sum(W, dim=1).to_dense()
    diag_idx = torch.arange(n, device=device)
    diag_idx = torch.stack([diag_idx, diag_idx])

    L = torch.sparse_coo_tensor(
        torch.cat([diag_idx, W.indices()], dim=1),
        torch.cat([row_sums, -W.values()]),
        (n, n),
        device=device,
        dtype=dtype,
    ).coalesce()

    return L.to_sparse_csr()


def build_cotangent_laplacian(V, F):
    """
    Build cotangent Laplacian matrix in sparse CSR format.

    Args:
        V: Vertices (n_verts, 3)
        F: Faces (n_faces, 3)

    Returns:
        Sparse CSR tensor of shape (n_verts, n_verts)
    """
    device = V.device
    dtype = V.dtype
    n_verts = V.shape[0]

    i_indices, j_indices, values = cotangent_weights(V, F)

    indices = torch.stack([i_indices, j_indices])
    W = torch.sparse_coo_tensor(indices, values, (n_verts, n_verts), device=device, dtype=dtype)
    W = W.coalesce()

    # Ensure symmetry
    W = ((W + W.t()) / 2).coalesce()

    return _weights_to_laplacian(W)


def build_uniform_laplacian(F, n_verts, device=None, dtype=None):
    """
    Build the uniform (graph) Laplacian: L = D - A, where A is the binary
    adjacency matrix and D is the diagonal degree matrix.

    Unlike the cotangent Laplacian this is geometry-independent -- it only
    depends on mesh connectivity.

    Args:
        F: Faces (n_faces, 3), int tensor
        n_verts: Number of vertices in the mesh
        device: Torch device (inferred from F if None)
        dtype: Value dtype (default float32)

    Returns:
        Sparse CSR tensor of shape (n_verts, n_verts)
    """
    if device is None:
        device = F.device
    if dtype is None:
        dtype = torch.float32
    F = F.to(device)

    edges = torch.cat(
        [
            F[:, [0, 1]],
            F[:, [1, 2]],
            F[:, [0, 2]],
            F[:, [1, 0]],
            F[:, [2, 1]],
            F[:, [2, 0]],
        ],
        dim=0,
    )

    indices = edges.t()
    values = torch.ones(indices.shape[1], device=device, dtype=dtype)

    A = torch.sparse_coo_tensor(
        indices, values, (n_verts, n_verts), device=device, dtype=dtype
    ).coalesce()
    # Clamp to 1 for a binary adjacency matrix
    A = torch.sparse_coo_tensor(
        A.indices(), torch.clamp(A.values(), max=1.0), A.shape, device=device, dtype=dtype
    ).coalesce()

    return _weights_to_laplacian(A)


def power_laplacian(L, order):
    """
    Compute L^order for higher-order Laplacian.

    Args:
        L: Sparse Laplacian matrix (CSR format)
        order: Power to raise L to

    Returns:
        L^order as sparse CSR tensor
    """
    if order <= 1:
        return L

    result = L
    for _ in range(1, order):
        result = torch.sparse.mm(result, L)

    # Convert back to CSR if needed (mm might return COO)
    if not result.is_sparse_csr:
        result = result.to_sparse_csr()

    return result


def _pytorch_sparse_solve(A, b):
    """Solve Ax = b using torch.sparse.spsolve (requires cuDSS on CUDA)."""
    if not A.is_sparse_csr:
        A = A.to_sparse_csr()
    if b.ndim == 1:
        return torch.sparse.spsolve(A, b.unsqueeze(-1)).squeeze(-1)
    else:
        solutions = []
        for i in range(b.shape[1]):
            sol = torch.sparse.spsolve(A, b[:, i])
            solutions.append(sol)
        return torch.stack(solutions, dim=1)


def _create_cholesky_factor(A):
    """Factor a sparse SPD matrix: returns dense lower-triangular L with A ≈ L @ L^T.
    torch.cholesky_solve(b, L) solves Ax = b on any device without external libraries.
    """
    if not A.is_sparse_csr:
        A = A.to_sparse_csr()
    return torch.linalg.cholesky(A.to_dense())


class LaplacianMesh(torch.nn.Module):
    """
    Pure PyTorch implementation of Laplacian mesh editing.
    Supports both hard and soft constraints for deformation.
    """

    def __init__(
        self,
        V,
        F,
        mask_anchors,
        order=1,
        constraint_mode="hard",
        soft_weight=1e-5,
        jitter=0.0,
        solver="cholespy",
    ):
        """
        Initialize Laplacian mesh editor.

        Args:
            V: Reference vertices (n_verts, 3)
            F: Faces (n_faces, 3)
            mask_anchors: Boolean mask (n_verts,) - True for anchors, False for free
            order: Order of Laplacian (1 or 2)
            constraint_mode: "hard" or "soft"
            soft_weight: Weight for soft constraints (only used if constraint_mode="soft")
            jitter: Small value added to diagonal for numerical stability
            solver: "cholespy" (pre-factored Cholesky via torch.linalg.cholesky) or
                    "pytorch" (torch.sparse.spsolve, requires cuDSS on CUDA)
        """
        super().__init__()
        if solver not in ("cholespy", "pytorch"):
            raise ValueError(f"solver must be 'cholespy' or 'pytorch', got '{solver}'")
        self.solver_backend = solver
        self.device = V.device
        self.dtype = V.dtype
        self.order = int(order)
        self.constraint_mode = constraint_mode
        self.soft_weight = float(soft_weight)
        self.jitter = float(jitter)

        self.register_buffer("V", V, persistent=False)
        self.register_buffer("F", F, persistent=False)

        # Build Laplacian
        L = build_cotangent_laplacian(V, F)
        if order > 1:
            L = power_laplacian(L, order)
        self.L = L  # plain attribute — not a buffer, so DDP will not try to broadcast it

        # Identify free and constrained vertices
        self.register_buffer("vid_unknown", torch.where(~mask_anchors)[0], persistent=False)
        self.register_buffer("vid_constrained", torch.where(mask_anchors)[0], persistent=False)

        # Dense matrices are registered as buffers (auto-moved by .to() / DDP-safe).
        # Sparse CSR matrices are plain attributes moved manually in _apply().
        self.register_buffer("btilde", None, persistent=False)
        self.L_FG = None
        self.L_FF = None
        self.C = None
        self.K = None
        self.register_buffer("core_rhs", None, persistent=False)
        # Cholesky factor (dense, lower-triangular); auto-moves with .to(device).
        self.register_buffer("_chol_factor", None, persistent=False)

        if constraint_mode == "hard":
            self._setup_hard_constraints()
        elif constraint_mode == "soft":
            self._setup_soft_constraints()
        else:
            raise ValueError("constraint_mode must be 'hard' or 'soft'")

    def _apply(self, fn):
        super()._apply(fn)
        # Manually move sparse plain attributes (not registered buffers).
        self.L = fn(self.L)
        if self.L_FG is not None:
            self.L_FG = fn(self.L_FG)
        if self.L_FF is not None:
            self.L_FF = fn(self.L_FF)
        if self.C is not None:
            self.C = fn(self.C)
        if self.K is not None:
            self.K = fn(self.K)
        return self

    def _setup_hard_constraints(self):
        """Setup matrices for hard constraint mode."""
        n = self.V.shape[0]

        # Convert L to COO temporarily for easier indexing/filtering
        L_coo = self.L.to_sparse_coo()
        L_indices = L_coo.indices()
        L_values = L_coo.values()

        # Find entries where row is in vid_unknown
        mask = torch.isin(L_indices[0], self.vid_unknown)
        L_U_indices = L_indices[:, mask]
        L_U_values = L_values[mask]

        # Remap row indices to 0...|U|-1
        row_mapping = torch.zeros(n, dtype=torch.long, device=self.device)
        row_mapping[self.vid_unknown] = torch.arange(len(self.vid_unknown), device=self.device)
        L_U_indices[0] = row_mapping[L_U_indices[0]]

        L_U = (
            torch.sparse_coo_tensor(
                L_U_indices,
                L_U_values,
                (len(self.vid_unknown), n),
                device=self.device,
                dtype=self.dtype,
            )
            .coalesce()
            .to_sparse_csr()
        )

        # Compute b_tilde = L_U @ V
        self.btilde = torch.sparse.mm(L_U, self.V)  # (|U|, 3)

        # Extract L_FF (unknown x unknown) and L_FG (unknown x constrained)
        mask_FF = torch.isin(L_U_indices[1], self.vid_unknown)
        mask_FG = torch.isin(L_U_indices[1], self.vid_constrained)

        # L_FF
        L_FF_indices = L_U_indices[:, mask_FF].clone()
        L_FF_values = L_U_values[mask_FF]
        col_mapping = torch.zeros(n, dtype=torch.long, device=self.device)
        col_mapping[self.vid_unknown] = torch.arange(len(self.vid_unknown), device=self.device)
        L_FF_indices[1] = col_mapping[L_FF_indices[1]]

        L_FF = (
            torch.sparse_coo_tensor(
                L_FF_indices,
                L_FF_values,
                (len(self.vid_unknown), len(self.vid_unknown)),
                device=self.device,
                dtype=self.dtype,
            )
            .coalesce()
            .to_sparse_csr()
        )

        # L_FG
        L_FG_indices = L_U_indices[:, mask_FG].clone()
        L_FG_values = L_U_values[mask_FG]
        col_mapping_G = torch.zeros(n, dtype=torch.long, device=self.device)
        col_mapping_G[self.vid_constrained] = torch.arange(
            len(self.vid_constrained), device=self.device
        )
        L_FG_indices[1] = col_mapping_G[L_FG_indices[1]]

        self.L_FG = (
            torch.sparse_coo_tensor(
                L_FG_indices,
                L_FG_values,
                (len(self.vid_unknown), len(self.vid_constrained)),
                device=self.device,
                dtype=self.dtype,
            )
            .coalesce()
            .to_sparse_csr()
        )

        # Factor L_FF directly — L_FF is square (|F|x|F|) and (semi-)definite with
        # Dirichlet BCs, so normal equations are unnecessary and harmful (they
        # square the condition number). The cotangent Laplacian is NSD for odd
        # orders and PSD for even orders, so we negate for odd orders to get SPD.
        self._hard_chol_sign = -1 if (self.order % 2 == 1) else 1
        self.L_FF = L_FF  # stored for pytorch solver path

        def _negate_sparse(A):
            coo = A.to_sparse_coo().coalesce()
            return (
                torch.sparse_coo_tensor(
                    coo.indices(), -coo.values(), coo.shape, device=self.device, dtype=self.dtype
                )
                .coalesce()
                .to_sparse_csr()
            )

        L_FF_for_chol = _negate_sparse(L_FF) if self._hard_chol_sign == -1 else L_FF

        # Add jitter if needed (added to the SPD matrix used for Cholesky)
        if self.jitter > 0:
            n_unknown = len(self.vid_unknown)
            diag_idx = torch.arange(n_unknown, device=self.device)
            diag_idx = torch.stack([diag_idx, diag_idx])
            diag_val = torch.full((n_unknown,), self.jitter, device=self.device, dtype=self.dtype)
            jitter_mat = torch.sparse_coo_tensor(
                diag_idx, diag_val, (n_unknown, n_unknown)
            ).to_sparse_csr()
            L_FF_for_chol = L_FF_for_chol + jitter_mat
            if not L_FF_for_chol.is_sparse_csr:
                L_FF_for_chol = L_FF_for_chol.to_sparse_csr()

        if self.solver_backend == "cholespy":
            self._chol_factor = _create_cholesky_factor(L_FF_for_chol)

    def _setup_soft_constraints(self):
        """Setup matrices for soft constraint mode."""
        n = self.V.shape[0]
        n_constrained = len(self.vid_constrained)

        # Build selection matrix C
        rows = torch.arange(n_constrained, device=self.device)
        cols = self.vid_constrained
        vals = torch.ones(n_constrained, device=self.device, dtype=self.dtype)
        C_indices = torch.stack([rows, cols])
        self.C = torch.sparse_coo_tensor(
            C_indices, vals, (n_constrained, n), device=self.device, dtype=self.dtype
        ).to_sparse_csr()

        # K = L^T @ L + w * C^T @ C
        L_T = self.L.t()
        if not L_T.is_sparse_csr:
            L_T = L_T.to_sparse_csr()
        K = torch.sparse.mm(L_T, self.L)
        if not K.is_sparse_csr:
            K = K.to_sparse_csr()

        if self.soft_weight > 0:
            C_T = self.C.t()
            if not C_T.is_sparse_csr:
                C_T = C_T.to_sparse_csr()
            C_TC = torch.sparse.mm(C_T, self.C)
            if not C_TC.is_sparse_csr:
                C_TC = C_TC.to_sparse_csr()
            # Scale by soft_weight - need to convert to COO temporarily
            C_TC_coo = C_TC.to_sparse_coo()
            C_TC_scaled = torch.sparse_coo_tensor(
                C_TC_coo.indices(),
                C_TC_coo.values() * self.soft_weight,
                C_TC_coo.shape,
                device=self.device,
                dtype=self.dtype,
            ).to_sparse_csr()
            K = K + C_TC_scaled
            if not K.is_sparse_csr:
                K = K.to_sparse_csr()

        # Add jitter
        if self.jitter > 0:
            diag_idx = torch.arange(n, device=self.device)
            diag_idx = torch.stack([diag_idx, diag_idx])
            diag_val = torch.full((n,), self.jitter, device=self.device, dtype=self.dtype)
            jitter_mat = torch.sparse_coo_tensor(diag_idx, diag_val, (n, n)).to_sparse_csr()
            K = K + jitter_mat
            if not K.is_sparse_csr:
                K = K.to_sparse_csr()

        self.K = K

        # Precompute core RHS: L^T @ (L @ V)
        LV = torch.sparse.mm(self.L, self.V)
        self.core_rhs = torch.sparse.mm(L_T, LV)

        if self.solver_backend == "cholespy":
            self._chol_factor = _create_cholesky_factor(self.K)

    def solve(self, Vdef):
        """
        Solve for deformed mesh given constraint vertices.

        Args:
            Vdef: Target vertices (B, n, 3) or (n, 3)
                  For hard constraints: only values at constrained vertices are used
                  For soft constraints: constrained vertices pull the solution

        Returns:
            Deformed vertices (B, n, 3) or (n, 3)
        """
        # Handle single vs batch
        single = Vdef.ndim == 2
        if single:
            Vdef = Vdef.unsqueeze(0)

        B, n, _ = Vdef.shape

        if self.constraint_mode == "hard":
            out = Vdef.clone()
            XG_all = Vdef[:, self.vid_constrained, :]  # (B, |G|, 3)

            if self.solver_backend == "cholespy":
                # Batched sparse-mm: reshape (B, |G|, 3) → (|G|, B*3)
                XG_2d = XG_all.permute(1, 0, 2).reshape(len(self.vid_constrained), B * 3)
                btilde_2d = self.btilde.repeat(1, B)
                rhs_2d = btilde_2d - torch.sparse.mm(self.L_FG, XG_2d)  # (|U|, B*3)
                # Solve L_FF x = rhs directly (no normal equations).
                # Cholesky was built from chol_sign * L_FF, so pass chol_sign * rhs.
                x_2d = torch.cholesky_solve(
                    self._hard_chol_sign * rhs_2d, self._chol_factor
                )  # (|U|, B*3)
                x_all = x_2d.reshape(len(self.vid_unknown), B, 3).permute(1, 0, 2)  # (B,|U|,3)
                out[:, self.vid_unknown, :] = x_all
            else:
                for b in range(B):
                    rhs = self.btilde - torch.sparse.mm(self.L_FG, XG_all[b])  # (|U|, 3)
                    x_unknown = _pytorch_sparse_solve(self.L_FF, rhs)  # (|U|, 3)
                    out[b, self.vid_unknown, :] = x_unknown

        else:  # soft
            C_T = self.C.t()

            if self.solver_backend == "cholespy":
                if self.soft_weight > 0:
                    d_all = Vdef[:, self.vid_constrained, :]  # (B, |G|, 3)
                    d_2d = d_all.permute(1, 0, 2).reshape(len(self.vid_constrained), B * 3)
                    rhs_2d = self.core_rhs.repeat(1, B) + self.soft_weight * torch.sparse.mm(
                        C_T, d_2d
                    )  # (n, B*3)
                else:
                    rhs_2d = self.core_rhs.repeat(1, B)
                x_2d = torch.cholesky_solve(rhs_2d, self._chol_factor)  # (n, B*3)
                out = x_2d.reshape(n, B, 3).permute(1, 0, 2)  # (B, n, 3)
            else:
                out = torch.empty_like(Vdef)
                for b in range(B):
                    rhs = self.core_rhs.clone()
                    if self.soft_weight > 0:
                        d = Vdef[b, self.vid_constrained, :]
                        rhs += self.soft_weight * torch.sparse.mm(C_T, d)
                    x = _pytorch_sparse_solve(self.K, rhs)
                    out[b] = x

        return out[0] if single else out
