"""Differentiable adapter for the exact neutral SMPL rig shipped by the gallery.

This is a validation rig (four skin weights, rank-64 pose correctives), not a
replacement for the full SMPL model or validation of other shapes/genders.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues


class GallerySMPL(torch.nn.Module):
    def __init__(self, model_dir: Path):
        super().__init__()
        meta = json.loads((model_dir / "model.json").read_text())
        files = meta["files"]
        n = meta["vertices"]
        corrective = meta["pose_correctives"]
        rank = corrective["rank"]

        def read(name, dtype, shape):
            return np.fromfile(model_dir / files[name], dtype).reshape(shape)

        self.register_buffer("rest", torch.from_numpy(read("joints", "<f4", (24, 3))))
        self.register_buffer("parents", torch.tensor(meta["parents"], dtype=torch.long))
        self.register_buffer(
            "v_template", torch.from_numpy(read("vertices", "<f4", (n, 3)))
        )
        self.faces = read("faces", "<u2", (-1, 3)).astype(np.int64)
        indices = read("skin_indices", "u1", (n, 4))
        weights4 = read("skin_weights", "<u2", (n, 4)).astype(np.float32)
        weights4 /= weights4.sum(-1, keepdims=True)
        weights = np.zeros((n, 24), np.float32)
        # add.at also handles repeated joint indices in padded sparse weights.
        np.add.at(weights, (np.arange(n)[:, None], indices), weights4)
        self.register_buffer("weights", torch.from_numpy(weights))
        basis = np.fromfile(model_dir / corrective["files"]["basis"], "<f2")
        basis = basis.reshape(rank, n, 4)[..., :3].astype(np.float32).reshape(rank, -1)
        projection = np.fromfile(
            model_dir / corrective["files"]["projection"], "<f4"
        ).reshape(rank, 207)
        self.register_buffer("basis", torch.from_numpy(basis))
        self.register_buffer("projection", torch.from_numpy(projection))

    def forward(self, *, betas, body_pose, global_orient, transl, **kwargs):
        if torch.count_nonzero(betas):
            raise ValueError("Gallery audit rig only supports neutral beta=0")
        n = len(global_orient)
        rot = batch_rodrigues(
            torch.cat([global_orient, body_pose], -1).reshape(-1, 3)
        ).reshape(n, 24, 3, 3)
        joints, transforms = batch_rigid_transform(
            rot, self.rest[None].expand(n, -1, -1), self.parents
        )
        features = (rot[:, 1:] - torch.eye(3, device=rot.device)).reshape(n, 207)
        posed = self.v_template[None] + (
            (features @ self.projection.T) @ self.basis
        ).reshape(n, -1, 3)
        skin = torch.einsum("vj,bjkl->bvkl", self.weights, transforms)
        homogeneous = torch.cat([posed, torch.ones_like(posed[..., :1])], -1)
        vertices = (skin @ homogeneous[..., None])[..., :3, 0] + transl[:, None]
        return SimpleNamespace(joints=joints + transl[:, None], vertices=vertices)
