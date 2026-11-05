# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh

from .geometry.barycentric_interp import BarycentricInterpolator
from .geometry.laplacian import LaplacianMesh
from .units import Unit


class CoordAxis:
    """Named axis constants for declaring a model's native coordinate convention.

    Each constant is a ``(axis_index, sign)`` tuple, where ``axis_index`` is
    0=X, 1=Y, 2=Z and ``sign`` is +1 or -1.

    SOMA standard: Y+ up (``Y``), Z+ forward (``Z``).
    """

    X = (0, +1)
    Y = (1, +1)
    Z = (2, +1)
    NEG_X = (0, -1)
    NEG_Y = (1, -1)
    NEG_Z = (2, -1)


# Parity of each (right_idx, up_idx, fwd_idx) permutation of (0,1,2).
# Used by _apply_coord_transform to derive the right-axis sign so that
# the resulting transform always has determinant +1 (proper rotation).
_PERM_PARITY = {
    (0, 1, 2): +1,
    (1, 2, 0): +1,
    (2, 0, 1): +1,
    (0, 2, 1): -1,
    (2, 1, 0): -1,
    (1, 0, 2): -1,
}


class NonPersistentModuleWrapper(nn.Module):
    """Wrap a module but drop all of its state_dict entries."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def load_state_dict(self, state_dict, strict=True):
        return torch.nn.modules.module._IncompatibleKeys([], [])


class SMPLSimplified(nn.Module):
    """Wrapper around SMPL/SMPLX/SMPLH to simplify the forward pass"""

    def __init__(self, smpl_model, device):
        super().__init__()
        self.smpl_model = smpl_model
        self.device = device
        self.faces = smpl_model.faces
        self.v_template = self.smpl_model.v_template
        self.shape_dirs = self.smpl_model.shapedirs
        self.num_betas = smpl_model.num_betas

    def forward(self, betas=None):
        blend_shape = torch.einsum("bl,mkl->bmk", [betas, self.shape_dirs])
        v_shaped = self.v_template + blend_shape
        return v_shaped


class AnnySimplified(nn.Module):
    """Wrapper around Anny to simplify the forward pass"""

    def __init__(self, anny_model, device):
        super().__init__()
        self.anny_model = anny_model.to(device=device, dtype=torch.float32)
        self.device = device
        # ignore some local change labels
        full_local_change_labels = self.anny_model.local_change_labels
        self.phenotype_labels = ["gender", "age", "muscle", "weight", "height", "proportions"]

        ignore_names = ["mouth", "eye", "nipple", "cheek", "chin", "ear", "lip", "nose"]
        self.local_change_labels = []
        self.ignore_change_labels = []
        for label in full_local_change_labels:
            keep_label = True
            for ignore_label in ignore_names:
                if ignore_label in label:
                    keep_label = False
                    break
            if keep_label and label not in self.local_change_labels:
                self.local_change_labels.append(label)
            elif not keep_label and label not in self.ignore_change_labels:
                self.ignore_change_labels.append(label)

    def forward(self, phenotype_kwargs=None, local_changes_kwargs=None):
        phenotype_kwargs = self.anny_model.parse_phenotype_kwargs(phenotype_kwargs)
        assert set(phenotype_kwargs) <= set(self.anny_model.phenotype_labels), (
            f"Invalid phenotype: {set(phenotype_kwargs) - set(self.anny_model.phenotype_labels)}; available: {self.anny_model.phenotype_labels}"
        )
        blendshape_coeffs = self.anny_model.get_phenotype_blendshape_coefficients(
            **phenotype_kwargs, local_changes=local_changes_kwargs
        )
        rest_vertices = self.anny_model.get_rest_vertices(blendshape_coeffs)
        return rest_vertices


class BaseIdentityModel(nn.Module, ABC):
    """Abstract base class for identity models.

    Each subclass **must** declare its native length unit via a ``NATIVE_UNIT``
    class attribute (a :class:`Unit` enum member).  Omitting it raises
    :class:`TypeError` at construction time.

    All internal computation (``get_rest_shape``, ``identity_model_to_soma``,
    LaplacianMesh) operates in native units.  The conversion to the caller's
    desired ``output_unit`` happens once, at the output boundary of ``forward()``.
    """

    NATIVE_UNIT: Unit
    NATIVE_UP: tuple = CoordAxis.Y  # SOMA standard: Y+ up
    NATIVE_FORWARD: tuple = CoordAxis.Z  # SOMA standard: Z+ forward

    def __init__(self, data_root, low_lod, device, output_unit=Unit.METERS, **kwargs):
        nv_lod_mid_to_low = kwargs.pop("nv_lod_mid_to_low", None)
        soma_low_lod_faces = kwargs.pop("soma_low_lod_faces", None)
        super().__init__()
        if not hasattr(self, "NATIVE_UNIT") or not isinstance(self.NATIVE_UNIT, Unit):
            raise TypeError(
                f"{type(self).__name__} must define a NATIVE_UNIT class attribute (a Unit enum member)"
            )
        self.data_root = Path(data_root)
        self.low_lod = low_lod
        self.device = device
        self._unit_conversion = self.NATIVE_UNIT.meters_per_unit / output_unit.meters_per_unit
        if nv_lod_mid_to_low is not None:
            self.register_buffer("_nv_lod_mid_to_low", nv_lod_mid_to_low, persistent=False)
        else:
            self._nv_lod_mid_to_low = None
        self._soma_low_lod_faces = soma_low_lod_faces

    def _apply_soma_lod(self, V_soma, F_soma=None):
        """Subset SOMA-topology vertices/faces for low LOD.

        Returns (V_soma, F_soma) unchanged when not in low-LOD mode, or the
        subsetted pair when ``_nv_lod_mid_to_low`` is available.
        """
        if self._nv_lod_mid_to_low is None:
            return V_soma, F_soma
        V_low = V_soma[self._nv_lod_mid_to_low]
        F_low = self._soma_low_lod_faces if F_soma is not None else None
        return V_low, F_low

    @property
    @abstractmethod
    def num_identity_coeffs(self):
        """Number of identity coefficients expected by ``get_rest_shape``."""
        ...

    @property
    def num_scale_params(self):
        """Number of scale parameters expected by ``get_rest_shape``, or ``None`` if unused."""
        return None

    @abstractmethod
    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        """Return the rest shape in NATIVE_UNIT scale."""
        pass

    def _setup_topology_transfer(self, V_source, F_source, V_soma):
        """Set up barycentric interpolation from source topology to SOMA topology.

        Use this when the source mesh already has well-defined inner-face
        geometry (e.g. Anny) and no Laplacian blending is needed.
        All vertex data must be in the model's native units.
        """
        self._to_soma_interp = BarycentricInterpolator(V_source, F_source, V_soma)
        self._laplacian_mesh = None

    def _setup_topology_transfer_with_blending(
        self, V_source, F_source, V_soma, F_soma, vertex_ids_to_exclude
    ):
        """Set up barycentric interpolation plus Laplacian blending.

        Use this when the source mesh lacks inner-face geometry (e.g. eye bags,
        mouth bag) and the excluded vertices need to be solved via a Laplacian
        system to blend smoothly with the surrounding surface.
        All vertex data must be in the model's native units.
        """
        self._to_soma_interp = BarycentricInterpolator(V_source, F_source, V_soma)
        mask_anchors = torch.ones(V_soma.shape[0], dtype=torch.bool, device=self.device)
        mask_anchors[vertex_ids_to_exclude] = False
        self._laplacian_mesh = LaplacianMesh(V_soma, F_soma, mask_anchors=mask_anchors)

    def identity_model_to_soma(self, identity_rest_shape):
        """Transform from source topology to SOMA topology (with optional Laplacian blending)."""
        if hasattr(self, "_to_soma_interp"):
            soma_verts = self._to_soma_interp(identity_rest_shape)
            if self._laplacian_mesh is not None:
                soma_verts = self._laplacian_mesh.solve(soma_verts)
            return soma_verts
        return identity_rest_shape

    def _apply_coord_transform(self, verts: torch.Tensor) -> torch.Tensor:
        """Reorder/negate axes from the model's native convention to SOMA (Y+ up, Z+ forward)."""
        if self.NATIVE_UP == CoordAxis.Y and self.NATIVE_FORWARD == CoordAxis.Z:
            return verts  # already in SOMA frame, no-op
        up_idx, up_sign = self.NATIVE_UP
        fwd_idx, fwd_sign = self.NATIVE_FORWARD
        right_idx = 3 - up_idx - fwd_idx  # 0+1+2=3, so the remaining index
        parity = _PERM_PARITY[(right_idx, up_idx, fwd_idx)]
        right_sign = parity * up_sign * fwd_sign  # ensures det(transform) = +1
        right = verts[..., right_idx : right_idx + 1] * right_sign
        up = verts[..., up_idx : up_idx + 1] * up_sign
        fwd = verts[..., fwd_idx : fwd_idx + 1] * fwd_sign
        return torch.cat([right, up, fwd], dim=-1)

    def forward(self, identity_coeffs, scale_params=None, kwargs=None, global_scale=1.0):
        identity_rest_shape = self.get_rest_shape(identity_coeffs, scale_params, kwargs)
        result = self.identity_model_to_soma(identity_rest_shape)
        result = self._apply_coord_transform(result)
        if self._unit_conversion != 1.0:
            result = result * self._unit_conversion
        if isinstance(global_scale, torch.Tensor):
            result = result * global_scale.reshape(-1, 1, 1)
        elif global_scale != 1.0:
            result = result * global_scale
        return result


class MHRIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.CENTIMETERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self):
        return 45

    @property
    def num_scale_params(self):
        # 68 body-part scales.  The MHR TorchScript model expects 204
        # model_parameters = 136 pose + 68 scale.
        return 68

    def __init__(self, data_root, low_lod, device, **kwargs):
        vertex_ids_to_exclude = kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)

        lod = "lod1" if not low_lod else "lod6"
        self.identity_model = NonPersistentModuleWrapper(
            torch.jit.load(
                self.data_root / "MHR" / f"mhr_model_{lod}.pt",
                map_location=self.device,
            )
        )

        mesh_mhr = trimesh.load(
            self.data_root / "MHR" / f"base_body_{lod}.obj", maintain_order=True, process=False
        )
        V_mhr = torch.from_numpy(mesh_mhr.vertices).float().to(device)
        F_mhr = torch.from_numpy(mesh_mhr.faces).to(device)
        mesh_soma = trimesh.load(
            self.data_root / "MHR" / "SOMA_wrap_lod1.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(
            V_mhr, F_mhr, V_soma, F_soma, vertex_ids_to_exclude
        )

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        """Return the rest shape in centimeters (native MHR unit)."""
        assert scale_params is not None, "scale_params is required for MHR"
        B = identity_coeffs.shape[0]
        pose_params = torch.zeros(B, 136).to(identity_coeffs.device)
        face_expr_params = torch.zeros(B, 72).to(identity_coeffs.device)
        identity_rest_shape, _ = self.identity_model(
            identity_coeffs,
            torch.cat([pose_params, scale_params], dim=1),
            face_expr_params,
        )
        return identity_rest_shape


class AnnyIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Z
    NATIVE_FORWARD = CoordAxis.NEG_Y

    @property
    def num_identity_coeffs(self):
        return len(self.identity_model.phenotype_labels)

    def __init__(self, data_root, low_lod, device, **kwargs):
        # Anny mesh has mouth bag and eye bags so no need to exclude them
        kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)

        # TODO: reduce Anny's forward pass to just shape parameters
        import anny

        anny_model = anny.create_fullbody_model(
            all_phenotypes=True, local_changes=True, remove_unattached_vertices=True
        )
        self.identity_model = AnnySimplified(anny_model, device)
        mesh_anny = trimesh.load(
            self.data_root / "Anny" / "base_body.obj", maintain_order=True, process=False
        )
        V_anny = torch.from_numpy(mesh_anny.vertices).float().to(device)
        F_anny = torch.from_numpy(mesh_anny.faces).to(device)
        mesh_soma = trimesh.load(
            self.data_root / "Anny" / "SOMA_wrap.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        V_soma, _ = self._apply_soma_lod(V_soma)
        self._setup_topology_transfer(V_anny, F_anny, V_soma)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        rest_shape = self.identity_model(
            phenotype_kwargs=identity_coeffs, local_changes_kwargs=scale_params
        )
        return rest_shape


class SMPLIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self):
        return self.identity_model.num_betas

    def __init__(self, data_root, low_lod, device, model_type="smpl", **kwargs):
        vertex_ids_to_exclude = kwargs.pop("vertex_ids_to_exclude", None)
        smpl_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("output_unit", "nv_lod_mid_to_low", "soma_low_lod_faces", "model_path")
        }
        super().__init__(data_root, low_lod, device, **kwargs)

        try:
            import smplx
        except ImportError as e:
            raise ImportError(
                "SMPL/SMPL-X support requires 'smplx' and 'chumpy'. Install with:\n"
                "  pip install smplx\n"
                "  pip install --no-build-isolation chumpy\n"
                "If that fails, install chumpy from source:\n"
                "  pip install --no-build-isolation git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa8"
            ) from e

        imt = model_type
        model_kwargs = smpl_kwargs if smpl_kwargs else {}
        gender = kwargs.pop("gender", "neutral")
        explicit_model_path = kwargs.pop("model_path", None)

        if explicit_model_path is not None:
            model_path = Path(explicit_model_path).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"SMPL model not found at '{model_path}'")
            print(f"Loading {imt.upper()} model from {model_path}")
        else:
            model_dir = self.data_root / imt.upper()
            model_path_npz = model_dir / f"{imt.upper()}_{gender.upper()}.npz"
            model_path_pkl = model_dir / f"{imt.upper()}_{gender.upper()}.pkl"
            if model_path_npz.exists():
                model_path = model_path_npz
                print(f"Loading {imt.upper()} model from {model_path_npz}")
            elif model_path_pkl.exists():
                model_path = model_path_pkl
                print(f"Loading {imt.upper()} model from {model_path_pkl}")
            else:
                raise FileNotFoundError(
                    f"Neither {model_path_npz} nor {model_path_pkl} found. Cannot load {imt.upper()} model.\n"
                    "Pass model_path via identity_model_kwargs, or place the file in "
                    f"<data_root>/{imt.upper()}/."
                )

        smpl_base_model = smplx.create(
            model_type=imt,
            model_path=model_path,
            device=self.device,
            ext=model_path.suffix[1:],
            **model_kwargs,
        ).to(self.device)

        self.identity_model = SMPLSimplified(smpl_base_model, self.device)

        mesh_smpl = trimesh.load(
            self.data_root / imt.upper() / "base_body.obj",
            maintain_order=True,
            process=False,
        )
        V_smpl = torch.from_numpy(mesh_smpl.vertices).float().to(self.device)
        F_smpl = torch.from_numpy(mesh_smpl.faces).to(self.device)
        mesh_soma = trimesh.load(
            self.data_root / imt.upper() / "SOMA_wrap.obj", maintain_order=True, process=False
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(self.device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(self.device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(
            V_smpl, F_smpl, V_soma, F_soma, vertex_ids_to_exclude
        )

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        rest_shape = self.identity_model(identity_coeffs)
        return rest_shape


class GarmentMeasurementIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.METERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self):
        return self.eigenvalues.shape[0]

    def __init__(self, data_root, low_lod, device, **kwargs):
        vertex_ids_to_exclude = kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)
        self.pca_npz_file = self.data_root / "GarmentMeasurements" / "point.npz"

        data = np.load(self.pca_npz_file, allow_pickle=False)
        self.pca_matrix = torch.from_numpy(data["pca_matrix"]).float().to(device)
        self.pca_mean = torch.from_numpy(data["pca_mean"]).float().to(device)
        self.eigenvalues = torch.from_numpy(data["eigenvalues"]).float().to(device)

        mesh_garment = trimesh.load(
            self.data_root / "GarmentMeasurements" / "mean.obj", maintain_order=True, process=False
        )
        F_garment = torch.from_numpy(mesh_garment.faces).to(device)

        V_garment_from_pca = self.pca_mean.reshape(-1, 3)

        mesh_soma = trimesh.load(
            self.data_root / "GarmentMeasurements" / "SOMA_wrap.obj",
            maintain_order=True,
            process=False,
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        F_soma = torch.from_numpy(mesh_soma.faces).to(device)
        V_soma, F_soma = self._apply_soma_lod(V_soma, F_soma)
        self._setup_topology_transfer_with_blending(
            V_garment_from_pca, F_garment, V_soma, F_soma, vertex_ids_to_exclude
        )

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        weighted_coeffs = identity_coeffs * torch.sqrt(self.eigenvalues)
        weighted_pcas = torch.matmul(weighted_coeffs, self.pca_matrix.T)
        shape_garment = self.pca_mean.unsqueeze(0) + weighted_pcas
        shape_garment = shape_garment.reshape(identity_coeffs.shape[0], -1, 3)
        return shape_garment


class SOMAIdentityModel(BaseIdentityModel):
    NATIVE_UNIT = Unit.CENTIMETERS
    NATIVE_UP = CoordAxis.Y
    NATIVE_FORWARD = CoordAxis.Z

    @property
    def num_identity_coeffs(self):
        return self.eigenvalues.shape[0]

    def __init__(self, data_root, low_lod, device, **kwargs):
        kwargs.pop("vertex_ids_to_exclude", None)
        super().__init__(data_root, low_lod, device, **kwargs)
        self.pca_npz_file = self.data_root / "SOMA_neutral.npz"

        data = np.load(self.pca_npz_file, allow_pickle=False)
        self.pca_matrix = torch.from_numpy(data["shapedirs"]).float().to(device).T
        self.pca_mean = torch.from_numpy(data["mean"]).float().to(device).flatten()
        self.eigenvalues = torch.from_numpy(data["eigenvalues"]).float().to(device)

    def get_rest_shape(self, identity_coeffs, scale_params=None, kwargs=None):
        weighted_coeffs = identity_coeffs * torch.sqrt(self.eigenvalues)
        weighted_pcas = torch.matmul(weighted_coeffs, self.pca_matrix.T)
        shape_soma = self.pca_mean.unsqueeze(0) + weighted_pcas
        shape_soma = shape_soma.reshape(identity_coeffs.shape[0], -1, 3)
        if self._nv_lod_mid_to_low is not None:
            shape_soma = shape_soma[:, self._nv_lod_mid_to_low, :]
        return shape_soma


def create_identity_model(
    identity_model_type, data_root, low_lod, device, output_unit=Unit.METERS, **kwargs
):
    """Factory function to create the appropriate identity model.

    Args:
        output_unit: Desired unit for the model's ``forward()`` output.
            Internally the model operates in its native units; conversion
            to *output_unit* is applied at the output boundary.
    """
    identity_model_type = identity_model_type.lower()

    if identity_model_type == "soma":
        return SOMAIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    if identity_model_type == "mhr":
        return MHRIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    elif identity_model_type == "anny":
        return AnnyIdentityModel(data_root, low_lod, device, output_unit=output_unit, **kwargs)
    elif identity_model_type in ["smplx", "smplh", "smpl"]:
        return SMPLIdentityModel(
            data_root,
            low_lod,
            device,
            model_type=identity_model_type,
            output_unit=output_unit,
            **kwargs,
        )
    elif identity_model_type == "garment":
        return GarmentMeasurementIdentityModel(
            data_root, low_lod, device, output_unit=output_unit, **kwargs
        )
    else:
        raise ValueError(f"Invalid identity model: {identity_model_type}")
