"""Public cross-representation conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping


def hml263_to_joints(m263, joints_num: int = 22):
    from motius.motion.representation.humanml import hml263_to_joints as fn

    return fn(m263, joints_num=joints_num)


def motion135_to_motion272(motion_135, **kwargs):
    from motius.motion.representation.motion272 import motion135_to_272

    return motion135_to_272(motion_135, **kwargs)


def motion272_to_joints(motion_272):
    from motius.motion.representation.motion272 import motion272_to_joints

    return motion272_to_joints(motion_272)


def motion272_to_motion135(motion_272):
    from motius.motion.representation.motion272 import motion272_to_motion135 as fn

    return fn(motion_272)


def motion135_to_joints(motion_135, *, bone_offsets, rotation_space: str = "local"):
    import numpy as np
    import torch

    from motius.motion.skeleton.fk import motion135_to_fk

    is_numpy = isinstance(motion_135, np.ndarray)
    motion = torch.as_tensor(motion_135, dtype=torch.float32) if is_numpy else motion_135
    offsets = torch.as_tensor(bone_offsets, dtype=motion.dtype, device=motion.device)
    joints, _, _, _ = motion135_to_fk(motion[..., :135], offsets, rotation_space=rotation_space)
    return joints.detach().cpu().numpy() if is_numpy else joints


def joints_to_hml263(joints, **kwargs):
    from motius.motion.representation.humanml import joints_to_hml263 as fn

    return fn(joints, **kwargs)


def motion135_to_interhuman262(
    motion_135,
    *,
    bone_offsets,
    rotation_space: str = "local",
    feet_threshold: float = 0.001,
    reference_frame: int = 0,
    source_coordinates: str = "y_up",
):
    """Encode one or two SMPL-22 ``motion135`` tracks as InterHuman-262."""

    import numpy as np

    from motius.motion.representation.interhuman262 import (
        joints_pair_to_interhuman262,
        joints_to_interhuman262,
    )

    motion = np.asarray(motion_135, dtype=np.float32)
    if motion.ndim == 2 and motion.shape[-1] == 135:
        joints = motion135_to_joints(
            motion, bone_offsets=bone_offsets, rotation_space=rotation_space
        )
        return joints_to_interhuman262(
            joints,
            motion[:, 9:135],
            feet_threshold=feet_threshold,
            reference_frame=reference_frame,
            source_coordinates=source_coordinates,
        )
    if motion.ndim == 3 and motion.shape[1:] == (2, 135):
        joints = np.stack(
            [
                motion135_to_joints(
                    motion[:, person],
                    bone_offsets=bone_offsets,
                    rotation_space=rotation_space,
                )
                for person in range(2)
            ],
            axis=1,
        )
        return joints_pair_to_interhuman262(
            joints,
            motion[:, :, 9:135],
            feet_threshold=feet_threshold,
            reference_frame=reference_frame,
            source_coordinates=source_coordinates,
        )
    raise ValueError(f"motion135 must have shape (T,135) or (T,2,135), got {motion.shape}")


def _to_humanml_coordinates(joints, coordinate_system: str):
    import numpy as np

    coordinate_system = coordinate_system.lower()
    output = np.asarray(joints).copy()
    if coordinate_system in {"humanml", "humanml3d", "y_up"}:
        return output
    if coordinate_system in {"amass", "z_up"}:
        transform = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=output.dtype,
        )
        output = output @ transform
        output[..., 0] *= -1
        return output
    raise ValueError(
        "coordinate_system must be 'humanml' (Y-up) or 'amass' (Z-up), "
        f"got {coordinate_system!r}"
    )


def _resample_for_hml263(joints, *, src_fps: float, dst_fps: float, mode: str):
    import numpy as np

    positions = np.asarray(joints)
    if abs(src_fps - dst_fps) < 1e-9 or mode == "none":
        return positions.copy()
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("src_fps and dst_fps must be positive")
    ratio = src_fps / dst_fps
    integer_ratio = int(round(ratio))
    if mode == "auto" and ratio >= 1 and abs(ratio - integer_ratio) < 1e-9:
        mode = "stride"
    if mode == "stride":
        if ratio < 1 or abs(ratio - integer_ratio) >= 1e-9:
            raise ValueError(
                f"stride resampling requires an integer src/dst ratio, got {src_fps}/{dst_fps}"
            )
        return positions[::integer_ratio].copy()
    if mode not in {"auto", "linear"}:
        raise ValueError(f"resample must be auto/stride/linear/none, got {mode!r}")
    from motius.motion.representation.humanml import linear_resample_joints

    return linear_resample_joints(positions, src_fps, dst_fps)


def motion135_to_hml263(
    motion_135,
    *,
    bone_offsets,
    rotation_space: str = "local",
    src_fps: float = 20.0,
    dst_fps: float = 20.0,
    resample: str = "auto",
    coordinate_system: str = "humanml",
    feet_threshold: float = 0.002,
    target_offsets=None,
):
    """Convert ``motion135`` to HML263 through explicit SMPL-22 FK."""

    joints = motion135_to_joints(
        motion_135, bone_offsets=bone_offsets, rotation_space=rotation_space
    )
    if hasattr(joints, "detach"):
        joints = joints.detach().cpu().numpy()
    joints = _resample_for_hml263(
        joints, src_fps=src_fps, dst_fps=dst_fps, mode=resample
    )
    joints = _to_humanml_coordinates(joints, coordinate_system)
    return joints_to_hml263(
        joints, feet_threshold=feet_threshold, target_offsets=target_offsets
    )


def motion272_to_hml263(
    motion_272,
    *,
    src_fps: float = 30.0,
    dst_fps: float = 20.0,
    resample: str = "auto",
    coordinate_system: str = "humanml",
    feet_threshold: float = 0.002,
    target_offsets=None,
):
    """Convert MS272 using its native stored SMPL-22 position channels."""

    joints = motion272_to_joints(motion_272)
    joints = _resample_for_hml263(
        joints, src_fps=src_fps, dst_fps=dst_fps, mode=resample
    )
    joints = _to_humanml_coordinates(joints, coordinate_system)
    return joints_to_hml263(
        joints, feet_threshold=feet_threshold, target_offsets=target_offsets
    )


def smpl_to_motion135(global_orient, body_pose, transl):
    """Pack SMPL body parameters as translation + 22 local 6D rotations."""

    import numpy as np
    import torch

    from motius.motion.representation.rotation import (
        axis_angle_to_matrix,
        matrix_to_rotation_6d,
    )

    is_torch = torch.is_tensor(global_orient)
    is_numpy = not is_torch
    root = np.asarray(global_orient) if is_numpy else global_orient
    frames = root.shape[0]
    pose = np.asarray(body_pose) if is_numpy else body_pose
    pose = pose.reshape(frames, -1, 3)
    if pose.shape[1] < 21:
        raise ValueError(f"body_pose needs at least 21 joints, got {pose.shape}")
    local_axis_angle = (
        np.concatenate([root.reshape(frames, 1, 3), pose[:, :21]], axis=1)
        if is_numpy
        else torch.cat([root.reshape(frames, 1, 3), pose[:, :21]], dim=1)
    )
    flat_axis_angle = local_axis_angle.reshape(-1, 3)
    rotations = matrix_to_rotation_6d(
        axis_angle_to_matrix(flat_axis_angle).reshape(frames, 22, 3, 3),
        convention="row",
    ).reshape(frames, 132)
    translation = np.asarray(transl) if is_numpy else transl
    translation = translation.reshape(frames, 3)
    return (
        np.concatenate([translation, rotations], axis=-1).astype(np.float32)
        if is_numpy
        else torch.cat([translation, rotations], dim=-1)
    )


def smpl_to_joints(
    global_orient,
    body_pose,
    transl,
    *,
    betas=None,
    gender: str = "neutral",
    model_type: str = "smplh",
    model_path,
):
    """Public shape-aware SMPL-family parameters to SMPL-22 joints API."""

    from motius.motion.skeleton.body_models import smpl_to_joints as fn

    return fn(
        global_orient,
        body_pose,
        transl,
        betas=betas,
        gender=gender,
        model_type=model_type,
        model_path=model_path,
    )


def smpl_to_hml263(
    global_orient,
    body_pose,
    transl,
    *,
    betas=None,
    gender: str = "neutral",
    model_type: str = "smplh",
    model_path,
    src_fps: float = 20.0,
    dst_fps: float = 20.0,
    resample: str = "auto",
    coordinate_system: str = "humanml",
    feet_threshold: float = 0.002,
    target_offsets=None,
):
    """Convert SMPL-family parameters to official HumanML3D-263 features.

    ``betas`` and ``gender`` are applied while materializing the source joints.
    For AMASS parameters, pass ``coordinate_system="amass"``. Integer frame-rate
    reductions use phase-aligned striding by default, matching HumanML3D's AMASS
    preprocessing; other ratios use linear joint interpolation.
    """

    joints = smpl_to_joints(
        global_orient,
        body_pose,
        transl,
        betas=betas,
        gender=gender,
        model_type=model_type,
        model_path=model_path,
    )
    joints = _resample_for_hml263(
        joints, src_fps=src_fps, dst_fps=dst_fps, mode=resample
    )
    joints = _to_humanml_coordinates(joints, coordinate_system)
    return joints_to_hml263(
        joints, feet_threshold=feet_threshold, target_offsets=target_offsets
    )


def smpl_to_humanml263(*args, **kwargs):
    """Descriptive alias of :func:`smpl_to_hml263`."""

    return smpl_to_hml263(*args, **kwargs)


def _smpl_parameters(data: Mapping):
    import numpy as np

    if "poses" in data:
        poses = np.asarray(data["poses"])
        global_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
    else:
        global_orient = data["global_orient"]
        body_pose = data["body_pose"]
    transl = data.get("transl", data.get("trans"))
    if transl is None:
        raise KeyError("SMPL input needs 'transl' or 'trans'")
    return global_orient, body_pose, transl


def _metadata_string(value) -> str:
    import numpy as np

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar metadata, got shape {array.shape}")
    item = array.reshape(()).item()
    return item.decode() if isinstance(item, bytes) else str(item)


def motion135_to_hymotion201(motion_135, bone_offsets):
    from motius.motion.representation.hymotion import motion135_to_hymotion201 as fn

    return fn(motion_135, bone_offsets)


def hymotion201_to_motion135(motion_201):
    from motius.motion.representation.hymotion import hymotion201_to_motion135 as fn

    return fn(motion_201)


def hymotion201_to_joints(motion_201, **kwargs):
    from motius.motion.representation.hymotion import hymotion201_to_joints as fn

    return fn(motion_201, **kwargs)


def hml263_to_motion135(m263, **kwargs):
    from motius.motion.retarget.hml263_smpl import hml263_to_motion135 as fn

    return fn(m263, **kwargs)


def hml263_to_motion272(m263, ik_kwargs: dict | None = None, **kwargs):
    motion135 = hml263_to_motion135(m263, **dict(ik_kwargs or {}))
    return motion135_to_motion272(motion135, **kwargs)


def dart276_to_smpl_params(motion, **kwargs):
    from motius.motion.representation.dart276 import dart276_to_smpl_params as fn

    return fn(motion, **kwargs)


def dart276_to_joints(motion, **kwargs):
    from motius.motion.representation.dart276 import dart276_to_joints as fn

    return fn(motion, **kwargs)


def dart276_to_motion135(motion, **kwargs):
    from motius.motion.representation.dart276 import dart276_to_motion135 as fn

    return fn(motion, **kwargs)


def smpl_params_and_joints_to_dart276(smpl_params, joints, **kwargs):
    from motius.motion.representation.dart276 import smpl_params_and_joints_to_dart276 as fn

    return fn(smpl_params, joints, **kwargs)


def dart276_to_motion272(motion, *, motion135_kwargs: dict | None = None, **kwargs):
    """DART276 -> repository row-major motion_135 -> MotionStreamer/MS272."""

    motion135_kwargs = dict(motion135_kwargs or {})
    motion135_kwargs.setdefault("rotation_convention", "row")
    m135 = dart276_to_motion135(motion, **motion135_kwargs)
    return motion135_to_motion272(m135, **kwargs)


def convert_motion(data, source: str, target: str, **kwargs):
    """Convert one time-major motion array through a supported public route.

    The dispatcher only exposes routes with explicit semantics. Conversions
    that require a skeleton accept ``bone_offsets``; HML263 -> motion135 also
    requires SMPL assets through ``model_dir`` or ``MOTIUS_SMPL_MODEL_DIR``.
    """

    def normalize(name: str) -> str:
        key = name.lower().replace("-", "").replace("_", "")
        aliases = {
            "humanml3d263": "hml263",
            "humanml263": "hml263",
            "motionstreamer272": "ms272",
            "motion272": "ms272",
            "hymotion201": "hymotion201",
            "interhuman": "interhuman262",
            "interhuman262": "interhuman262",
            "intergen262": "interhuman262",
            "joints22": "joints",
            "smplhjoints": "joints",
            "smplparams": "smpl",
            "smplhparams": "smpl",
            "g138": "g1_38",
            "g1motion38": "g1_38",
            "g1qpos36": "g1_qpos",
            "g1qpos": "g1_qpos",
            "ardycore330": "ardy_core330",
            "ardyg1414": "ardy_g1_414",
            "smpl22joints": "smpl22_joints",
            "smpljoints": "smpl22_joints",
        }
        return aliases.get(key, key)

    source = normalize(source)
    target = normalize(target)
    if source == target:
        return data

    if source in {"ardy_core330", "ardy_g1_414"}:
        motion_rep = kwargs.get("motion_rep")
        if motion_rep is None:
            raise ValueError(
                f"{source} conversion requires the checkpoint's exact motion_rep object"
            )
        from motius.motion.representation.ardy import decode_ardy_features

        output = decode_ardy_features(
            data,
            motion_rep=motion_rep,
            is_normalized=kwargs.get("is_normalized", True),
            return_numpy=kwargs.get("return_numpy", False),
        )
        if target == "joints":
            return output["posed_joints"]
        if target == "smpl22_joints" and source == "ardy_core330":
            from motius.motion.retarget.ardy_core import ardy_core27_to_smpl22_joints

            return ardy_core27_to_smpl22_joints(
                output["posed_joints"],
                recenter_root=kwargs.get("recenter_root", False),
            )
        if target == "g1_qpos" and source == "ardy_g1_414":
            from motius.models.ardy.network.exports.mujoco import MujocoQposConverter

            return MujocoQposConverter(motion_rep.skeleton).dict_to_qpos(
                output,
                device=kwargs.get("device"),
                numpy=kwargs.get("return_numpy", False),
            )
        targets = (
            "joints and g1_qpos"
            if source == "ardy_g1_414"
            else "joints and smpl22_joints"
        )
        raise ValueError(f"{source} supports exact conversion only to {targets}")

    if source == "g1_38" and target == "g1_qpos":
        import numpy as np
        import torch

        from motius.motion.representation.g1 import decode_g1_to_qpos

        is_numpy = isinstance(data, np.ndarray)
        tensor = torch.as_tensor(data, dtype=torch.float32) if is_numpy else data
        result = decode_g1_to_qpos(tensor, root_velocity=kwargs.get("root_velocity", True))
        return result.detach().cpu().numpy() if is_numpy else result
    if source == "g1_qpos" and target == "g1_38":
        from motius.motion.representation.g1 import encode_g1_qpos

        return encode_g1_qpos(
            data,
            canonicalize=kwargs.get("canonicalize", True),
            root_velocity=kwargs.get("root_velocity", True),
        )

    if source == "joints":
        if target == "interhuman262":
            from motius.motion.representation.interhuman262 import (
                joints_pair_to_interhuman262,
                joints_to_interhuman262,
            )

            if "local_rot6d" not in kwargs:
                raise ValueError("joints -> InterHuman-262 requires local_rot6d")
            import numpy as np

            value = np.asarray(data)
            encoder = joints_pair_to_interhuman262 if value.ndim == 4 else joints_to_interhuman262
            return encoder(
                value,
                kwargs["local_rot6d"],
                feet_threshold=kwargs.get("feet_threshold", 0.001),
                reference_frame=kwargs.get("reference_frame", 0),
                source_coordinates=kwargs.get("source_coordinates", "interhuman_raw"),
            )
        if target != "hml263":
            raise ValueError("raw joints support hml263 and interhuman262 targets")
        accepted = {
            key: kwargs[key]
            for key in ("feet_threshold", "target_offsets")
            if key in kwargs
        }
        return joints_to_hml263(data, **accepted)

    if source == "smpl":
        if not isinstance(data, Mapping):
            raise TypeError("source='smpl' expects a mapping of SMPL parameter arrays")
        global_orient, body_pose, transl = _smpl_parameters(data)
        if target == "motion135":
            return smpl_to_motion135(global_orient, body_pose, transl)
        if target not in {"joints", "hml263"}:
            data = smpl_to_motion135(global_orient, body_pose, transl)
            source = "motion135"
        else:
            common = {
                "betas": kwargs.get("betas", data.get("betas")),
                "gender": _metadata_string(
                    kwargs.get("gender", data.get("gender", "neutral"))
                ),
                "model_type": _metadata_string(
                    kwargs.get(
                        "model_type",
                        data.get("model_type", data.get("smpl_type", "smplh")),
                    )
                ),
                "model_path": kwargs.get("model_path", kwargs.get("model_dir")),
            }
            if common["model_path"] is None:
                raise ValueError("SMPL conversion to joints/HML263 requires model_path")
            if target == "joints":
                return smpl_to_joints(global_orient, body_pose, transl, **common)
            accepted = {
                key: kwargs[key]
                for key in (
                    "src_fps",
                    "dst_fps",
                    "resample",
                    "coordinate_system",
                    "feet_threshold",
                    "target_offsets",
                )
                if key in kwargs
            }
            return smpl_to_hml263(
                global_orient, body_pose, transl, **common, **accepted
            )

    if source == "hml263":
        if target == "joints":
            return hml263_to_joints(data)
        source_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"bone_offsets", "rotation_space", "skeleton"}
        }
        data = hml263_to_motion135(data, **source_kwargs)
        source = "motion135"
    elif source == "ms272":
        if target == "joints":
            return motion272_to_joints(data)
        if target == "hml263":
            accepted = {
                key: kwargs[key]
                for key in (
                    "src_fps",
                    "dst_fps",
                    "resample",
                    "coordinate_system",
                    "feet_threshold",
                    "target_offsets",
                )
                if key in kwargs
            }
            return motion272_to_hml263(data, **accepted)
        data = motion272_to_motion135(data)
        source = "motion135"
    elif source == "hymotion201":
        if target == "joints":
            accepted = {"source": kwargs.get("joint_source", "stored")}
            if "bone_offsets" in kwargs:
                accepted["bone_offsets"] = kwargs["bone_offsets"]
            return hymotion201_to_joints(data, **accepted)
        data = hymotion201_to_motion135(data)
        source = "motion135"
    elif source == "dart276":
        if target == "joints":
            accepted = {key: kwargs[key] for key in ("recover_from_velocity", "equal_length", "coord") if key in kwargs}
            return dart276_to_joints(data, **accepted)
        accepted = {
            key: kwargs[key]
            for key in (
                "recover_from_velocity",
                "equal_length",
                "coord_conversion",
                "translation_source",
                "rotation_convention",
            )
            if key in kwargs
        }
        data = dart276_to_motion135(data, **accepted)
        source = "motion135"
    elif source == "interhuman262":
        if target != "joints":
            raise ValueError(
                "InterHuman-262 currently decodes exactly to joints; rotation-complete "
                "SMPL recovery requires the documented position-IK bridge"
            )
        from motius.motion.representation.interhuman262 import interhuman262_to_joints

        return interhuman262_to_joints(data)

    if source != "motion135":
        raise ValueError(f"unsupported source representation: {source!r}")
    if target == "motion135":
        return data
    if target == "joints":
        if "bone_offsets" not in kwargs:
            raise ValueError("motion135 -> joints requires bone_offsets")
        return motion135_to_joints(
            data,
            bone_offsets=kwargs["bone_offsets"],
            rotation_space=kwargs.get("rotation_space", "local"),
        )
    if target == "ms272":
        accepted = {k: kwargs[k] for k in ("rotation_space", "bone_offsets", "skeleton") if k in kwargs}
        return motion135_to_motion272(data, **accepted)
    if target == "hymotion201":
        if "bone_offsets" not in kwargs:
            raise ValueError("motion135 -> HY-Motion-201 requires bone_offsets")
        return motion135_to_hymotion201(data, kwargs["bone_offsets"])
    if target == "hml263":
        if "bone_offsets" not in kwargs:
            raise ValueError("motion135 -> HML263 requires bone_offsets")
        accepted = {
            key: kwargs[key]
            for key in (
                "rotation_space",
                "src_fps",
                "dst_fps",
                "resample",
                "coordinate_system",
                "feet_threshold",
                "target_offsets",
            )
            if key in kwargs
        }
        return motion135_to_hml263(
            data, bone_offsets=kwargs["bone_offsets"], **accepted
        )
    if target == "interhuman262":
        if "bone_offsets" not in kwargs:
            raise ValueError("motion135 -> InterHuman-262 requires bone_offsets")
        accepted = {
            key: kwargs[key]
            for key in (
                "rotation_space",
                "feet_threshold",
                "reference_frame",
                "source_coordinates",
            )
            if key in kwargs
        }
        return motion135_to_interhuman262(
            data, bone_offsets=kwargs["bone_offsets"], **accepted
        )
    raise ValueError(f"unsupported conversion target: {target!r}")


__all__ = [
    "hml263_to_joints",
    "hml263_to_motion135",
    "hml263_to_motion272",
    "motion135_to_motion272",
    "motion272_to_joints",
    "motion272_to_motion135",
    "motion135_to_joints",
    "joints_to_hml263",
    "motion135_to_hml263",
    "motion135_to_interhuman262",
    "motion272_to_hml263",
    "smpl_to_motion135",
    "smpl_to_joints",
    "smpl_to_hml263",
    "smpl_to_humanml263",
    "motion135_to_hymotion201",
    "hymotion201_to_motion135",
    "hymotion201_to_joints",
    "dart276_to_smpl_params",
    "dart276_to_joints",
    "dart276_to_motion135",
    "dart276_to_motion272",
    "smpl_params_and_joints_to_dart276",
    "convert_motion",
]
