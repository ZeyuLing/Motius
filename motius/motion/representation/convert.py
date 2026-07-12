"""Public cross-representation conversion helpers."""

from __future__ import annotations


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


def motion272_to_hml263(motion_272, **kwargs):
    raise NotImplementedError(
        "MS272 -> HML263 is a dataset-production conversion requiring the official "
        "HumanML3D canonicalization, IK, and contact extraction pipeline"
    )


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
            "joints22": "joints",
            "smplhjoints": "joints",
            "g138": "g1_38",
            "g1motion38": "g1_38",
            "g1qpos36": "g1_qpos",
            "g1qpos": "g1_qpos",
        }
        return aliases.get(key, key)

    source = normalize(source)
    target = normalize(target)
    if source == target:
        return data

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
    raise ValueError(f"unsupported conversion target: {target!r}")


__all__ = [
    "hml263_to_joints",
    "hml263_to_motion135",
    "hml263_to_motion272",
    "motion135_to_motion272",
    "motion272_to_joints",
    "motion272_to_motion135",
    "motion135_to_joints",
    "motion272_to_hml263",
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
