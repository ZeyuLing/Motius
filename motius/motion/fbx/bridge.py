"""Bridge every public Motius representation to a rigged character FBX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motius.motion.fbx.api import FBXExportResult, SMPLAnimation, retarget_smpl_to_fbx
from motius.motion.fbx.characters import resolve_character_fbx
from motius.motion.representation import get_spec


_NON_SPEC_ALIASES = {
    "smpl": "smpl",
    "smplparams": "smpl",
    "smplhparams": "smpl",
    "joints": "smpl22_joints",
    "joints22": "smpl22_joints",
    "smpljoints": "smpl22_joints",
    "smpl22joints": "smpl22_joints",
    "g1qpos": "g1_qpos",
    "g1qpos36": "g1_qpos",
}


@dataclass(frozen=True)
class MotionBridgeResult:
    """One representation decoded as an animatable SMPL-22 body."""

    animation: SMPLAnimation
    source_representation: str
    bridge: str
    lossy: bool
    source_fps: float
    output_fps: float
    person_index: int | None
    fit_mpjpe_mm: np.ndarray | None
    metadata: Mapping[str, object]


def _normalize_source(name: str) -> str:
    key = str(name).casefold().replace("-", "").replace("_", "")
    if key in _NON_SPEC_ALIASES:
        return _NON_SPEC_ALIASES[key]
    try:
        return get_spec(name).name
    except KeyError as error:
        raise ValueError(str(error)) from error


def _default_fps(source: str) -> float:
    if source in {"smpl", "smpl22_joints", "g1_qpos"}:
        return 30.0
    value = get_spec(source).fps
    if value is None:
        raise ValueError(f"source_fps is required for {source!r}.")
    return float(value)


def _resample_animation(animation: SMPLAnimation, output_fps: float) -> SMPLAnimation:
    if abs(animation.fps - output_fps) < 1e-9 or animation.frames == 1:
        return SMPLAnimation(
            animation.local_rotations,
            animation.translations,
            animation.betas,
            output_fps,
        )
    duration = (animation.frames - 1) / animation.fps
    frames = max(2, int(round(duration * output_fps)) + 1)
    source_time = np.arange(animation.frames, dtype=np.float64) / animation.fps
    target_time = np.linspace(0.0, duration, frames)
    translation = np.stack(
        [
            np.interp(target_time, source_time, animation.translations[:, axis])
            for axis in range(3)
        ],
        axis=-1,
    )
    rotations = np.empty(
        (frames, animation.local_rotations.shape[1], 3, 3), dtype=np.float64
    )
    for joint in range(animation.local_rotations.shape[1]):
        rotations[:, joint] = Slerp(
            source_time,
            Rotation.from_matrix(
                np.array(animation.local_rotations[:, joint], copy=True)
            ),
        )(target_time).as_matrix()
    return SMPLAnimation(rotations, translation, animation.betas, output_fps)


def _ik_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    return path.parent if path.is_file() else path


def _fit_joint_positions(
    joints,
    *,
    model_path: str | Path,
    source_fps: float,
    output_fps: float,
    betas,
    gender: str,
    bridge_kwargs: Mapping[str, object],
) -> tuple[SMPLAnimation, np.ndarray, dict[str, object]]:
    from motius.motion.retarget.hml263_smpl import retarget_hml263_clip

    accepted = {
        key: bridge_kwargs[key]
        for key in (
            "batch_size",
            "device",
            "floor_align",
            "refine_iters",
            "refine_lr",
            "orientation_mode",
            "parent_ref_weight",
            "pose_keep_weight",
            "pose_l2_weight",
            "angle_prior_weight",
        )
        if key in bridge_kwargs
    }
    fit = retarget_hml263_clip(
        np.asarray(joints, dtype=np.float32),
        model_dir=_ik_model_dir(model_path),
        source_fps=source_fps,
        target_fps=output_fps,
        rotation_init="position_ik",
        gender=gender,
        **accepted,
    )
    animation = SMPLAnimation.from_smpl(
        fit["global_orient"],
        fit["body_pose"],
        fit["transl"],
        betas=betas,
        fps=output_fps,
    )
    errors = np.asarray(fit["fit_mpjpe_mm"], dtype=np.float32)
    return animation, errors, {
        "fit_mpjpe_mm_mean": float(errors.mean()),
        "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
        "fit_mpjpe_mm_max": float(errors.max()),
        "rotation_init": "position_ik",
        "ik_body_shape": "zero-beta SMPL",
        "ik_body_gender": gender,
        "output_betas_applied": betas is not None,
    }


_G1_ALIASES = {
    "Pelvis": ("pelvis", "pelvis_skel"),
    "L_Hip": ("left_hip_roll_link", "left_hip_pitch_skel"),
    "R_Hip": ("right_hip_roll_link", "right_hip_pitch_skel"),
    "Spine1": ("waist_yaw_link", "waist_yaw_skel"),
    "L_Knee": ("left_knee_link", "left_knee_skel"),
    "R_Knee": ("right_knee_link", "right_knee_skel"),
    "Spine2": ("waist_roll_link", "waist_roll_skel"),
    "L_Ankle": ("left_ankle_roll_link", "left_ankle_roll_skel"),
    "R_Ankle": ("right_ankle_roll_link", "right_ankle_roll_skel"),
    "Spine3": ("torso_link", "waist_pitch_skel"),
    "L_Foot": ("left_toe_link", "left_toe_base"),
    "R_Foot": ("right_toe_link", "right_toe_base"),
    "L_Collar": ("left_shoulder_pitch_link", "left_shoulder_pitch_skel"),
    "R_Collar": ("right_shoulder_pitch_link", "right_shoulder_pitch_skel"),
    "L_Shoulder": ("left_shoulder_yaw_link", "left_shoulder_yaw_skel"),
    "R_Shoulder": ("right_shoulder_yaw_link", "right_shoulder_yaw_skel"),
    "L_Elbow": ("left_elbow_link", "left_elbow_skel"),
    "R_Elbow": ("right_elbow_link", "right_elbow_skel"),
    "L_Wrist": ("left_wrist_yaw_link", "left_wrist_yaw_skel"),
    "R_Wrist": ("right_wrist_yaw_link", "right_wrist_yaw_skel"),
}


def g1_joints_to_smpl22_joints(
    joints,
    joint_names,
    *,
    coordinate_system: str = "y_up",
) -> np.ndarray:
    """Map named G1 joints to a directional SMPL-22 position-IK target."""

    value = np.asarray(joints, dtype=np.float32)
    names = [str(name) for name in joint_names]
    if value.shape[-2:] != (len(names), 3):
        raise ValueError(
            f"G1 joints must end in ({len(names)},3), got {value.shape}."
        )
    index = {name: position for position, name in enumerate(names)}

    def find(smpl_name: str) -> np.ndarray:
        for alias in _G1_ALIASES[smpl_name]:
            if alias in index:
                return value[..., index[alias], :]
        raise ValueError(
            f"G1 skeleton lacks a joint for {smpl_name}: {_G1_ALIASES[smpl_name]}."
        )

    mapped: dict[str, np.ndarray] = {
        name: find(name) for name in _G1_ALIASES
    }
    if "head_link" in index:
        mapped["Head"] = value[..., index["head_link"], :]
        mapped["Neck"] = 0.5 * (mapped["Head"] + mapped["Spine3"])
    else:
        mapped["Neck"] = 0.5 * (mapped["L_Collar"] + mapped["R_Collar"])
        mapped["Head"] = mapped["Neck"] + 0.65 * (
            mapped["Neck"] - mapped["Spine3"]
        )
    from motius.motion.skeleton.names import SMPL22_NAMES

    output = np.stack([mapped[name] for name in SMPL22_NAMES], axis=-2)
    if coordinate_system.casefold() in {"z_up", "mujoco"}:
        from motius.motion.retarget.smpl_g1 import GMR_Y_UP_FROM_Z_UP

        output = output @ GMR_Y_UP_FROM_Z_UP.T
    elif coordinate_system.casefold() not in {"y_up", "ardy", "motionbricks"}:
        raise ValueError("coordinate_system must be y_up or z_up/mujoco.")
    return output.astype(np.float32)


def _g1_qpos_to_named_joints(qpos, robot_xml: str | Path | None):
    try:
        import mujoco
    except ImportError as error:
        raise ImportError("G1 FBX export requires the optional mujoco package.") from error
    if robot_xml is None:
        from motius.motion.retarget.smpl_g1 import _GMR_ROBOT_XML

        robot_xml = _GMR_ROBOT_XML["unitree_g1"]
    model = mujoco.MjModel.from_xml_path(str(Path(robot_xml).resolve()))
    state = mujoco.MjData(model)
    motion = np.asarray(qpos, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != model.nq:
        raise ValueError(f"G1 qpos must have shape (T,{model.nq}), got {motion.shape}.")
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(1, model.nbody)
    ]
    names = [name for name in names if name]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
    joints = np.empty((len(motion), len(names), 3), dtype=np.float32)
    for frame, frame_qpos in enumerate(motion):
        state.qpos[:] = frame_qpos
        mujoco.mj_forward(model, state)
        joints[frame] = state.xpos[body_ids]
    return joints, names


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _motion_rep_joints(
    data,
    motion_rep,
    source: str,
    *,
    is_normalized: bool,
):
    if motion_rep is None:
        raise ValueError("This representation requires its checkpoint motion_rep object.")

    feature_dim = int(np.shape(data)[-1])
    expected_dim = int(get_spec(source).dim)
    if feature_dim != expected_dim:
        raise ValueError(
            f"{source} features must end in {expected_dim} channels, got {feature_dim}."
        )
    decoder = motion_rep
    decoder_dim = int(getattr(decoder, "motion_rep_dim", feature_dim))
    if decoder_dim != feature_dim:
        mode = {
            "motionbricks_g1_414": "global",
            "motionbricks_g1_413": "local",
        }.get(source)
        if mode is None or not hasattr(decoder, "get_motion_rep_subset"):
            raise ValueError(
                f"motion_rep has {decoder_dim} channels but {source} has {feature_dim}."
            )
        decoder = decoder.get_motion_rep_subset(mode)
        decoder_dim = int(getattr(decoder, "motion_rep_dim", -1))
        if decoder_dim != feature_dim:
            raise ValueError(
                f"The {mode} motion_rep subset has {decoder_dim} channels, expected "
                f"{feature_dim}."
            )

    output = decoder.inverse(
        data,
        is_normalized=is_normalized,
        return_numpy=False,
    )
    skeleton = decoder.skeleton
    return _as_numpy(output["posed_joints"]), tuple(skeleton.bone_order_names)


def motion_to_smpl_animation(
    data,
    source_representation: str,
    *,
    model_path: str | Path,
    gender: str = "neutral",
    source_fps: float | None = None,
    output_fps: float | None = None,
    betas=None,
    person_index: int | None = None,
    motion_rep=None,
    is_normalized: bool = True,
    robot_xml: str | Path | None = None,
    bridge_kwargs: Mapping[str, object] | None = None,
) -> MotionBridgeResult:
    """Decode a public Motius representation as an SMPL animation."""

    source = _normalize_source(source_representation)
    source_fps = float(source_fps or _default_fps(source))
    output_fps = float(output_fps or source_fps)
    options = dict(bridge_kwargs or {})
    fit_errors = None
    extra: dict[str, object] = {}
    lossy = False

    if source == "smpl":
        if not isinstance(data, Mapping):
            raise TypeError("source='smpl' expects a mapping of SMPL parameter arrays.")
        pose = data.get("poses")
        if pose is not None:
            pose = np.asarray(pose)
            root, body = pose[:, :3], pose[:, 3:]
        else:
            root, body = data["global_orient"], data["body_pose"]
        transl = data.get("transl", data.get("trans"))
        if transl is None:
            raise KeyError("SMPL input needs 'transl' or 'trans'.")
        animation = SMPLAnimation.from_smpl(
            root,
            body,
            transl,
            betas=betas if betas is not None else data.get("betas"),
            fps=source_fps,
        )
        bridge = "raw SMPL parameters"
    elif source in {"motion135", "hymotion201", "ms272", "dart276", "babel135"}:
        if source == "motion135":
            motion135 = data
            bridge = "native motion135"
        elif source == "hymotion201":
            from motius.motion.representation.hymotion import hymotion201_to_motion135

            motion135 = hymotion201_to_motion135(data)
            bridge = "exact HY-Motion prefix"
        elif source == "ms272":
            from motius.motion.representation.motion272 import motion272_to_motion135

            motion135 = motion272_to_motion135(data)
            bridge = "MS272 root and rotation decode"
        elif source == "dart276":
            from motius.motion.representation.dart276 import dart276_to_motion135

            dart_options = {
                key: options[key]
                for key in (
                    "recover_from_velocity",
                    "equal_length",
                    "coord_conversion",
                    "translation_source",
                    "rotation_convention",
                )
                if key in options
            }
            motion135 = dart276_to_motion135(data, **dart_options)
            bridge = "DART coordinate and rotation decode"
        else:
            from motius.motion.representation.babel135 import babel135_to_motion135

            babel_options = {
                key: options[key]
                for key in ("mean", "std", "target_up_axis")
                if key in options
            }
            motion135 = babel135_to_motion135(data, **babel_options)
            bridge = "BABEL root integration and rotation decode"
        animation = SMPLAnimation.from_motion135(
            motion135, betas=betas, fps=source_fps
        )
    else:
        joints = None
        if source == "hml263":
            from motius.motion.retarget.hml263_smpl import retarget_hml263_clip

            hml_options = {
                key: options[key]
                for key in (
                    "batch_size",
                    "device",
                    "floor_align",
                    "refine_iters",
                    "refine_lr",
                    "rotation_init",
                    "orientation_mode",
                    "parent_ref_weight",
                    "pose_keep_weight",
                    "pose_l2_weight",
                    "angle_prior_weight",
                    "target_len",
                    "mean",
                    "std",
                    "source_motion135_transl",
                    "root_translation_mode",
                    "lock_global_orient",
                    "lock_body_joint_ids",
                )
                if key in options
            }
            # HML local rotations are tied to the HumanML3D rest skeleton.
            # Position IK preserves the visible source joints across body models.
            hml_options.setdefault("rotation_init", "position_ik")
            fit = retarget_hml263_clip(
                data,
                model_dir=_ik_model_dir(model_path),
                source_fps=source_fps,
                target_fps=output_fps,
                gender=gender,
                **hml_options,
            )
            animation = SMPLAnimation.from_smpl(
                fit["global_orient"],
                fit["body_pose"],
                fit["transl"],
                betas=betas,
                fps=output_fps,
            )
            fit_errors = np.asarray(fit["fit_mpjpe_mm"], dtype=np.float32)
            bridge = "HumanML3D joints plus SMPL position IK"
            lossy = True
            extra = {
                "fit_mpjpe_mm_mean": float(fit_errors.mean()),
                "fit_mpjpe_mm_p95": float(np.percentile(fit_errors, 95)),
                "fit_mpjpe_mm_max": float(fit_errors.max()),
                "rotation_init": str(np.asarray(fit["rotation_init"]).item()),
                "ik_body_shape": "zero-beta SMPL",
                "ik_body_gender": gender,
                "output_betas_applied": betas is not None,
            }
        elif source == "interhuman262":
            from motius.motion.representation.interhuman262 import interhuman262_to_joints

            value = np.asarray(data)
            if value.ndim == 3 and value.shape[1] == 2:
                if person_index not in {0, 1}:
                    raise ValueError(
                        "Paired InterHuman-262 export requires person_index=0 or 1."
                    )
                value = value[:, person_index]
            joints = interhuman262_to_joints(value)
            bridge = "InterHuman stored joints plus position IK"
        elif source == "ardy_330":
            from motius.motion.representation.convert import convert_motion

            joints = convert_motion(
                data,
                source,
                "smpl22_joints",
                motion_rep=motion_rep,
                is_normalized=is_normalized,
                return_numpy=True,
            )
            bridge = "ARDY-27 joint bridge plus position IK"
        elif source == "ardy_g1_414":
            from motius.motion.representation.ardy import decode_ardy_features

            if motion_rep is None:
                raise ValueError("ARDY-G1 export requires the checkpoint motion_rep.")
            decoded = decode_ardy_features(
                data,
                motion_rep=motion_rep,
                is_normalized=is_normalized,
                return_numpy=True,
            )
            joints = g1_joints_to_smpl22_joints(
                decoded["posed_joints"], motion_rep.skeleton.bone_order_names
            )
            bridge = "ARDY G1 named-joint bridge plus position IK"
        elif source.startswith("motionbricks_g1_"):
            joints_g1, names = _motion_rep_joints(
                data, motion_rep, source, is_normalized=is_normalized
            )
            joints = g1_joints_to_smpl22_joints(joints_g1, names)
            bridge = "MotionBricks G1 named-joint bridge plus position IK"
        elif source in {"g1_38", "g1_qpos"}:
            if source == "g1_38":
                import torch

                from motius.motion.representation.g1 import decode_g1_to_qpos

                qpos = decode_g1_to_qpos(
                    torch.as_tensor(data, dtype=torch.float32),
                    root_velocity=bool(options.get("root_velocity", True)),
                ).cpu().numpy()
            else:
                qpos = data
            joints_g1, names = _g1_qpos_to_named_joints(qpos, robot_xml)
            joints = g1_joints_to_smpl22_joints(
                joints_g1, names, coordinate_system="z_up"
            )
            bridge = "G1 MuJoCo FK and named-joint bridge plus position IK"
        elif source == "smpl22_joints":
            joints = data
            bridge = "SMPL-22 joints plus position IK"
        else:  # pragma: no cover - guarded by source normalization
            raise ValueError(f"Unsupported FBX source representation: {source!r}.")

        if source != "hml263":
            animation, fit_errors, extra = _fit_joint_positions(
                joints,
                model_path=model_path,
                source_fps=source_fps,
                output_fps=output_fps,
                betas=betas,
                gender=gender,
                bridge_kwargs=options,
            )
            lossy = True

    animation = _resample_animation(animation, output_fps)
    metadata = {
        "source_representation": source,
        "bridge": bridge,
        "lossy": lossy,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "person_index": person_index,
        "gender": gender,
        **extra,
    }
    return MotionBridgeResult(
        animation=animation,
        source_representation=source,
        bridge=bridge,
        lossy=lossy,
        source_fps=source_fps,
        output_fps=output_fps,
        person_index=person_index,
        fit_mpjpe_mm=fit_errors,
        metadata=metadata,
    )


def export_motion_to_fbx(
    data,
    source_representation: str,
    character_fbx: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    model_type: str = "smpl",
    gender: str = "neutral",
    source_fps: float | None = None,
    output_fps: float | None = None,
    betas=None,
    person_index: int | None = None,
    motion_rep=None,
    is_normalized: bool = True,
    robot_xml: str | Path | None = None,
    bridge_kwargs: Mapping[str, object] | None = None,
    bone_map: Mapping[str, str] | None = None,
    target_armature: str | None = None,
    strict_bone_map: bool = True,
    root_motion_scale: float | str = "auto",
    character_root: str | Path | None = None,
    backend: str = "auto",
    blender_executable: str | Path | None = None,
    fbxsdk_python: str | Path | None = None,
    fbxsdk_module_path: str | Path | None = None,
) -> FBXExportResult:
    """Export any supported Motius representation onto a character FBX."""

    bridge = motion_to_smpl_animation(
        data,
        source_representation,
        model_path=model_path,
        gender=gender,
        source_fps=source_fps,
        output_fps=output_fps,
        betas=betas,
        person_index=person_index,
        motion_rep=motion_rep,
        is_normalized=is_normalized,
        robot_xml=robot_xml,
        bridge_kwargs=bridge_kwargs,
    )
    return retarget_smpl_to_fbx(
        bridge.animation,
        resolve_character_fbx(character_fbx, root=character_root),
        output_path,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        bone_map=bone_map,
        target_armature=target_armature,
        strict_bone_map=strict_bone_map,
        root_motion_scale=root_motion_scale,
        backend=backend,
        blender_executable=blender_executable,
        fbxsdk_python=fbxsdk_python,
        fbxsdk_module_path=fbxsdk_module_path,
        source_metadata=bridge.metadata,
    )


export_motion_to_mixamo_fbx = export_motion_to_fbx


__all__ = [
    "MotionBridgeResult",
    "export_motion_to_fbx",
    "export_motion_to_mixamo_fbx",
    "g1_joints_to_smpl22_joints",
    "motion_to_smpl_animation",
]
