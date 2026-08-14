#!/usr/bin/env python3
"""Animate a Motius SMPL22 character from an existing joint trajectory.

This Blender-side demo bridge avoids a licensed SMPL mesh dependency: the
rest skeleton is the auto-rigged target itself and local rotations are solved
from the repository's persisted HumanML3D joint positions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

SMPL22_NAMES = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)
SMPL22_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

# A 22-joint SMPL/HumanML3D trajectory contains the head *joint centre*, but
# no face landmarks or head-top joint.  Consequently Neck->Head locates the
# head; it is not a measurement of where the face is looking.  Treating that
# offset as a full head orientation produces a systematic false nod (about
# 45 degrees in common HumanML3D clips), which is especially conspicuous on
# children and big-head characters.  These joints therefore inherit the
# nearest observable torso orientation.  A caller that needs true gaze/head
# rotation must supply SMPL pose rotations or face landmarks instead.
UNOBSERVABLE_ORIENTATION_JOINTS = frozenset({12, 15})  # Neck, Head


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--start-frame", type=int, default=0)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _load_motion(
    path: Path, frames: int, start_frame: int
) -> tuple[np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    representation = payload["representations"]["humanml3d"]
    joints = np.asarray(representation["positions"], dtype=np.float64)
    parents = np.asarray(representation["parents"], dtype=np.int64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected (T,22,3) joints, got {joints.shape}.")
    if not np.array_equal(parents, SMPL22_PARENTS):
        raise ValueError("Motion parents do not match the Motius SMPL22 contract.")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative.")
    if frames < 2 or len(joints) < start_frame + frames:
        raise ValueError(
            f"Requested frames [{start_frame}, {start_frame + frames}) from a "
            f"{len(joints)}-frame clip."
        )
    # The demo JSON is Y-up SMPL/HumanML3D; Blender is Z-up, -Y forward.
    transform = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    return joints[start_frame : start_frame + frames] @ transform.T, payload


def _import_character(path: Path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path.resolve()), use_anim=False)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise ValueError(
            f"Expected one armature and at least one mesh, got {len(armatures)} and {len(meshes)}."
        )
    armature = armatures[0]
    missing = sorted(set(SMPL22_NAMES).difference(armature.data.bones.keys()))
    if missing:
        raise ValueError(f"Target is missing canonical bones: {missing}.")
    return imported, armature, meshes


def _align_vectors(source_vectors, target_vectors) -> Matrix:
    source = [Vector(value).normalized() for value in source_vectors]
    target = [Vector(value).normalized() for value in target_vectors]
    if len(source) == 1:
        return source[0].rotation_difference(target[0]).to_matrix()
    source_matrix = np.stack([np.asarray(value) for value in source])
    target_matrix = np.stack([np.asarray(value) for value in target])
    covariance = target_matrix.T @ source_matrix
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return Matrix(rotation.tolist())


def _global_rotations(joints: np.ndarray, rest: np.ndarray) -> np.ndarray:
    children = [[] for _ in SMPL22_NAMES]
    for joint, parent in enumerate(SMPL22_PARENTS):
        if parent >= 0:
            children[int(parent)].append(joint)
    result = np.tile(np.eye(3), (len(joints), len(SMPL22_NAMES), 1, 1))
    previous = [Matrix.Identity(3) for _ in SMPL22_NAMES]
    for frame, posed in enumerate(joints):
        for joint in range(len(SMPL22_NAMES)):
            if joint in UNOBSERVABLE_ORIENTATION_JOINTS:
                parent = int(SMPL22_PARENTS[joint])
                # Neck inherits Spine3. Head then inherits Neck, preserving
                # the character's authored upright head/neck rest relation.
                rotation = (
                    Matrix(result[frame, parent].tolist())
                    if parent >= 0
                    else previous[joint]
                )
                previous[joint] = rotation
                result[frame, joint] = np.asarray(rotation)
                continue
            child_ids = children[joint]
            if not child_ids:
                # SMPL22 has no finger, toe, or head-tip joints from which to
                # recover terminal twist.  The least surprising solution is
                # to inherit the parent's global orientation.  Keeping the
                # identity rotation here makes hands and feet remain fixed in
                # world space while their forearms/shins move beneath them.
                parent = int(SMPL22_PARENTS[joint])
                rotation = (
                    Matrix(result[frame, parent].tolist())
                    if parent >= 0
                    else previous[joint]
                )
                previous[joint] = rotation
                result[frame, joint] = np.asarray(rotation)
                continue
            source = [rest[child] - rest[joint] for child in child_ids]
            target = [posed[child] - posed[joint] for child in child_ids]
            valid = [
                index
                for index, (a, b) in enumerate(zip(source, target))
                if np.linalg.norm(a) > 1e-8 and np.linalg.norm(b) > 1e-8
            ]
            if not valid:
                rotation = previous[joint]
            else:
                rotation = _align_vectors(
                    [source[index] for index in valid],
                    [target[index] for index in valid],
                )
            previous[joint] = rotation
            result[frame, joint] = np.asarray(rotation)
    return result


def _matrix(rotation, translation) -> Matrix:
    value = Matrix(np.asarray(rotation, dtype=np.float64).tolist()).to_4x4()
    value.translation = Vector(translation)
    return value


def _local_basis_rotations(rotations: np.ndarray, armature) -> list[list[Matrix]]:
    """Convert desired armature-space rotations to stable pose-bone deltas.

    A Blender pose bone stores a *local* delta after its rest transform and its
    parent's posed transform.  Writing armature-space matrices bone-by-bone
    makes the first evaluated frame depend on Blender's hierarchy update order.
    Solving this relation explicitly produces the same result on every frame.
    """
    rest = [
        armature.data.bones[name].matrix_local.to_3x3()
        for name in SMPL22_NAMES
    ]
    # FBX/Mixamo rigs can use connected hierarchy helpers whose authored
    # parent is not the canonical SMPL22 parent (for example Head is parented
    # to Spine2 when a zero-length Neck helper exists).  Pose deltas must be
    # solved against the actual Blender hierarchy even though the desired
    # rotations come from the canonical SMPL22 kinematic tree.
    index_by_name = {name: index for index, name in enumerate(SMPL22_NAMES)}
    actual_parents = []
    for name in SMPL22_NAMES:
        parent = armature.data.bones[name].parent
        actual_parents.append(index_by_name.get(parent.name, -1) if parent else -1)
    result: list[list[Matrix]] = []
    for frame in range(len(rotations)):
        target = [
            Matrix(rotations[frame, joint].tolist()) @ rest[joint]
            for joint in range(len(SMPL22_NAMES))
        ]
        local = []
        for joint, parent in enumerate(actual_parents):
            if parent < 0:
                without_delta = rest[joint]
            else:
                parent = int(parent)
                rest_relative = rest[parent].inverted() @ rest[joint]
                without_delta = target[parent] @ rest_relative
            local.append(without_delta.inverted() @ target[joint])
        result.append(local)
    return result


def _key_pose(pose_bone, frame: int, previous: Quaternion | None) -> Quaternion:
    current = pose_bone.rotation_quaternion.copy()
    if previous is not None and current.dot(previous) < 0:
        current.negate()
        pose_bone.rotation_quaternion = current
    pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert("scale", frame=frame, group=pose_bone.name)
    return current.copy()


def _mesh_positions(meshes, frame: int):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    values = []
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        values.extend(evaluated.matrix_world @ vertex.co for vertex in data.vertices)
        evaluated.to_mesh_clear()
    return values


def _direction_error_report(armature, joints: np.ndarray) -> dict[str, object]:
    sample_indices = sorted(
        {0, len(joints) // 4, len(joints) // 2, 3 * len(joints) // 4, len(joints) - 1}
    )
    per_frame = {}
    combined = []
    frame_means = []
    frame_p95s = []
    for index in range(len(joints)):
        bpy.context.scene.frame_set(index)
        errors = []
        by_bone = {}
        children = [
            np.flatnonzero(SMPL22_PARENTS == joint)
            for joint in range(len(SMPL22_NAMES))
        ]
        axial = {0: 3, 3: 6, 6: 9, 9: 12, 12: 15}
        for joint, child_ids in enumerate(children):
            if not len(child_ids):
                continue
            if joint in UNOBSERVABLE_ORIENTATION_JOINTS:
                # Neck->Head is a joint-location vector, not a gaze vector.
                # Excluding it prevents a knowingly unobservable quantity
                # from being presented as a retargeting accuracy metric.
                continue
            preferred = axial.get(joint)
            child = (
                preferred
                if preferred is not None and preferred in child_ids
                else int(child_ids[0])
            )
            expected = joints[index, child] - joints[index, joint]
            bone = armature.pose.bones[SMPL22_NAMES[joint]]
            actual = np.asarray(bone.tail[:]) - np.asarray(bone.head[:])
            denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
            if denominator <= 1e-10:
                continue
            cosine = float(np.clip(np.dot(expected, actual) / denominator, -1.0, 1.0))
            error = float(np.degrees(np.arccos(cosine)))
            errors.append(error)
            by_bone[f"{SMPL22_NAMES[joint]}->{SMPL22_NAMES[child]}"] = error
        combined.extend(errors)
        frame_mean = float(np.mean(errors))
        frame_p95 = float(np.quantile(errors, 0.95))
        frame_means.append(frame_mean)
        frame_p95s.append(frame_p95)
        if index in sample_indices:
            per_frame[str(index)] = {
                "mean_degrees": frame_mean,
                "p95_degrees": frame_p95,
                "max_degrees": float(np.max(errors)),
                "by_bone_degrees": by_bone,
            }
    return {
        "evaluated_frames": len(joints),
        "sampled_frames": sample_indices,
        "mean_degrees": float(np.mean(combined)),
        "p95_degrees": float(np.quantile(combined, 0.95)),
        "max_degrees": float(np.max(combined)),
        "maximum_frame_mean_degrees": float(np.max(frame_means)),
        "maximum_frame_p95_degrees": float(np.max(frame_p95s)),
        "frames": per_frame,
        "unobservable_orientation_joints": [
            SMPL22_NAMES[index] for index in sorted(UNOBSERVABLE_ORIENTATION_JOINTS)
        ],
        "head_orientation_policy": (
            "SMPL22 positions do not encode gaze/head orientation; Neck and "
            "Head inherit the observable upper-torso orientation."
        ),
    }


def _main() -> None:
    args = _args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    imported, armature, meshes = _import_character(args.input)
    joints, payload = _load_motion(args.motion, args.frames, args.start_frame)
    rest = np.asarray(
        [armature.data.bones[name].head_local[:] for name in SMPL22_NAMES],
        dtype=np.float64,
    )
    rotations = _global_rotations(joints, rest)
    local_rotations = _local_basis_rotations(rotations, armature)
    target_height = float(np.ptp(rest[:, 2]))
    source_height = float(np.ptp(joints[0, :, 2]))
    root_scale = target_height / max(source_height, 1e-8)
    source_origin = joints[0, 0].copy()
    root_origin = rest[0].copy()

    armature.animation_data_clear()
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
        # Set the keyed representation before assigning any pose matrices.
        # Switching Euler -> quaternion after the first matrix assignment
        # records a stale identity quaternion on frame one; later frames look
        # correct only because the mode has already changed.
        bone.rotation_mode = "QUATERNION"
    previous = {name: None for name in SMPL22_NAMES}
    scene = bpy.context.scene
    scene.render.fps = max(1, round(args.fps))
    scene.render.fps_base = scene.render.fps / args.fps
    # Blender's FBX round-trip shifts this action by +1. Author at 0..T-1 so
    # the imported artifact used by validators/renderers lands at 1..T.
    scene.frame_start = 0
    scene.frame_end = len(joints) - 1
    for index, posed in enumerate(joints):
        frame = index
        scene.frame_set(frame)
        desired_root = _matrix(
            Matrix(rotations[index, 0].tolist())
            @ armature.data.bones[SMPL22_NAMES[0]].matrix_local.to_3x3(),
            root_origin + (posed[0] - source_origin) * root_scale,
        )
        root = armature.pose.bones[SMPL22_NAMES[0]]
        root.matrix_basis = (
            armature.data.bones[SMPL22_NAMES[0]].matrix_local.inverted()
            @ desired_root
        )
        for joint, name in enumerate(SMPL22_NAMES[1:], start=1):
            armature.pose.bones[name].matrix_basis = local_rotations[index][
                joint
            ].to_4x4()
        bpy.context.view_layer.update()
        for name in SMPL22_NAMES:
            previous[name] = _key_pose(
                armature.pose.bones[name], frame, previous[name]
            )
    if armature.animation_data and armature.animation_data.action:
        armature.animation_data.action.name = "Motius_HumanML3D_004822"
        for curve in armature.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    else:
        raise AssertionError("No animation action was produced.")

    direction_errors = _direction_error_report(armature, joints)

    start_positions = _mesh_positions(meshes, 0)
    sample_frames = sorted(
        {0, len(joints) // 3, 2 * len(joints) // 3, len(joints) - 1}
    )
    max_deformation = 0.0
    for sample in sample_frames[1:]:
        positions = _mesh_positions(meshes, sample)
        max_deformation = max(
            max((current - start).length for current, start in zip(positions, start_positions)),
            max_deformation,
        )
    if max_deformation <= 1e-4:
        raise AssertionError("The persisted action does not deform the character mesh.")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        if obj.type in {"MESH", "ARMATURE", "EMPTY"}:
            obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(args.output.resolve()),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_simplify_factor=0.0,
        axis_forward="-Z",
        axis_up="Y",
        path_mode="COPY",
        embed_textures=True,
    )
    report = {
        "schema_version": 1,
        "input_rig": str(args.input.resolve()),
        "motion": str(args.motion.resolve()),
        "motion_case_id": payload["case_id"],
        "motion_representation": "persisted HumanML3D SMPL22 joints",
        "orientation_observability": {
            "unobservable_joints": ["Neck", "Head"],
            "policy": (
                "preserve authored head/neck rest relation by inheriting the "
                "upper-torso orientation; use SMPL pose rotations or face "
                "landmarks when true head orientation is required"
            ),
        },
        "output": str(args.output.resolve()),
        "frames": len(joints),
        "source_start_frame": args.start_frame,
        "fps": args.fps,
        "bones": len(armature.data.bones),
        "meshes": [mesh.name for mesh in meshes],
        "root_motion_scale": root_scale,
        "max_sampled_vertex_deformation": max_deformation,
        "bone_direction_error": direction_errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
