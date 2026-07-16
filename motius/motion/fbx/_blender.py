"""Blender subprocess backend for Motius FBX export.

This file is executed by Blender's bundled Python. Keep it independent from
the rest of Motius so Blender does not need the project's Torch environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _load_mapping_module():
    path = Path(__file__).with_name("_mapping.py")
    spec = importlib.util.spec_from_file_location("motius_fbx_mapping", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FBX bone mapping helpers from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAPPING = _load_mapping_module()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    values = list(__import__("sys").argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    else:
        values = []
    return parser.parse_args(values)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.actions,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for item in tuple(collection):
            if item.users == 0:
                collection.remove(item)


def _bone_tail(rest_joints: np.ndarray, parents: np.ndarray, joint: int) -> Vector:
    head = Vector(rest_joints[joint])
    children = np.flatnonzero(parents == joint)
    if len(children):
        vectors = [Vector(rest_joints[child]) - head for child in children]
        direction = max(vectors, key=lambda value: value.length)
    elif parents[joint] >= 0:
        direction = head - Vector(rest_joints[parents[joint]])
    else:
        direction = Vector((0.0, 0.0, 0.1))
    if direction.length < 1e-5:
        direction = Vector((0.0, 0.0, 0.1))
    length = max(0.02, min(0.12, direction.length * 0.4))
    return head + direction.normalized() * length


def _create_armature(payload, *, name: str = "SMPL_Armature"):
    rest_joints = np.asarray(payload["rest_joints"])
    parents = np.asarray(payload["parents"], dtype=np.int64)
    joint_names = [str(name) for name in payload["joint_names"]]

    armature_data = bpy.data.armatures.new(name)
    armature = bpy.data.objects.new(name, armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for joint, joint_name in enumerate(joint_names):
        bone = armature_data.edit_bones.new(joint_name)
        bone.head = Vector(rest_joints[joint])
        bone.tail = _bone_tail(rest_joints, parents, joint)
        bone.use_connect = False
        if parents[joint] >= 0:
            bone.parent = armature_data.edit_bones[joint_names[parents[joint]]]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def _create_skinned_mesh(payload, armature, *, name: str = "SMPL_Mesh"):
    vertices = np.asarray(payload["vertices"])
    faces = np.asarray(payload["faces"], dtype=np.int64)
    weights = np.asarray(payload["weights"])
    joint_names = [str(value) for value in payload["joint_names"]]

    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh_data.update()
    mesh = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(mesh)
    for polygon in mesh_data.polygons:
        polygon.use_smooth = True

    for joint, joint_name in enumerate(joint_names):
        group = mesh.vertex_groups.new(name=joint_name)
        indices = np.flatnonzero(weights[:, joint] > 1e-8)
        for vertex in indices:
            group.add([int(vertex)], float(weights[vertex, joint]), "REPLACE")

    modifier = mesh.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = False
    mesh.parent = armature
    return mesh


def _matrix(rotation, translation) -> Matrix:
    value = Matrix(np.asarray(rotation, dtype=np.float64).tolist()).to_4x4()
    value.translation = Vector(np.asarray(translation, dtype=np.float64))
    return value


def _key_pose_bone(pose_bone, frame: int, previous_quaternion) -> object:
    pose_bone.rotation_mode = "QUATERNION"
    quaternion = pose_bone.rotation_quaternion.copy()
    if previous_quaternion is not None and quaternion.dot(previous_quaternion) < 0:
        quaternion.negate()
        pose_bone.rotation_quaternion = quaternion
    pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert(
        data_path="rotation_quaternion", frame=frame, group=pose_bone.name
    )
    pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)
    return quaternion.copy()


def _linearize_action(action) -> None:
    if action is None:
        return
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def _animate_smpl_armature(armature, payload, fps: float) -> None:
    joints = np.asarray(payload["joints"])
    global_rotations = np.asarray(payload["global_rotations"])
    joint_names = [str(value) for value in payload["joint_names"]]
    frames = len(joints)
    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = scene.render.fps / float(fps)
    scene.frame_start = 1
    scene.frame_end = frames
    armature.animation_data_clear()
    previous = {name: None for name in joint_names}

    for index in range(frames):
        frame = index + 1
        scene.frame_set(frame)
        for joint, joint_name in enumerate(joint_names):
            rest_basis = armature.data.bones[joint_name].matrix_local.to_3x3()
            rotation = Matrix(global_rotations[index, joint].tolist()) @ rest_basis
            pose_bone = armature.pose.bones[joint_name]
            pose_bone.matrix = _matrix(rotation, joints[index, joint])
            previous[joint_name] = _key_pose_bone(
                pose_bone, frame, previous[joint_name]
            )
    if armature.animation_data and armature.animation_data.action:
        armature.animation_data.action.name = "SMPL_Animation"
        _linearize_action(armature.animation_data.action)
    scene.frame_set(1)


def _import_target(path: Path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    imported = [item for item in bpy.data.objects if item not in before]
    if not imported:
        raise ValueError(f"FBX import created no objects: {path}.")
    return imported


def _select_target_armature(imported, requested: str | None):
    armatures = [item for item in imported if item.type == "ARMATURE"]
    if requested:
        matches = [item for item in armatures if item.name == requested]
        if len(matches) != 1:
            available = [item.name for item in armatures]
            raise ValueError(
                f"Target armature {requested!r} was not found; available: {available}."
            )
        armature = matches[0]
    elif len(armatures) == 1:
        armature = armatures[0]
    else:
        raise ValueError(
            "The character FBX must contain exactly one armature unless "
            f"target_armature is supplied; found {[item.name for item in armatures]}."
        )

    skinned_meshes = []
    for item in imported:
        if item.type != "MESH":
            continue
        if any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in item.modifiers
        ):
            skinned_meshes.append(item)
    if not skinned_meshes:
        raise ValueError(
            f"Armature {armature.name!r} has no skinned mesh in the character FBX. "
            "Motius retargets animation to an already rigged/skinned FBX; it does "
            "not automatically rig a static mesh."
        )
    return armature, skinned_meshes


def _bone_depth(bone) -> int:
    depth = 0
    current = bone.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth


def _skeleton_height(points: np.ndarray) -> float:
    extent = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    if extent <= 1e-6:
        raise ValueError("Cannot infer root-motion scale from a zero-height skeleton.")
    return extent


def _retarget_animation(target, payload, bone_map, fps: float, root_scale) -> float:
    joint_names = [str(value) for value in payload["joint_names"]]
    source_index = {name: index for index, name in enumerate(joint_names)}
    joints = np.asarray(payload["joints"])
    global_rotations = np.asarray(payload["global_rotations"])
    frames = len(joints)

    if root_scale == "auto":
        target_points = np.asarray(
            [target.data.bones[name].head_local[:] for name in bone_map.values()],
            dtype=np.float64,
        )
        root_scale = _skeleton_height(target_points) / _skeleton_height(
            np.asarray(payload["rest_joints"], dtype=np.float64)[: len(joint_names)]
        )
    root_scale = float(root_scale)

    target.animation_data_clear()
    for pose_bone in target.pose.bones:
        pose_bone.matrix_basis.identity()
    target.data.pose_position = "POSE"

    ordered = sorted(
        bone_map.items(),
        key=lambda item: _bone_depth(target.data.bones[item[1]]),
    )
    root_source = "Pelvis"
    root_target = bone_map[root_source]
    target_root_rest = target.data.bones[root_target].head_local.copy()
    source_root_origin = Vector(joints[0, source_index[root_source]])
    previous = {target_name: None for target_name in bone_map.values()}

    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = scene.render.fps / float(fps)
    scene.frame_start = 1
    scene.frame_end = frames
    for index in range(frames):
        frame = index + 1
        scene.frame_set(frame)
        for source_name, target_name in ordered:
            source_joint = source_index[source_name]
            data_bone = target.data.bones[target_name]
            pose_bone = target.pose.bones[target_name]
            rest_basis = data_bone.matrix_local.to_3x3()
            rotation = Matrix(global_rotations[index, source_joint].tolist()) @ rest_basis
            location = pose_bone.head.copy()
            if source_name == root_source:
                displacement = (
                    Vector(joints[index, source_joint]) - source_root_origin
                ) * root_scale
                location = target_root_rest + displacement
            pose_bone.matrix = _matrix(rotation, location)
            previous[target_name] = _key_pose_bone(
                pose_bone, frame, previous[target_name]
            )

    if target.animation_data and target.animation_data.action:
        target.animation_data.action.name = "Motius_Retargeted_SMPL_Animation"
        _linearize_action(target.animation_data.action)
    scene.frame_set(1)
    return root_scale


def _select_objects(objects) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    eligible = [item for item in objects if item.type in {"ARMATURE", "MESH", "EMPTY"}]
    for item in eligible:
        item.select_set(True)
    armatures = [item for item in eligible if item.type == "ARMATURE"]
    if armatures:
        bpy.context.view_layer.objects.active = armatures[0]


def _export_fbx(path: Path, objects) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _select_objects(objects)
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        axis_forward="-Z",
        axis_up="Y",
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Blender reported success but wrote no FBX to {path}.")


def _main() -> None:
    args = _parse_args()
    job = json.loads(args.job.read_text())
    payload = np.load(job["payload_path"], allow_pickle=False)
    output = Path(job["output_path"])
    _clear_scene()

    bone_map = {}
    root_scale = None
    mesh_names = []
    if job["mode"] == "smpl_export":
        armature = _create_armature(payload)
        mesh = _create_skinned_mesh(payload, armature)
        _animate_smpl_armature(armature, payload, float(job["fps"]))
        exported_objects = [armature, mesh]
        mesh_names = [mesh.name]
    elif job["mode"] == "character_retarget":
        imported = _import_target(Path(job["character_fbx"]))
        armature, meshes = _select_target_armature(
            imported, job.get("target_armature")
        )
        bone_map = MAPPING.resolve_bone_map(
            armature.data.bones.keys(),
            job.get("bone_map"),
            strict=bool(job.get("strict_bone_map", True)),
        )
        if "Pelvis" not in bone_map:
            raise ValueError("The target bone map must include Pelvis for root motion.")
        root_scale = _retarget_animation(
            armature,
            payload,
            bone_map,
            float(job["fps"]),
            job.get("root_motion_scale", "auto"),
        )
        exported_objects = imported
        mesh_names = [mesh.name for mesh in meshes]
    else:
        raise ValueError(f"Unknown FBX job mode: {job['mode']!r}.")

    _export_fbx(output, exported_objects)
    manifest = {
        "schema_version": 1,
        "mode": job["mode"],
        "output_path": str(output),
        "source_model_path": job["source_model_path"],
        "model_type": job["model_type"],
        "gender": job["gender"],
        "frames": int(job["frames"]),
        "fps": float(job["fps"]),
        "armature_name": armature.name,
        "mesh_names": mesh_names,
        "bone_map": bone_map,
        "root_motion_scale": root_scale,
        "coordinates": {
            "input": "SMPL Y-up, +Z forward",
            "blender_scene": "Z-up, -Y forward",
            "fbx_export": "Y-up, -Z forward",
        },
        "skin": (
            "SMPL shaped rest mesh and official linear-blend skin weights"
            if job["mode"] == "smpl_export"
            else "target character skin weights preserved"
        ),
        "pose_correctives": (
            "SMPL pose-dependent corrective blend shapes are not embedded; "
            "the exported character uses standard armature linear-blend skinning."
        ),
    }
    Path(job["manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    _main()
