"""Blender subprocess for importing, fitting, skinning, and exporting a character.

Keep this file independent from the main Motius environment.  Blender loads
the small NumPy-only template module from the absolute path in the job file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    values = list(sys.argv)
    values = values[values.index("--") + 1 :] if "--" in values else []
    return parser.parse_args(values)


def _load_template(path: Path):
    module_name = "motius_rigging_template"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load rigging template helpers from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _import_character(path: Path):
    before = set(bpy.data.objects)
    suffix = path.suffix.casefold()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(
                filepath=str(path),
                forward_axis="NEGATIVE_Y",
                up_axis="Z",
            )
        else:
            bpy.ops.import_scene.obj(
                filepath=str(path),
                axis_forward="-Y",
                axis_up="Z",
            )
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(path))
    else:
        raise ValueError(f"Unsupported character input: {path}.")
    imported = [item for item in bpy.data.objects if item not in before]
    if not imported:
        raise ValueError(f"Character import created no objects: {path}.")
    armatures = [item for item in imported if item.type == "ARMATURE"]
    custom_shapes = {
        bone.custom_shape
        for armature in armatures
        for bone in armature.pose.bones
        if bone.custom_shape is not None
    }
    # Blender's GLTF importer creates editor-only meshes (usually an
    # Icosphere) as custom shapes for imported bones. They are not character
    # geometry and must not be rebound or exported as an extra body part.
    for helper in custom_shapes:
        if helper in imported:
            imported.remove(helper)
            bpy.data.objects.remove(helper, do_unlink=True)
    meshes = [
        item for item in imported if item.type == "MESH" and len(item.data.vertices)
    ]
    if not meshes:
        raise ValueError(f"Character import contains no non-empty mesh: {path}.")
    return imported, meshes


def _apply_explicit_up_axis(imported, up_axis: str) -> None:
    if up_axis == "auto" or up_axis == "Z":
        return
    vectors = {
        "X": Vector((1.0, 0.0, 0.0)),
        "Y": Vector((0.0, 1.0, 0.0)),
        "-X": Vector((-1.0, 0.0, 0.0)),
        "-Y": Vector((0.0, -1.0, 0.0)),
        "-Z": Vector((0.0, 0.0, -1.0)),
    }
    if up_axis not in vectors:
        raise ValueError(f"Unsupported explicit up axis: {up_axis!r}.")
    transform = (
        vectors[up_axis]
        .rotation_difference(Vector((0.0, 0.0, 1.0)))
        .to_matrix()
        .to_4x4()
    )
    imported_set = set(imported)
    roots = [item for item in imported if item.parent not in imported_set]
    for item in roots:
        item.matrix_world = transform @ item.matrix_world
    bpy.context.view_layer.update()


def _remove_existing_rig(imported, meshes, *, replace: bool) -> list[str]:
    armatures = [item for item in imported if item.type == "ARMATURE"]
    armature_modifiers = [
        (mesh, modifier)
        for mesh in meshes
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE"
    ]
    if (armatures or armature_modifiers) and not replace:
        names = [item.name for item in armatures]
        raise ValueError(
            "The character already contains an armature or armature modifier "
            f"({names}). Pass replace_existing_rig=True only when re-rigging is intentional."
        )
    removed = [item.name for item in armatures]
    if not replace:
        return removed

    for mesh, modifier in armature_modifiers:
        mesh.modifiers.remove(modifier)
    for mesh in meshes:
        for group in tuple(mesh.vertex_groups):
            mesh.vertex_groups.remove(group)
        if mesh.parent in armatures:
            world = mesh.matrix_world.copy()
            mesh.parent = None
            mesh.matrix_world = world
    for armature in armatures:
        bpy.data.objects.remove(armature, do_unlink=True)
    imported[:] = [item for item in imported if item not in armatures]
    bpy.context.view_layer.update()
    return removed


def _world_vertices(mesh) -> np.ndarray:
    matrix = mesh.matrix_world
    return np.asarray(
        [matrix @ vertex.co for vertex in mesh.data.vertices], dtype=np.float64
    )


def _bone_tail(joints: np.ndarray, parents: np.ndarray, joint: int) -> Vector:
    head = Vector(joints[joint])
    children = np.flatnonzero(parents == joint)
    terminal = not len(children)
    if len(children):
        # Keep the axial chain vertical when a spine joint also owns hips or
        # clavicles. Bone roll/rest bases feed directly into FBX retargeting.
        axial_child = {0: 3, 3: 6, 6: 9, 9: 12, 12: 15}.get(joint)
        if axial_child in children:
            direction = Vector(joints[axial_child]) - head
        else:
            vectors = [Vector(joints[child]) - head for child in children]
            direction = max(vectors, key=lambda value: value.length)
    elif parents[joint] >= 0:
        direction = head - Vector(joints[parents[joint]])
    else:
        direction = Vector((0.0, 0.0, 0.1))
    if direction.length < 1e-7:
        direction = Vector((0.0, 0.0, 0.1))
    # Non-terminal bones span the body segment they control. Terminal joints
    # have no SMPL child landmark, so use an anatomical proxy instead of
    # extending a wrist by a full forearm or a head by a full neck-to-head
    # distance.
    terminal_scale = {10: 0.70, 11: 0.70, 15: 0.35, 20: 0.34, 21: 0.34}
    scale = terminal_scale.get(joint, 0.35) if terminal else 1.0
    length = max(0.005, direction.length * scale)
    return head + direction.normalized() * length


def _create_armature(skeleton, *, name: str = "Motius_SMPL22_Rig"):
    joints = np.asarray(skeleton.joints, dtype=np.float64)
    parents = np.asarray(skeleton.parents, dtype=np.int64)
    names = list(skeleton.names)
    data = bpy.data.armatures.new(name)
    armature = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for joint, joint_name in enumerate(names):
        bone = data.edit_bones.new(joint_name)
        bone.head = Vector(joints[joint])
        bone.tail = _bone_tail(joints, parents, joint)
        bone.use_connect = False
        if parents[joint] >= 0:
            bone.parent = data.edit_bones[names[parents[joint]]]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    armature["motius_skeleton_profile"] = "smpl22"
    armature["motius_auto_rig_method"] = "template"
    return armature


def _bind_mesh_capsules(mesh, armature, weights: np.ndarray, joint_names) -> None:
    for group in tuple(mesh.vertex_groups):
        mesh.vertex_groups.remove(group)
    groups = [mesh.vertex_groups.new(name=name) for name in joint_names]
    for vertex, row in enumerate(weights):
        for joint in np.flatnonzero(row > 1e-8):
            groups[int(joint)].add([int(vertex)], float(row[joint]), "REPLACE")
    modifier = mesh.modifiers.new(name="Motius Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = True


def _bind_meshes_automatic(meshes, armature, *, top_k: int) -> None:
    """Bind with Blender's topology-aware bone-heat solver."""

    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        for group in tuple(mesh.vertex_groups):
            mesh.vertex_groups.remove(group)
        mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)

    for mesh in meshes:
        bpy.ops.object.select_all(action="DESELECT")
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.vertex_group_limit_total(
            group_select_mode="ALL", limit=int(top_k)
        )
        bpy.ops.object.vertex_group_normalize_all(
            group_select_mode="ALL", lock_active=False
        )
        modifiers = [
            modifier for modifier in mesh.modifiers if modifier.type == "ARMATURE"
        ]
        if len(modifiers) != 1 or modifiers[0].object != armature:
            raise RuntimeError(
                f"Automatic weights did not create one Armature modifier for {mesh.name}."
            )
        modifiers[0].use_deform_preserve_volume = True


def _weight_diagnostics(meshes, joint_names, *, top_k: int) -> dict[str, object]:
    joint_names = set(joint_names)
    vertices = 0
    unbound = 0
    maximum_influences = 0
    dominant = []
    active = set()
    maximum_sum_error = 0.0
    for mesh in meshes:
        names = {group.index: group.name for group in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            values = [
                item
                for item in vertex.groups
                if item.weight > 1e-8 and names.get(item.group) in joint_names
            ]
            total = sum(item.weight for item in values)
            vertices += 1
            if total <= 1e-8:
                unbound += 1
                continue
            maximum_sum_error = max(maximum_sum_error, abs(total - 1.0))
            maximum_influences = max(maximum_influences, len(values))
            dominant.append(max(item.weight for item in values))
            active.update(names[item.group] for item in values)
    return {
        "vertices": vertices,
        "top_k": int(top_k),
        "active_joints": len(active),
        "weight_sum_error_max": float(maximum_sum_error),
        "dominant_weight_mean": float(np.mean(dominant)) if dominant else 0.0,
        "unbound_vertices": unbound,
        "max_influences": maximum_influences,
    }


def _select_export_objects(imported, armature) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    eligible = [item for item in imported if item.type in {"MESH", "EMPTY"}]
    eligible.append(armature)
    for item in eligible:
        item.hide_set(False)
        item.hide_viewport = False
        item.hide_render = False
        item.select_set(True)
    bpy.context.view_layer.objects.active = armature


def _export_character(path: Path, imported, armature) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _select_export_objects(imported, armature)
    suffix = path.suffix.casefold()
    if suffix == ".fbx":
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            check_existing=False,
            use_selection=True,
            object_types={"ARMATURE", "MESH", "EMPTY"},
            use_mesh_modifiers=True,
            add_leaf_bones=False,
            primary_bone_axis="Y",
            secondary_bone_axis="X",
            bake_anim=False,
            path_mode="COPY",
            embed_textures=True,
            axis_forward="-Z",
            axis_up="Y",
        )
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.export_scene.gltf(
            filepath=str(path),
            check_existing=False,
            use_selection=True,
            export_format="GLB" if suffix == ".glb" else "GLTF_SEPARATE",
            export_skins=True,
            export_animations=False,
            export_yup=True,
        )
    else:
        raise ValueError(f"Unsupported rigged-character output: {path}.")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Blender reported success but wrote no asset to {path}.")


def _jsonable(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _main() -> None:
    args = _parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    template = _load_template(Path(job["template_module"]))
    _clear_scene()
    source = Path(job["character_path"])
    output = Path(job["output_path"])
    imported, meshes = _import_character(source)
    _apply_explicit_up_axis(imported, job.get("up_axis", "auto"))
    removed_rigs = _remove_existing_rig(
        imported,
        meshes,
        replace=bool(job.get("replace_existing_rig", False)),
    )

    per_mesh_vertices = [_world_vertices(mesh) for mesh in meshes]
    all_vertices = np.concatenate(per_mesh_vertices, axis=0)
    skeleton = template.fit_humanoid_skeleton(all_vertices)
    config = template.TemplateRiggingConfig(**job.get("config", {}))
    armature = _create_armature(skeleton)
    weight_method = job.get("weight_method", "capsules")
    if weight_method == "automatic":
        _bind_meshes_automatic(meshes, armature, top_k=config.top_k)
        skin_diagnostics = _weight_diagnostics(
            meshes, skeleton.names, top_k=config.top_k
        )
        if skin_diagnostics["unbound_vertices"]:
            raise RuntimeError(
                "Blender automatic weights left "
                f"{skin_diagnostics['unbound_vertices']} vertices unbound. "
                "Use a connected/watertight mesh or weight_method='capsules'."
            )
    elif weight_method == "capsules":
        skin = template.compute_skin_weights(all_vertices, skeleton, config=config)
        offset = 0
        for mesh, vertices in zip(meshes, per_mesh_vertices):
            stop = offset + len(vertices)
            _bind_mesh_capsules(
                mesh, armature, skin.weights[offset:stop], skeleton.names
            )
            offset = stop
        skin_diagnostics = dict(skin.diagnostics)
        skin_diagnostics["max_influences"] = int(config.top_k)
    else:
        raise ValueError(
            f"weight_method must be automatic or capsules, got {weight_method!r}."
        )

    _export_character(output, imported, armature)
    warnings = list(skeleton.diagnostics.get("warnings", ()))
    if skin_diagnostics["active_joints"] < len(skeleton.names):
        warnings.append(
            "One or more bones have no weighted vertices; inspect thin or disconnected body parts."
        )
    manifest = {
        "schema_version": 1,
        "method": job["method"],
        "weight_method": weight_method,
        "input_path": str(source),
        "input_format": source.suffix.casefold(),
        "output_path": str(output),
        "output_format": output.suffix.casefold(),
        "armature_name": armature.name,
        "mesh_names": [mesh.name for mesh in meshes],
        "joint_names": list(skeleton.names),
        "joint_parents": np.asarray(skeleton.parents).tolist(),
        "rest_joints": np.asarray(skeleton.joints).tolist(),
        "mesh_count": len(meshes),
        "vertex_count": len(all_vertices),
        "face_count": int(sum(len(mesh.data.polygons) for mesh in meshes)),
        "removed_rigs": removed_rigs,
        "replace_existing_rig": bool(job.get("replace_existing_rig", False)),
        "fit_diagnostics": _jsonable(skeleton.diagnostics),
        "skin_diagnostics": _jsonable(skin_diagnostics),
        "warnings": warnings,
        "coordinates": {
            "input_up_axis": job.get("up_axis", "auto"),
            "working_scene": "Blender Z-up, -Y forward",
            "export": "FBX/GLTF conventional Y-up",
        },
        "compatibility": {
            "skeleton_profile": "SMPL22",
            "motius_retarget_target": output.suffix.casefold() == ".fbx",
        },
    }
    Path(job["manifest_path"]).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    _main()
