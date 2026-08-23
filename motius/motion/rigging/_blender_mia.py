"""Normalize a Make-It-Animatable FBX to Motius's SMPL22 contract.

This module runs inside Blender and communicates through a JSON job file.  It
is not imported by the normal Python runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _short_bone_name(name: str) -> str:
    return name.split(":")[-1]


def _export_selected(output: Path) -> None:
    suffix = output.suffix.casefold()
    if suffix == ".fbx":
        bpy.ops.export_scene.fbx(
            filepath=str(output),
            check_existing=False,
            use_selection=True,
            object_types={"ARMATURE", "MESH"},
            use_mesh_modifiers=True,
            add_leaf_bones=False,
            bake_anim=False,
            axis_forward="-Z",
            axis_up="Y",
            path_mode="COPY",
            embed_textures=True,
        )
        return
    export_format = "GLB" if suffix == ".glb" else "GLTF_SEPARATE"
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        check_existing=False,
        use_selection=True,
        export_format=export_format,
        export_skins=True,
        export_animations=False,
    )


def _main() -> None:
    args = _arguments()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    source = Path(job["raw_fbx"]).resolve()
    output = Path(job["output_path"]).resolve()
    manifest = Path(job["manifest_path"]).resolve()
    mapping = {str(key): str(value) for key, value in job["bone_mapping"].items()}

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(source), use_anim=False)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise ValueError(
            "Make-It-Animatable output must contain one armature and mesh geometry."
        )
    armature = armatures[0]

    # FBX commonly imports with a 0.01 object scale and a Y-up-to-Z-up object
    # rotation. Bake that transform into the armature before retargeting.
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.view_layer.update()

    by_short = {_short_bone_name(bone.name): bone for bone in armature.data.bones}
    missing = sorted(set(mapping).difference(by_short))
    if missing:
        raise ValueError(f"Make-It-Animatable output is missing bones: {missing}")

    # Two-phase renaming avoids collisions such as Spine -> Spine1 while the
    # source Spine1 bone still exists. Vertex groups follow the same temporary
    # names so skin binding remains intact.
    temporary = {}
    for index, (source_name, target_name) in enumerate(mapping.items()):
        bone = by_short[source_name]
        old_full_name = bone.name
        temporary_name = f"__MOTIUS_RENAME_{index:02d}__"
        bone.name = temporary_name
        temporary[temporary_name] = (target_name, old_full_name)

    for mesh in meshes:
        by_group = {group.name: group for group in mesh.vertex_groups}
        for temporary_name, (_target_name, old_full_name) in temporary.items():
            group = by_group.get(old_full_name)
            if group is not None:
                group.name = temporary_name

    for temporary_name, (target_name, _old_full_name) in temporary.items():
        armature.data.bones[temporary_name].name = target_name
        for mesh in meshes:
            group = mesh.vertex_groups.get(temporary_name)
            if group is not None:
                group.name = target_name

    canonical_names = tuple(mapping.values())
    armature.name = "Motius_SMPL22_Rig"
    armature.data.name = "Motius_SMPL22_Rig"
    armature["motius_skeleton_profile"] = "smpl22"
    armature["motius_auto_rig_method"] = "make_it_animatable"

    bpy.ops.object.select_all(action="DESELECT")
    for obj in (armature, *meshes):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    output.parent.mkdir(parents=True, exist_ok=True)
    _export_selected(output)

    extra_bones = sorted(
        bone.name for bone in armature.data.bones if bone.name not in canonical_names
    )
    warnings = []
    if extra_bones:
        warnings.append(
            "The upstream rig contains additional non-SMPL22 bones; the canonical "
            "SMPL22 subset is available for Motius retargeting."
        )
    metadata = {
        "schema_version": 1,
        "method": "make_it_animatable",
        "armature_name": armature.name,
        "mesh_names": [mesh.name for mesh in meshes],
        "joint_names": list(canonical_names),
        "extra_bone_names": extra_bones,
        "warnings": warnings,
        "backend": job["backend"],
        "source_character": job["source_character"],
        "output_path": str(output),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    _main()
