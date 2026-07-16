#!/usr/bin/env python3
"""Build Motius-owned, redistributable Mixamo-compatible FBX characters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


BONES = (
    "Hips",
    "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg", "Spine1",
    "LeftFoot", "RightFoot", "Spine2", "LeftToeBase", "RightToeBase", "Neck",
    "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
)

PARENTS = {
    "Hips": None,
    "LeftUpLeg": "Hips", "RightUpLeg": "Hips", "Spine": "Hips",
    "LeftLeg": "LeftUpLeg", "RightLeg": "RightUpLeg", "Spine1": "Spine",
    "LeftFoot": "LeftLeg", "RightFoot": "RightLeg", "Spine2": "Spine1",
    "LeftToeBase": "LeftFoot", "RightToeBase": "RightFoot", "Neck": "Spine2",
    "LeftShoulder": "Spine2", "RightShoulder": "Spine2", "Head": "Neck",
    "LeftArm": "LeftShoulder", "RightArm": "RightShoulder",
    "LeftForeArm": "LeftArm", "RightForeArm": "RightArm",
    "LeftHand": "LeftForeArm", "RightHand": "RightForeArm",
}

BASE_JOINTS = {
    "Hips": (0.0, 0.0, 1.00),
    "LeftUpLeg": (0.105, 0.0, 0.94), "RightUpLeg": (-0.105, 0.0, 0.94),
    "Spine": (0.0, 0.0, 1.13),
    "LeftLeg": (0.105, 0.0, 0.55), "RightLeg": (-0.105, 0.0, 0.55),
    "Spine1": (0.0, 0.0, 1.29),
    "LeftFoot": (0.105, 0.0, 0.13), "RightFoot": (-0.105, 0.0, 0.13),
    "Spine2": (0.0, 0.0, 1.45),
    "LeftToeBase": (0.105, -0.18, 0.07), "RightToeBase": (-0.105, -0.18, 0.07),
    "Neck": (0.0, 0.0, 1.58),
    "LeftShoulder": (0.18, 0.0, 1.49), "RightShoulder": (-0.18, 0.0, 1.49),
    "Head": (0.0, 0.0, 1.76),
    "LeftArm": (0.34, 0.0, 1.49), "RightArm": (-0.34, 0.0, 1.49),
    "LeftForeArm": (0.61, 0.0, 1.49), "RightForeArm": (-0.61, 0.0, 1.49),
    "LeftHand": (0.84, 0.0, 1.49), "RightHand": (-0.84, 0.0, 1.49),
}

STYLES = {
    "atlas": {
        "height": 1.04, "width": 1.08, "depth": 1.0, "limb": 1.08,
        "primary": (0.02, 0.48, 0.63, 1.0), "secondary": (0.87, 0.95, 0.96, 1.0),
        "accent": (0.95, 0.38, 0.22, 1.0), "shape": "smooth",
    },
    "nova": {
        "height": 0.98, "width": 0.91, "depth": 0.90, "limb": 0.88,
        "primary": (0.86, 0.20, 0.25, 1.0), "secondary": (0.12, 0.14, 0.18, 1.0),
        "accent": (0.96, 0.69, 0.22, 1.0), "shape": "smooth",
    },
    "gear": {
        "height": 1.01, "width": 1.02, "depth": 1.06, "limb": 1.10,
        "primary": (0.19, 0.22, 0.25, 1.0), "secondary": (0.72, 0.75, 0.76, 1.0),
        "accent": (0.96, 0.68, 0.08, 1.0), "shape": "block",
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials):
        for item in tuple(collection):
            if item.users == 0:
                collection.remove(item)


def _material(name: str, color) -> object:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.62
    principled.inputs["Metallic"].default_value = 0.05
    return material


def _joints(style: dict) -> dict[str, Vector]:
    output = {}
    for name, value in BASE_JOINTS.items():
        x, y, z = value
        output[name] = Vector(
            (x * style["width"], y * style["depth"], z * style["height"])
        )
    return output


def _tail(name: str, joints: dict[str, Vector]) -> Vector:
    children = [child for child, parent in PARENTS.items() if parent == name]
    if children:
        direction = max(
            (joints[child] - joints[name] for child in children),
            key=lambda value: value.length,
        )
    elif name == "Head":
        direction = Vector((0.0, 0.0, 0.12))
    elif "Toe" in name:
        direction = Vector((0.0, -0.12, 0.0))
    else:
        direction = joints[name] - joints[PARENTS[name]]
    return joints[name] + direction.normalized() * max(0.055, direction.length * 0.82)


def _armature(slug: str, joints: dict[str, Vector]):
    data = bpy.data.armatures.new(f"{slug.title()}Rig")
    armature = bpy.data.objects.new(f"{slug.title()}Rig", data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for name in BONES:
        bone = data.edit_bones.new(f"mixamorig:{name}")
        bone.head = joints[name]
        bone.tail = _tail(name, joints)
        bone.use_connect = False
        parent = PARENTS[name]
        if parent:
            bone.parent = data.edit_bones[f"mixamorig:{parent}"]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def _finish_mesh(obj, *, name: str, bone: str, armature, material) -> None:
    obj.name = name
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if material:
        obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    group = obj.vertex_groups.new(name=f"mixamorig:{bone}")
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    obj.parent = armature


def _ellipsoid(name, center, scale, bone, armature, material, block=False):
    if block:
        bpy.ops.mesh.primitive_cube_add(location=center)
        obj = bpy.context.object
        obj.scale = scale
        bevel = obj.modifiers.new(name="SoftEdges", type="BEVEL")
        bevel.width = min(scale) * 0.22
        bevel.segments = 2
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=bevel.name)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=center)
        obj = bpy.context.object
        obj.scale = scale
    _finish_mesh(obj, name=name, bone=bone, armature=armature, material=material)


def _segment(name, start, end, radius, bone, armature, material, block=False):
    vector = end - start
    center = (start + end) * 0.5
    if block:
        bpy.ops.mesh.primitive_cube_add(location=center)
        obj = bpy.context.object
        obj.scale = (radius, radius * 0.86, vector.length * 0.5)
    else:
        bpy.ops.mesh.primitive_cone_add(
            vertices=20,
            radius1=radius * 0.88,
            radius2=radius,
            depth=vector.length,
            location=center,
        )
        obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(vector)
    _finish_mesh(obj, name=name, bone=bone, armature=armature, material=material)


def _build(slug: str, output: Path) -> None:
    _clear()
    style = STYLES[slug]
    joints = _joints(style)
    armature = _armature(slug, joints)
    primary = _material(f"{slug}_primary", style["primary"])
    secondary = _material(f"{slug}_secondary", style["secondary"])
    accent = _material(f"{slug}_accent", style["accent"])
    block = style["shape"] == "block"
    limb = style["limb"]

    _ellipsoid(f"{slug}_pelvis", joints["Hips"], (0.18, 0.12, 0.13), "Hips", armature, primary, block)
    _ellipsoid(f"{slug}_abdomen", (joints["Spine"] + joints["Spine1"]) / 2, (0.17, 0.105, 0.18), "Spine", armature, secondary, block)
    _ellipsoid(f"{slug}_chest", (joints["Spine1"] + joints["Spine2"]) / 2, (0.225, 0.12, 0.21), "Spine1", armature, primary, block)
    _segment(f"{slug}_neck", joints["Neck"] - Vector((0, 0, 0.07)), joints["Neck"] + Vector((0, 0, 0.06)), 0.055, "Neck", armature, secondary, block)
    _ellipsoid(f"{slug}_head", joints["Head"] + Vector((0, 0, 0.04)), (0.105, 0.10, 0.13), "Head", armature, secondary, block)
    _ellipsoid(f"{slug}_visor", joints["Head"] + Vector((0, -0.085, 0.055)), (0.07, 0.025, 0.035), "Head", armature, accent, block)

    for side in ("Left", "Right"):
        _segment(f"{slug}_{side}_thigh", joints[f"{side}UpLeg"], joints[f"{side}Leg"], 0.075 * limb, f"{side}UpLeg", armature, primary, block)
        _segment(f"{slug}_{side}_shin", joints[f"{side}Leg"], joints[f"{side}Foot"], 0.058 * limb, f"{side}Leg", armature, secondary, block)
        foot_center = (joints[f"{side}Foot"] + joints[f"{side}ToeBase"]) / 2
        _ellipsoid(f"{slug}_{side}_foot", foot_center, (0.065, 0.13, 0.045), f"{side}Foot", armature, accent, True)
        _segment(f"{slug}_{side}_shoulder", joints[f"{side}Shoulder"], joints[f"{side}Arm"], 0.06 * limb, f"{side}Shoulder", armature, primary, block)
        _segment(f"{slug}_{side}_upper_arm", joints[f"{side}Arm"], joints[f"{side}ForeArm"], 0.052 * limb, f"{side}Arm", armature, primary, block)
        _segment(f"{slug}_{side}_forearm", joints[f"{side}ForeArm"], joints[f"{side}Hand"], 0.043 * limb, f"{side}ForeArm", armature, secondary, block)
        _ellipsoid(f"{slug}_{side}_hand", joints[f"{side}Hand"], (0.07, 0.04, 0.05), f"{side}Hand", armature, accent, block)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        bake_anim=False,
        path_mode="AUTO",
        axis_forward="-Z",
        axis_up="Y",
    )


def main() -> None:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "generator": "tools/build_mixamo_characters.py", "characters": {}}
    for slug in STYLES:
        output = (args.output_dir / f"{slug}.fbx").resolve()
        _build(slug, output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest["characters"][slug] = {
            "file": output.name,
            "sha256": digest,
            "bone_namespace": "mixamorig:",
            "license": "CC0-1.0",
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
