"""Validate an exported rigged asset inside Blender.

This is an opt-in smoke-test helper, not a pytest module.  Run it through
Blender's ``--python`` interface and pass ``--asset`` after ``--``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

SMPL22_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-animation", action="store_true")
    parser.add_argument("--require-deformation", action="store_true")
    values = list(sys.argv)
    values = values[values.index("--") + 1 :] if "--" in values else []
    return parser.parse_args(values)


def _main() -> None:
    args = _args()
    before = set(bpy.data.objects)
    suffix = args.asset.suffix.casefold()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(args.asset))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(
            filepath=str(args.asset), use_anim=args.require_animation
        )
    else:
        raise ValueError(f"Unsupported validation asset: {args.asset}.")

    imported = [item for item in bpy.data.objects if item not in before]
    armatures = [item for item in imported if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise AssertionError(
            f"Expected one armature, found {[item.name for item in armatures]}."
        )
    armature = armatures[0]
    custom_shapes = {
        bone.custom_shape
        for bone in armature.pose.bones
        if bone.custom_shape is not None
    }
    meshes = [
        item for item in imported if item.type == "MESH" and item not in custom_shapes
    ]
    missing_bones = sorted(set(SMPL22_NAMES).difference(armature.data.bones.keys()))
    if missing_bones:
        raise AssertionError(f"Missing canonical bones: {missing_bones}.")
    if not meshes:
        raise AssertionError("Rigged asset has no mesh objects.")

    unbound_meshes = []
    unbound_vertices = 0
    max_influences = 0
    for mesh in meshes:
        modifiers = [
            modifier
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE" and modifier.object == armature
        ]
        if not modifiers:
            unbound_meshes.append(mesh.name)
        for vertex in mesh.data.vertices:
            influences = [group for group in vertex.groups if group.weight > 1e-8]
            max_influences = max(max_influences, len(influences))
            if not influences:
                unbound_vertices += 1
    if unbound_meshes:
        raise AssertionError(f"Meshes without armature binding: {unbound_meshes}.")
    if unbound_vertices:
        raise AssertionError(f"Found {unbound_vertices} unbound vertices.")
    if max_influences > 4:
        raise AssertionError(
            f"Found {max_influences} influences on one vertex; expected <=4."
        )

    actions = [
        action
        for action in bpy.data.actions
        if action.frame_range[1] > action.frame_range[0]
    ]
    if args.require_animation and not actions:
        raise AssertionError("The imported rig has no non-empty animation action.")

    max_vertex_deformation = 0.0
    if actions:
        scene = bpy.context.scene

        def positions(frame: float):
            scene.frame_set(round(frame))
            depsgraph = bpy.context.evaluated_depsgraph_get()
            values = []
            for mesh in meshes:
                evaluated = mesh.evaluated_get(depsgraph)
                evaluated_mesh = evaluated.to_mesh()
                values.append(
                    [
                        evaluated.matrix_world @ vertex.co
                        for vertex in evaluated_mesh.vertices
                    ]
                )
                evaluated.to_mesh_clear()
            return values

        start_positions = positions(actions[0].frame_range[0])
        end_positions = positions(actions[0].frame_range[1])
        max_vertex_deformation = max(
            (end - start).length
            for start_mesh, end_mesh in zip(start_positions, end_positions)
            for start, end in zip(start_mesh, end_mesh)
        )
    if args.require_deformation and max_vertex_deformation <= 1e-4:
        raise AssertionError(
            "The animation action exists but does not deform or move any mesh vertex."
        )

    report = {
        "asset": str(args.asset),
        "armature": armature.name,
        "bones": len(armature.data.bones),
        "canonical_bones": len(SMPL22_NAMES),
        "meshes": [mesh.name for mesh in meshes],
        "vertices": sum(len(mesh.data.vertices) for mesh in meshes),
        "unbound_vertices": unbound_vertices,
        "max_influences": max_influences,
        "actions": [
            {
                "name": action.name,
                "frame_start": float(action.frame_range[0]),
                "frame_end": float(action.frame_range[1]),
            }
            for action in actions
        ],
        "max_vertex_deformation": float(max_vertex_deformation),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
