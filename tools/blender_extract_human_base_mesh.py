#!/usr/bin/env python3
"""Extract one static body from Blender's CC0 Human Base Meshes bundle.

Run this file with Blender, not regular Python.  The resulting GLB is the
public, genuinely unrigged input used by the auto-rigging demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--object-pattern", default="body_male_realistic")
    parser.add_argument(
        "--format",
        choices=("glb", "obj"),
        default="glb",
        help="Static unrigged interchange format.",
    )
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _mesh_candidates(pattern: str):
    wanted = pattern.casefold()
    return sorted(
        (
            obj
            for obj in bpy.data.objects
            if obj.type == "MESH" and wanted in obj.name.casefold()
        ),
        key=lambda obj: (-len(obj.data.vertices), obj.name.casefold()),
    )


def _evaluated_copy(source):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    copied = bpy.data.objects.new(
        "Blender_CC0_Human_Base_Male",
        bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph),
    )
    bpy.context.collection.objects.link(copied)
    copied.matrix_world = source.matrix_world.copy()
    for slot in source.material_slots:
        if slot.material is not None:
            copied.data.materials.append(slot.material)
    return copied


def _main() -> None:
    args = _args()
    candidates = _mesh_candidates(args.object_pattern)
    if not candidates:
        available = sorted(
            obj.name for obj in bpy.data.objects if obj.type == "MESH"
        )
        raise ValueError(
            f"No mesh matches {args.object_pattern!r}; available meshes: {available}"
        )
    source = candidates[0]
    body = _evaluated_copy(source)

    # The public input must not inherit any rigging metadata from the .blend.
    for modifier in tuple(body.modifiers):
        body.modifiers.remove(modifier)
    for group in tuple(body.vertex_groups):
        body.vertex_groups.remove(group)
    body.parent = None
    bpy.context.view_layer.update()

    armatures = [obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if armatures:
        raise AssertionError(
            "The selected Blender asset unexpectedly contains armatures: "
            f"{armatures}"
        )
    if body.vertex_groups:
        raise AssertionError("The exported demo input unexpectedly has vertex groups.")
    if any(mod.type == "ARMATURE" for mod in body.modifiers):
        raise AssertionError("The exported demo input unexpectedly has an armature modifier.")

    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "glb":
        bpy.ops.export_scene.gltf(
            filepath=str(args.output.resolve()),
            check_existing=False,
            use_selection=True,
            export_format="GLB",
            export_skins=False,
            export_animations=False,
            export_yup=True,
        )
    else:
        bpy.ops.wm.obj_export(
            filepath=str(args.output.resolve()),
            check_existing=False,
            export_selected_objects=True,
            apply_modifiers=True,
            export_materials=True,
            export_uv=True,
            export_normals=True,
            export_vertex_groups=False,
            forward_axis="NEGATIVE_Y",
            up_axis="Z",
        )
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Failed to export a non-empty GLB to {args.output}.")

    report = {
        "schema_version": 1,
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "source_object": source.name,
        "output": str(args.output.resolve()),
        "mesh_objects": 1,
        "vertices": len(body.data.vertices),
        "faces": len(body.data.polygons),
        "armatures": 0,
        "armature_modifiers": 0,
        "vertex_groups": 0,
        "skins_exported": False,
        "animations_exported": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
