#!/usr/bin/env python3
"""Verify that a demo GLB contains geometry but no rigging or animation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _main() -> None:
    args = _args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(args.asset.resolve()), import_pack_images=True)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    armature_modifiers = [
        f"{mesh.name}:{modifier.name}"
        for mesh in meshes
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE"
    ]
    vertex_groups = sum(len(mesh.vertex_groups) for mesh in meshes)
    actions = [action.name for action in bpy.data.actions]
    if not meshes or sum(len(mesh.data.vertices) for mesh in meshes) == 0:
        raise AssertionError("The supposed character input contains no mesh vertices.")
    if armatures or armature_modifiers or vertex_groups or actions:
        raise AssertionError(
            "The supposed unrigged input contains rigging/animation: "
            f"armatures={armatures}, modifiers={armature_modifiers}, "
            f"vertex_groups={vertex_groups}, actions={actions}."
        )
    report = {
        "schema_version": 1,
        "asset": str(args.asset.resolve()),
        "meshes": [mesh.name for mesh in meshes],
        "vertices": sum(len(mesh.data.vertices) for mesh in meshes),
        "faces": sum(len(mesh.data.polygons) for mesh in meshes),
        "armatures": 0,
        "armature_modifiers": 0,
        "vertex_groups": 0,
        "actions": 0,
        "verdict": "unrigged static mesh",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
