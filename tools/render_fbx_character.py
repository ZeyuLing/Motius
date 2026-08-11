#!/usr/bin/env python3
"""Render an animated character FBX preview with Blender."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _look_at(obj, point) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _material(name: str, color, roughness=0.65):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Roughness"].default_value = roughness
    return material


def _normalize(name: str) -> str:
    value = str(name).rsplit(":", 1)[-1].casefold()
    return "".join(character for character in value if character.isalnum())


def _pose_position(armature, names) -> Vector | None:
    wanted = {_normalize(name) for name in names}
    for bone in armature.pose.bones:
        if _normalize(bone.name) in wanted:
            return armature.matrix_world @ bone.head
    return None


def _mesh_bounds(meshes) -> tuple[Vector, Vector]:
    points = [mesh.matrix_world @ Vector(corner) for mesh in meshes for corner in mesh.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def _main() -> None:
    args = _args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(args.input.resolve()), use_anim=True)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise ValueError(
            f"Expected one armature and at least one mesh, got {len(armatures)} and {len(meshes)}."
        )
    armature = armatures[0]
    top_level = [obj for obj in imported if obj.parent not in imported]
    root = bpy.data.objects.new("PreviewRoot", None)
    bpy.context.collection.objects.link(root)
    for obj in top_level:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world

    scene = bpy.context.scene
    actions = [obj.animation_data.action for obj in imported if obj.animation_data and obj.animation_data.action]
    animation_end = max((int(math.ceil(action.frame_range[1])) for action in actions), default=1)
    start = max(1, args.start_frame)
    end = min(animation_end, start + args.frames - 1)
    if end < start:
        raise ValueError(f"No animation frames in requested range {start}..{end}.")

    foot_names = ("LeftFoot", "RightFoot", "L_Ankle", "R_Ankle", "Foot_L", "Foot_R")
    floor = float("inf")
    for frame in range(start, end + 1):
        scene.frame_set(frame)
        for name in foot_names:
            point = _pose_position(armature, (name,))
            if point is not None:
                floor = min(floor, point.z)
    scene.frame_set(start)
    pelvis = _pose_position(armature, ("Hips", "Pelvis"))
    minimum, maximum = _mesh_bounds(meshes)
    if not math.isfinite(floor):
        floor = minimum.z
    center = pelvis if pelvis is not None else (minimum + maximum) * 0.5
    root.location = Vector((-center.x, -center.y, -floor))
    scene.frame_set(start)
    minimum, maximum = _mesh_bounds(meshes)
    height = max(0.5, maximum.z - minimum.z)

    ground = _material("PreviewGround", (0.07, 0.08, 0.095, 1.0), roughness=0.82)
    bpy.ops.mesh.primitive_plane_add(size=max(12.0, height * 7.0), location=(0, 0, -0.01))
    bpy.context.object.data.materials.append(ground)

    bpy.ops.object.light_add(type="AREA", location=(-height * 1.4, -height * 1.8, height * 2.6))
    key = bpy.context.object
    key.data.energy = 850
    key.data.size = height * 2.0
    _look_at(key, (0, 0, height * 0.55))
    bpy.ops.object.light_add(type="AREA", location=(height * 1.8, 0, height * 1.7))
    fill = bpy.context.object
    fill.data.energy = 500
    fill.data.size = height * 1.5
    _look_at(fill, (0, 0, height * 0.65))

    bpy.ops.object.camera_add(location=(height * 0.8, -height * 3.0, height * 1.25))
    camera = bpy.context.object
    camera.data.lens = 58
    _look_at(camera, (0, 0, height * 0.55))
    scene.camera = camera

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 6
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(args.output.resolve())
    scene.render.fps = args.fps
    scene.frame_start = start
    scene.frame_end = end
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.018, 0.023, 0.030, 1.0)
    background.inputs["Strength"].default_value = 0.32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    _main()
