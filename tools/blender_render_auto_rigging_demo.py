#!/usr/bin/env python3
"""Render the real-mesh auto-rigging demo to MP4 and PNG frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--gif-resolution", type=int, default=420)
    parser.add_argument("--gif-step", type=int, default=3)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _look_at(obj, point) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _material(name: str, color, *, metallic=0.0, roughness=0.5):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def _mesh_bounds(meshes, frame: int) -> tuple[Vector, Vector]:
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
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
            f"Expected one armature and mesh geometry, got {len(armatures)} and {len(meshes)}."
        )
    actions = [
        obj.animation_data.action
        for obj in imported
        if obj.animation_data and obj.animation_data.action
    ]
    end = min(
        args.frames,
        max((math.floor(action.frame_range[1]) for action in actions), default=0),
    )
    if end < 2:
        raise ValueError("Animated FBX contains no usable frame range.")

    # Warm clay against a light neutral studio gives readable form shading on
    # the face, neck, fingers, and joint folds.  The previous emissive-looking
    # teal on a near-black set flattened these areas and made deformation
    # defects difficult to distinguish from lighting artifacts.
    character = _material(
        "Motius Warm Clay", (0.17, 0.034, 0.008, 1.0), roughness=0.64
    )
    accent = _material(
        "Motius Warm Accent",
        (0.24, 0.065, 0.015, 1.0),
        metallic=0.02,
        roughness=0.56,
    )
    for index, mesh in enumerate(meshes):
        mesh.data.materials.clear()
        mesh.data.materials.append(character if index == 0 else accent)
        for polygon in mesh.data.polygons:
            polygon.use_smooth = True

    sample_frames = sorted({1, end // 4, end // 2, 3 * end // 4, end})
    bounds = [_mesh_bounds(meshes, frame) for frame in sample_frames]
    minimum = Vector(
        tuple(min(value[0][axis] for value in bounds) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(value[1][axis] for value in bounds) for axis in range(3))
    )
    center = (minimum + maximum) * 0.5
    height = max(0.5, maximum.z - minimum.z)
    floor_z = minimum.z

    ground = _material("Ground", (0.80, 0.77, 0.71, 1.0), roughness=0.78)
    bpy.ops.mesh.primitive_plane_add(
        size=max(12.0, height * 7.0),
        location=(center.x, center.y, floor_z - 0.008),
    )
    bpy.context.object.data.materials.append(ground)

    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x - height * 1.5, center.y - height * 1.7, floor_z + height * 2.4),
    )
    key = bpy.context.object
    key.name = "Key Light"
    key.data.energy = 1150
    key.data.shape = "DISK"
    key.data.size = height * 1.8
    key.data.color = (1.0, 0.86, 0.70)
    _look_at(key, center)
    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x + height * 1.7, center.y + height * 0.7, floor_z + height * 1.7),
    )
    rim = bpy.context.object
    rim.name = "Rim Light"
    rim.data.energy = 780
    rim.data.size = height * 1.1
    rim.data.color = (0.68, 0.78, 1.0)
    _look_at(rim, center)
    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x, center.y - height * 1.5, floor_z + height * 0.8),
    )
    fill = bpy.context.object
    fill.name = "Fill Light"
    fill.data.energy = 540
    fill.data.size = height * 2.0
    _look_at(fill, center)

    bpy.ops.object.camera_add(
        location=(
            center.x + height * 0.68,
            center.y - height * 3.05,
            floor_z + height * 1.16,
        )
    )
    camera = bpy.context.object
    camera.data.lens = 58
    _look_at(camera, (center.x, center.y, floor_z + height * 0.52))

    scene = bpy.context.scene
    scene.camera = camera
    scene.frame_start = 1
    scene.frame_end = end
    scene.render.fps = args.fps
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.68, 0.66, 0.62, 1.0)
    background.inputs["Strength"].default_value = 0.66

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output.resolve())

    bpy.ops.render.render(animation=True)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Blender did not create a non-empty MP4 at {args.output}.")

    scene.render.resolution_x = args.gif_resolution
    scene.render.resolution_y = args.gif_resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    gif_frames = list(range(1, end + 1, max(1, args.gif_step)))
    if gif_frames[-1] != end:
        gif_frames.append(end)
    for frame in gif_frames:
        scene.frame_set(frame)
        scene.render.filepath = str(
            (args.frames_dir / f"frame_{frame:04d}.png").resolve()
        )
        bpy.ops.render.render(write_still=True)
    written_frames = sorted(args.frames_dir.glob("frame_*.png"))
    if len(written_frames) != len(gif_frames):
        raise AssertionError(
            f"Expected {len(gif_frames)} GIF source PNGs, found {len(written_frames)}."
        )

    report = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "frames": end,
        "fps": args.fps,
        "resolution": [args.resolution, args.resolution],
        "renderer": "Blender EEVEE Next",
        "gif_resolution": [args.gif_resolution, args.gif_resolution],
        "gif_frame_step": args.gif_step,
        "gif_source_pngs": len(written_frames),
        "sampled_bounds_min": list(minimum),
        "sampled_bounds_max": list(maximum),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
