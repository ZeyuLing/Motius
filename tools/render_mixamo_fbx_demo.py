#!/usr/bin/env python3
"""Render one source motion beside three retargeted FBX characters in Blender."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
SOURCE_TO_BLENDER = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--fbx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _look_at(obj, point) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _material(name: str, color, metallic=0.0, roughness=0.6):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    return material


def _resample(joints: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if source_fps == target_fps:
        return joints
    duration = (len(joints) - 1) / source_fps
    count = int(round(duration * target_fps)) + 1
    old = np.arange(len(joints), dtype=np.float64) / source_fps
    new = np.linspace(0.0, duration, count)
    flat = joints.reshape(len(joints), -1)
    output = np.stack([np.interp(new, old, flat[:, dim]) for dim in range(flat.shape[1])], axis=-1)
    return output.reshape(count, 22, 3)


def _source_skeleton(joints: np.ndarray, x_offset: float):
    value = joints @ SOURCE_TO_BLENDER.T
    value[..., 0] -= value[0, 0, 0]
    value[..., 1] -= value[0, 0, 1]
    value[..., 2] -= value[..., 2].min()
    value[..., 0] += x_offset
    joint_material = _material("source_joint", (0.05, 0.64, 0.78, 1.0), metallic=0.12)
    bone_material = _material("source_bone", (0.02, 0.20, 0.28, 1.0), metallic=0.08)
    joints_objects = []
    bone_objects = []
    for index in range(22):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.035)
        obj = bpy.context.object
        obj.name = f"HML_Joint_{index:02d}"
        obj.data.materials.append(joint_material)
        joints_objects.append(obj)
        if PARENTS[index] >= 0:
            bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.021, depth=1.0)
            bone = bpy.context.object
            bone.name = f"HML_Bone_{PARENTS[index]:02d}_{index:02d}"
            bone.data.materials.append(bone_material)
            bone_objects.append((index, PARENTS[index], bone))
    for frame_index, pose in enumerate(value):
        frame = frame_index + 1
        for index, obj in enumerate(joints_objects):
            obj.location = pose[index]
            obj.keyframe_insert(data_path="location", frame=frame)
        for child, parent, obj in bone_objects:
            vector = Vector(pose[child] - pose[parent])
            obj.location = (Vector(pose[child]) + Vector(pose[parent])) * 0.5
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(vector)
            obj.scale = (1.0, 1.0, vector.length)
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)


def _import_character(path: Path, x_offset: float):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path.resolve()), use_anim=True)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError(f"{path} imported {len(armatures)} armatures")
    imported_set = set(imported)
    root = bpy.data.objects.new(f"DemoRoot_{path.stem}", None)
    bpy.context.collection.objects.link(root)
    for obj in imported:
        if obj.parent not in imported_set:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    root.location.x = x_offset
    return armatures[0], imported


def _text(label: str, x: float) -> None:
    bpy.ops.object.text_add(location=(x, -0.72, 2.12), rotation=(math.pi / 2, 0.0, 0.0))
    text = bpy.context.object
    text.data.body = label
    text.data.align_x = "CENTER"
    text.data.align_y = "CENTER"
    text.data.size = 0.13
    text.data.extrude = 0.004
    text.data.materials.append(_material(f"label_{label}", (0.88, 0.91, 0.94, 1.0)))


def _scene(output: Path, fps: int, start_frame: int, end_frame: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 450
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(output.resolve())
    scene.render.film_transparent = False
    scene.render.fps = fps
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.018, 0.023, 0.030, 1.0)
    background.inputs["Strength"].default_value = 0.30

    ground = _material("ground", (0.075, 0.085, 0.10, 1.0), roughness=0.82)
    bpy.ops.mesh.primitive_plane_add(size=28, location=(0, 0, -0.01))
    bpy.context.object.data.materials.append(ground)

    bpy.ops.object.light_add(type="AREA", location=(-4.0, -5.0, 7.0))
    bpy.context.object.data.energy = 1050
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5.0
    _look_at(bpy.context.object, (0, 0, 1.0))
    bpy.ops.object.light_add(type="AREA", location=(5.0, -1.0, 4.0))
    bpy.context.object.data.energy = 700
    bpy.context.object.data.size = 4.0
    _look_at(bpy.context.object, (0, 0, 1.0))

    bpy.ops.object.camera_add(location=(0.0, -16.5, 2.7))
    camera = bpy.context.object
    camera.data.lens = 52
    _look_at(camera, (0.0, 0.0, 1.0))
    scene.camera = camera


def main() -> None:
    args = _args()
    _clear()
    joints = np.load(args.joints, allow_pickle=False)
    joints = _resample(np.asarray(joints)[:120], args.source_fps, args.fps)
    start_frame = max(1, args.start_frame)
    end_frame = min(len(joints), start_frame + args.frames - 1)
    joints = joints[:end_frame]
    offsets = (-4.15, -1.38, 1.38, 4.15)
    _source_skeleton(joints, offsets[0])
    for slug, offset in zip(("atlas", "nova", "gear"), offsets[1:]):
        _import_character(args.fbx_dir / f"004822_{slug}.fbx", offset)
    for label, offset in zip(("HumanML3D-263", "Atlas FBX", "Nova FBX", "Gear FBX"), offsets):
        _text(label, offset)
    _scene(args.output, args.fps, start_frame, end_frame)
    bpy.context.scene.frame_set(start_frame)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
