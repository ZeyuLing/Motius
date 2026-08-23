#!/usr/bin/env python3
"""Render a synchronized, textured multi-character auto-rigging demo video.

The script is deliberately strict about the FBX contract: every character
must contain one SMPL22 armature, animation, skinned geometry, and authored
texture images.  ``--preview`` renders a fast Eevee MP4; the default path is a
full-resolution Cycles render used only after the preview has been approved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

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


def _set_output_properties(scene, output: Path, resolution) -> None:
    scene.render.filepath = str(output.resolve())
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"


def _set_cycles_renderer(scene, camera, samples: int) -> None:
    scene.render.engine = "CYCLES"
    scene.camera = camera
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    try:
        scene.cycles.device = "GPU"
    except TypeError:
        scene.cycles.device = "CPU"


SMPL22_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--character",
        action="append",
        required=True,
        help="LABEL|SOURCE|LICENSE|animated.fbx",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview", action="store_true")
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _parse(value: str) -> dict[str, object]:
    parts = value.split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid --character value: {value!r}")
    label, source, license_name, path = parts
    return {
        "label": label,
        "source": source,
        "license": license_name,
        "path": Path(path),
    }


def _look_at(obj, point) -> None:
    obj.rotation_euler = (
        (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def _material(
    name: str, color, *, roughness: float = 0.7, emission_strength: float = 0.0
):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission is not None:
        emission.default_value = color
    strength = shader.inputs.get("Emission Strength")
    if strength is not None:
        strength.default_value = emission_strength
    return material


def _roots(objects):
    values = set(objects)
    return [obj for obj in objects if obj.parent not in values]


def _texture_audit(meshes) -> dict[str, int]:
    materials = {
        slot.material
        for mesh in meshes
        for slot in mesh.material_slots
        if slot.material is not None
    }
    texture_nodes = {
        node
        for material in materials
        if material.use_nodes
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    }
    images = {node.image for node in texture_nodes}
    return {
        "materials": len(materials),
        "texture_nodes": len(texture_nodes),
        "texture_images": len(images),
        "packed_textures": sum(image.packed_file is not None for image in images),
    }


def _frame_bounds(meshes, frame: int) -> tuple[Vector, Vector]:
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        points.extend(
            evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box
        )
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return low, high


def _sampled_bounds(meshes, end: int) -> tuple[Vector, Vector, list[int]]:
    frames = sorted({1, max(1, end // 4), max(1, end // 2), 103, end})
    frames = [frame for frame in frames if frame <= end]
    values = [_frame_bounds(meshes, frame) for frame in frames]
    low = Vector(tuple(min(value[0][axis] for value in values) for axis in range(3)))
    high = Vector(tuple(max(value[1][axis] for value in values) for axis in range(3)))
    return low, high, frames


def _import_character(character: dict[str, object], end: int):
    path = character["path"]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path.resolve()), use_anim=True)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise ValueError(f"Expected one armature and mesh geometry: {path}")
    missing = sorted(set(SMPL22_NAMES).difference(armatures[0].data.bones.keys()))
    if missing:
        raise ValueError(f"Missing SMPL22 bones {missing}: {path}")
    actions = [
        obj.animation_data.action
        for obj in imported
        if obj.animation_data and obj.animation_data.action
    ]
    action_end = max(
        (math.floor(action.frame_range[1]) for action in actions), default=0
    )
    if action_end < end:
        raise ValueError(
            f"Animation ends at {action_end}, before requested frame {end}: {path}"
        )
    if any(not mesh.vertex_groups for mesh in meshes):
        raise ValueError(f"Imported animated geometry lost skin weights: {path}")
    audit = _texture_audit(meshes)
    if audit["texture_images"] < 1 or audit["packed_textures"] < 1:
        raise ValueError(f"Animated FBX lost authored packed textures: {path}")
    for mesh in meshes:
        for polygon in mesh.data.polygons:
            polygon.use_smooth = True
    return imported, armatures[0], meshes, audit


def _normalize_character(imported, meshes, x: float, end: int):
    low, high, sampled_frames = _sampled_bounds(meshes, end)
    height = max(high.z - low.z, 1e-6)
    scale = 2.55 / height
    center = (low + high) * 0.5
    root = bpy.data.objects.new("Demo Character Root", None)
    bpy.context.collection.objects.link(root)
    for obj in _roots(imported):
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world
    root.scale = (scale, scale, scale)
    root.location = (
        x - center.x * scale,
        -center.y * scale,
        0.06 - low.z * scale,
    )
    bpy.context.view_layer.update()
    report = {
        "sampled_frames": sampled_frames,
        "sampled_bounds_min": list(low),
        "sampled_bounds_max": list(high),
        "display_scale": scale,
    }
    # The camera is on -Y. Put the diagnostic skeleton on a plane just in
    # front of the nearest sampled surface so every joint and limb remains
    # legible while retaining the character's authored opaque textures.
    front_y = root.location.y + low.y * scale - 0.10
    return report, front_y


def _key_linear(obj, frame: int, paths) -> None:
    for path in paths:
        obj.keyframe_insert(path, frame=frame)


def _segment_transform(obj, start: Vector, end: Vector) -> None:
    direction = end - start
    length = max(direction.length, 1e-6)
    obj.location = (start + end) * 0.5
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.scale = (1.0, 1.0, length * 0.5)


def _animate_skeleton_overlay(
    armature, front_y: float, end: int, joint_material, limb_material
) -> dict[str, object]:
    """Draw the canonical SMPL22 rig as an animated, front-plane overlay."""

    joints = []
    for index, name in enumerate(SMPL22_NAMES):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.038)
        obj = bpy.context.object
        obj.name = f"SMPL22 Joint {index:02d} {name}"
        obj.data.materials.append(joint_material)
        obj.visible_shadow = False
        joints.append(obj)

    limbs = []
    limb_indices = []
    for child, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            continue
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.022, depth=2.0)
        obj = bpy.context.object
        obj.name = f"SMPL22 Limb {SMPL22_NAMES[parent]} to {SMPL22_NAMES[child]}"
        obj.rotation_mode = "QUATERNION"
        obj.data.materials.append(limb_material)
        obj.visible_shadow = False
        limbs.append(obj)
        limb_indices.append((int(parent), child))

    for frame in range(1, end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        points = []
        for name in SMPL22_NAMES:
            point = armature.matrix_world @ armature.pose.bones[name].head
            points.append(Vector((point.x, front_y, point.z)))
        for obj, point in zip(joints, points):
            obj.location = point
            _key_linear(obj, frame, ("location",))
        for obj, (parent, child) in zip(limbs, limb_indices):
            _segment_transform(obj, points[parent], points[child])
            _key_linear(obj, frame, ("location", "rotation_quaternion", "scale"))

    for obj in joints + limbs:
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
    return {
        "profile": "SMPL22",
        "joints": len(joints),
        "limbs": len(limbs),
        "display": "animated front-plane overlay",
    }


def _main() -> None:
    args = _args()
    characters = [_parse(value) for value in args.character]
    if len(characters) != 3:
        raise ValueError("The synchronized demo expects exactly three characters.")
    for character in characters:
        if not character["path"].is_file():
            raise FileNotFoundError(character["path"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.render.fps = args.fps

    columns = (-2.75, 0.0, 2.75)
    audits = []
    normalization = []
    skeletons = []
    joint_material = _material(
        "SMPL22 joints",
        (0.97, 0.99, 1.0, 1.0),
        roughness=0.22,
        emission_strength=0.45,
    )
    limb_material = _material(
        "SMPL22 limbs",
        (0.02, 0.68, 0.92, 1.0),
        roughness=0.28,
        emission_strength=0.35,
    )
    for x, character in zip(columns, characters):
        imported, armature, meshes, audit = _import_character(character, args.frames)
        norm, front_y = _normalize_character(imported, meshes, x, args.frames)
        normalization.append(norm)
        audits.append(audit)
        skeletons.append(
            _animate_skeleton_overlay(
                armature, front_y, args.frames, joint_material, limb_material
            )
        )

    ground_material = _material("Studio floor", (0.80, 0.81, 0.82, 1.0), roughness=0.72)
    bpy.ops.mesh.primitive_plane_add(size=24.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(ground_material)

    for location, energy, size, color in (
        ((-5.0, -5.5, 7.0), 1450, 5.0, (1.0, 0.91, 0.82)),
        ((5.0, -2.0, 5.0), 1050, 4.0, (0.78, 0.87, 1.0)),
        ((0.0, 3.5, 6.0), 1150, 4.5, (1.0, 1.0, 1.0)),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.size = size
        light.data.color = color
        _look_at(light, (0.0, 0.0, 1.25))

    bpy.ops.object.camera_add(location=(0.0, -15.0, 1.75))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    # Blender's orthographic scale follows the fitted sensor dimension.  With
    # this landscape output, 4.35 framed only the centre character and clipped
    # the two outer columns.  Keep enough horizontal margin for all three
    # normalized characters and their widest sampled poses.
    camera.data.ortho_scale = 7.6
    _look_at(camera, (0.0, 0.0, 1.20))
    scene.camera = camera

    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.86, 0.87, 0.88, 1.0)
    background.inputs["Strength"].default_value = 0.75
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"

    if args.preview:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.camera = camera
        renderer = "Blender EEVEE Next"
        resolution = (960, 540)
        samples = 16
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        _set_cycles_renderer(scene, camera, samples=256)
        renderer = "Blender Cycles"
        resolution = (1920, 1080)
        samples = 256

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    _set_output_properties(scene, args.output, resolution)
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"
    bpy.ops.render.render(animation=True)
    if not args.output.is_file() or args.output.stat().st_size <= 0:
        raise RuntimeError(f"Blender did not create a non-empty MP4: {args.output}")

    review_frames = sorted({1, 37, 75, 103, args.frames})
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    for frame in review_frames:
        scene.frame_set(frame)
        scene.render.filepath = str(
            (args.frames_dir / f"frame_{frame:04d}.png").resolve()
        )
        bpy.ops.render.render(write_still=True)

    report = {
        "schema_version": 1,
        "preview": args.preview,
        "renderer": renderer,
        "samples": samples,
        "output": str(args.output.resolve()),
        "resolution": list(resolution),
        "frames": args.frames,
        "fps": args.fps,
        "duration_seconds": args.frames / args.fps,
        "review_frames": review_frames,
        "motion": {
            "case_id": "004822",
            "representation": "HumanML3D SMPL22 joints",
            "synchronized": True,
        },
        "presentation": {
            "text_overlays": False,
            "skeleton_overlay": True,
        },
        "characters": [
            {
                "label": character["label"],
                "source": character["source"],
                "license": character["license"],
                "animated_fbx": str(character["path"].resolve()),
                "texture_audit": audit,
                "normalization": norm,
                "skeleton": skeleton,
            }
            for character, audit, norm, skeleton in zip(
                characters, audits, normalization, skeletons
            )
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
