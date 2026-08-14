#!/usr/bin/env python3
"""Render visual QA sheets for a rigged and animated humanoid asset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rigged", type=Path, required=True)
    parser.add_argument("--animated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials):
        for item in tuple(collection):
            if item.users == 0:
                collection.remove(item)


def _import(path: Path, *, animation: bool):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path.resolve()), use_anim=animation)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise ValueError("Diagnostic input must contain one armature and a mesh.")
    return armatures[0], meshes


def _bounds(meshes) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        points.extend(evaluated.matrix_world @ Vector(value) for value in evaluated.bound_box)
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return low, high


def _material(name: str, color, *, emission=0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 0.62
    if emission:
        shader.inputs["Emission Color"].default_value = color
        shader.inputs["Emission Strength"].default_value = emission
    return material


def _look_at(obj, point) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _scene(meshes, output_dir: Path, resolution: int):
    low, high = _bounds(meshes)
    center = (low + high) * 0.5
    height = high.z - low.z
    body = _material("Diagnostic Warm Clay", (0.48, 0.19, 0.07, 1.0))
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(body)
        for polygon in mesh.data.polygons:
            polygon.use_smooth = True

    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x - height, center.y - height * 1.5, low.z + height * 2.0),
    )
    bpy.context.object.data.energy = 1100
    bpy.context.object.data.size = height * 1.8
    _look_at(bpy.context.object, center)
    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x + height, center.y + height, low.z + height * 1.25),
    )
    bpy.context.object.data.energy = 750
    bpy.context.object.data.size = height * 1.5
    _look_at(bpy.context.object, center)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 58
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.72, 0.70, 0.66, 1.0)
    background.inputs["Strength"].default_value = 0.72
    output_dir.mkdir(parents=True, exist_ok=True)
    return scene, camera, center, low, high, height


def _bone_geometry(armature, height: float, material, *, camera_offset=0.0):
    created = []
    matrix = armature.matrix_world
    for bone in armature.pose.bones:
        head = matrix @ bone.head
        tail = matrix @ bone.tail
        head.y += camera_offset
        tail.y += camera_offset
        direction = tail - head
        if direction.length <= 1e-6:
            continue
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12,
            radius=height * 0.006,
            depth=direction.length,
            location=(head + tail) * 0.5,
        )
        segment = bpy.context.object
        segment.rotation_mode = "QUATERNION"
        segment.rotation_quaternion = direction.to_track_quat("Z", "Y")
        segment.data.materials.append(material)
        created.append(segment)
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2,
            radius=height * 0.012,
            location=head,
        )
        joint = bpy.context.object
        joint.data.materials.append(material)
        created.append(joint)
    return created


def _render(scene, path: Path) -> None:
    scene.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing diagnostic render: {path}")


def _camera_front(camera, center, low, height):
    camera.location = (center.x, center.y - height * 2.9, low.z + height * 0.55)
    _look_at(camera, (center.x, center.y, low.z + height * 0.51))


def _camera_side(camera, center, low, height):
    camera.location = (center.x + height * 2.9, center.y, low.z + height * 0.55)
    _look_at(camera, (center.x, center.y, low.z + height * 0.51))


def _dominant_weight_summary(meshes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mesh in meshes:
        group_names = {group.index: group.name for group in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            if not vertex.groups:
                name = "UNBOUND"
            else:
                strongest = max(vertex.groups, key=lambda item: item.weight)
                name = group_names[strongest.group]
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _assign_dominant_weight_materials(meshes) -> None:
    palette = {
        "Pelvis": (0.92, 0.18, 0.16, 1.0),
        "Spine1": (1.00, 0.38, 0.12, 1.0),
        "Spine2": (1.00, 0.63, 0.10, 1.0),
        "Spine3": (0.96, 0.86, 0.13, 1.0),
        "Neck": (0.75, 0.91, 0.15, 1.0),
        "Head": (0.42, 0.88, 0.20, 1.0),
        "L_Hip": (0.12, 0.70, 0.38, 1.0),
        "L_Knee": (0.08, 0.76, 0.67, 1.0),
        "L_Ankle": (0.05, 0.67, 0.88, 1.0),
        "L_Foot": (0.10, 0.47, 0.94, 1.0),
        "R_Hip": (0.20, 0.35, 0.94, 1.0),
        "R_Knee": (0.40, 0.23, 0.94, 1.0),
        "R_Ankle": (0.64, 0.17, 0.90, 1.0),
        "R_Foot": (0.82, 0.14, 0.77, 1.0),
        "L_Collar": (0.06, 0.86, 0.58, 1.0),
        "L_Shoulder": (0.04, 0.83, 0.83, 1.0),
        "L_Elbow": (0.05, 0.57, 0.96, 1.0),
        "L_Wrist": (0.17, 0.30, 0.96, 1.0),
        "R_Collar": (0.63, 0.17, 0.96, 1.0),
        "R_Shoulder": (0.84, 0.13, 0.86, 1.0),
        "R_Elbow": (0.96, 0.10, 0.57, 1.0),
        "R_Wrist": (0.97, 0.13, 0.31, 1.0),
        "UNBOUND": (0.0, 0.0, 0.0, 1.0),
    }
    def canonical(name: str) -> str:
        value = name.split(":")[-1]
        mixamo = {
            "Hips": "Pelvis", "Spine": "Spine1", "Spine1": "Spine2",
            "Spine2": "Spine3", "Neck": "Neck", "Head": "Head",
            "LeftUpLeg": "L_Hip", "LeftLeg": "L_Knee",
            "LeftFoot": "L_Ankle", "LeftToeBase": "L_Foot",
            "RightUpLeg": "R_Hip", "RightLeg": "R_Knee",
            "RightFoot": "R_Ankle", "RightToeBase": "R_Foot",
            "LeftShoulder": "L_Collar", "LeftArm": "L_Shoulder",
            "LeftForeArm": "L_Elbow", "LeftHand": "L_Wrist",
            "RightShoulder": "R_Collar", "RightArm": "R_Shoulder",
            "RightForeArm": "R_Elbow", "RightHand": "R_Wrist",
        }
        return mixamo.get(value, value if value in palette else "UNBOUND")

    materials = {
        name: _material(f"Weight {name}", color, emission=0.18)
        for name, color in palette.items()
    }
    for mesh in meshes:
        mesh.data.materials.clear()
        names = list(materials)
        for name in names:
            mesh.data.materials.append(materials[name])
        slots = {name: index for index, name in enumerate(names)}
        group_names = {group.index: group.name for group in mesh.vertex_groups}
        dominant = {}
        for vertex in mesh.data.vertices:
            if vertex.groups:
                strongest = max(vertex.groups, key=lambda item: item.weight)
                dominant[vertex.index] = canonical(group_names[strongest.group])
            else:
                dominant[vertex.index] = "UNBOUND"
        for polygon in mesh.data.polygons:
            votes: dict[str, int] = {}
            for vertex in polygon.vertices:
                name = dominant[vertex]
                votes[name] = votes.get(name, 0) + 1
            name = max(votes, key=votes.get)
            polygon.material_index = slots[name]


def _edge_stretch_metrics(meshes, frames: list[int]) -> dict[str, object]:
    scene = bpy.context.scene
    per_frame = {}
    all_log_ratios = []
    topology = []
    for mesh in meshes:
        rest_flat = np.empty(len(mesh.data.vertices) * 3, dtype=np.float64)
        mesh.data.vertices.foreach_get("co", rest_flat)
        rest = rest_flat.reshape(-1, 3)
        world = np.asarray(mesh.matrix_world, dtype=np.float64)
        rest = rest @ world[:3, :3].T + world[:3, 3]
        edges = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
        mesh.data.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        rest_lengths = np.linalg.norm(
            rest[edges[:, 1]] - rest[edges[:, 0]], axis=1
        )
        valid = rest_lengths > 1e-7
        topology.append((mesh, edges, rest_lengths, valid))
    for frame in frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        frame_ratios = []
        for mesh, edges, rest_lengths, valid in topology:
            evaluated = mesh.evaluated_get(depsgraph)
            data = evaluated.to_mesh()
            posed_flat = np.empty(len(data.vertices) * 3, dtype=np.float64)
            data.vertices.foreach_get("co", posed_flat)
            posed = posed_flat.reshape(-1, 3)
            world = np.asarray(evaluated.matrix_world, dtype=np.float64)
            posed = posed @ world[:3, :3].T + world[:3, 3]
            evaluated.to_mesh_clear()
            posed_lengths = np.linalg.norm(
                posed[edges[:, 1]] - posed[edges[:, 0]], axis=1
            )
            frame_ratios.append(posed_lengths[valid] / rest_lengths[valid])
        ratios = np.concatenate(frame_ratios)
        log_ratios = np.abs(np.log(np.maximum(ratios, 1e-8)))
        all_log_ratios.append(log_ratios)
        per_frame[str(frame)] = {
            "p95_ratio_from_one": float(np.exp(np.quantile(log_ratios, 0.95))),
            "p99_ratio_from_one": float(np.exp(np.quantile(log_ratios, 0.99))),
            "p999_ratio_from_one": float(np.exp(np.quantile(log_ratios, 0.999))),
            "fraction_beyond_1_5x": float(np.mean(log_ratios > math.log(1.5))),
            "maximum_ratio_from_one": float(np.exp(np.max(log_ratios))),
        }
    combined = np.concatenate(all_log_ratios)
    frame_p99 = [value["p99_ratio_from_one"] for value in per_frame.values()]
    frame_p999 = [value["p999_ratio_from_one"] for value in per_frame.values()]
    frame_fraction = [value["fraction_beyond_1_5x"] for value in per_frame.values()]
    worst_frame_p999 = max(
        per_frame,
        key=lambda frame: per_frame[frame]["p999_ratio_from_one"],
    )
    return {
        "interpretation": (
            "Symmetric edge-length change: 1.0 is rigid, 1.5 is either 1.5x "
            "stretch or 1/1.5 compression. Maxima and p999 can be dominated "
            "by tiny finger/toe edges; p99 and the beyond-1.5 fraction are the "
            "primary robust gates. The worst-p999 frame is still rendered for "
            "visual inspection."
        ),
        "frames": per_frame,
        "combined_p99_ratio_from_one": float(
            np.exp(np.quantile(combined, 0.99))
        ),
        "combined_p999_ratio_from_one": float(
            np.exp(np.quantile(combined, 0.999))
        ),
        "combined_fraction_beyond_1_5x": float(
            np.mean(combined > math.log(1.5))
        ),
        "maximum_frame_p99_ratio_from_one": float(np.max(frame_p99)),
        "maximum_frame_p999_ratio_from_one": float(np.max(frame_p999)),
        "maximum_frame_fraction_beyond_1_5x": float(np.max(frame_fraction)),
        "worst_frame_p999": int(worst_frame_p999),
    }


def _main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    written = []

    _clear()
    armature, meshes = _import(args.rigged, animation=False)
    scene, camera, center, low, _high, height = _scene(
        meshes, output, args.resolution
    )
    skeleton = _material("Skeleton", (1.0, 0.10, 0.04, 1.0), emission=2.5)
    helpers = _bone_geometry(armature, height, skeleton, camera_offset=-height * 0.018)
    _camera_front(camera, center, low, height)
    rest_front = output / "rest_skeleton_front.png"
    _render(scene, rest_front)
    written.append(rest_front.name)
    _camera_side(camera, center, low, height)
    rest_side = output / "rest_skeleton_side.png"
    _render(scene, rest_side)
    written.append(rest_side.name)
    weight_counts = _dominant_weight_summary(meshes)
    for obj in helpers:
        bpy.data.objects.remove(obj, do_unlink=True)
    _assign_dominant_weight_materials(meshes)
    _camera_front(camera, center, low, height)
    weight_front = output / "dominant_weights_front.png"
    _render(scene, weight_front)
    written.append(weight_front.name)
    camera.location = (center.x, center.y + height * 2.9, low.z + height * 0.55)
    _look_at(camera, (center.x, center.y, low.z + height * 0.51))
    weight_back = output / "dominant_weights_back.png"
    _render(scene, weight_back)
    written.append(weight_back.name)

    _clear()
    armature, meshes = _import(args.animated, animation=True)
    scene, camera, center, low, _high, height = _scene(
        meshes, output, args.resolution
    )
    end = max(
        (
            math.floor(obj.animation_data.action.frame_range[1])
            for obj in [armature, *meshes]
            if obj.animation_data and obj.animation_data.action
        ),
        default=1,
    )
    frames = sorted({1, max(1, end // 4), max(1, end // 2), max(1, end * 3 // 4), end})
    edge_stretch = _edge_stretch_metrics(meshes, list(range(1, end + 1)))
    frames = sorted({*frames, int(edge_stretch["worst_frame_p999"])})
    _camera_front(camera, center, low, height)
    for frame in frames:
        scene.frame_set(frame)
        path = output / f"pose_{frame:04d}_front.png"
        _render(scene, path)
        written.append(path.name)
    _camera_side(camera, center, low, height)
    for frame in frames:
        scene.frame_set(frame)
        path = output / f"pose_{frame:04d}_side.png"
        _render(scene, path)
        written.append(path.name)

    report = {
        "schema_version": 1,
        "rigged": str(args.rigged.resolve()),
        "animated": str(args.animated.resolve()),
        "frames": frames,
        "renders": written,
        "dominant_weight_vertices": weight_counts,
        "edge_stretch": edge_stretch,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
