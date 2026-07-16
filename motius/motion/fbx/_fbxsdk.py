#!/usr/bin/env python3
"""Autodesk FBX SDK subprocess backend for character animation retargeting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import fbx
import FbxCommon
import numpy as np
from scipy.spatial.transform import Rotation


_ARM_DIRECTION_CHILD = {
    "L_Collar": "L_Shoulder",
    "R_Collar": "R_Shoulder",
    "L_Shoulder": "L_Elbow",
    "R_Shoulder": "R_Elbow",
    "L_Elbow": "L_Wrist",
    "R_Elbow": "R_Wrist",
}

_ROTATION_ORDER = {
    fbx.EFbxRotationOrder.eEulerXYZ: "xyz",
    fbx.EFbxRotationOrder.eEulerXZY: "xzy",
    fbx.EFbxRotationOrder.eEulerYXZ: "yxz",
    fbx.EFbxRotationOrder.eEulerYZX: "yzx",
    fbx.EFbxRotationOrder.eEulerZXY: "zxy",
    fbx.EFbxRotationOrder.eEulerZYX: "zyx",
}


def _load_mapping_module():
    path = Path(__file__).with_name("_mapping.py")
    spec = importlib.util.spec_from_file_location("motius_fbx_mapping", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FBX bone mapping helpers from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAPPING = _load_mapping_module()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args()


def _matrix4(value) -> np.ndarray:
    # FBX SDK matrices use row-vector composition. Motius and NumPy use
    # column-vector composition, so transpose once at the boundary.
    return np.asarray(
        [[float(value.Get(row, column)) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    ).T


def _proper_rotation(value) -> np.ndarray:
    matrix = _matrix4(value)[:3, :3]
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _translation(value) -> np.ndarray:
    vector = value.GetT()
    return np.asarray([vector[0], vector[1], vector[2]], dtype=np.float64)


def _fbx_euler_matrix(value) -> np.ndarray:
    matrix = fbx.FbxAMatrix()
    matrix.SetIdentity()
    matrix.SetR(fbx.FbxVector4(float(value[0]), float(value[1]), float(value[2])))
    return _proper_rotation(matrix)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-8 or target_norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(source, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.asarray([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi).as_matrix()
    axis = cross / sine
    return Rotation.from_rotvec(axis * math.atan2(sine, cosine)).as_matrix()


def _walk(node):
    yield node
    for index in range(node.GetChildCount()):
        yield from _walk(node.GetChild(index))


def _is_skeleton(node) -> bool:
    attribute = node.GetNodeAttribute()
    return bool(
        attribute
        and attribute.GetAttributeType()
        == fbx.FbxNodeAttribute.EType.eSkeleton
    )


def _collect_scene(scene, requested_root: str | None):
    all_nodes = list(_walk(scene.GetRootNode()))
    all_names: dict[str, list[object]] = {}
    for node in all_nodes:
        all_names.setdefault(node.GetName(), []).append(node)

    if requested_root:
        roots = all_names.get(requested_root, ())
        if not roots:
            raise ValueError(
                f"target_armature {requested_root!r} was not found in the FBX scene."
            )
        if len(roots) > 1:
            raise ValueError(
                f"target_armature {requested_root!r} is ambiguous in the FBX scene."
            )
        selected = {int(node.GetUniqueID()) for node in _walk(roots[0])}
        skeleton_nodes = [
            node
            for node in all_nodes
            if int(node.GetUniqueID()) in selected and _is_skeleton(node)
        ]
        armature_name = requested_root
    else:
        skeleton_nodes = [node for node in all_nodes if _is_skeleton(node)]
        roots = [
            node
            for node in skeleton_nodes
            if node.GetParent() is None or not _is_skeleton(node.GetParent())
        ]
        if not skeleton_nodes:
            raise ValueError("The target FBX contains no skeleton nodes.")
        armature_name = roots[0].GetName() if len(roots) == 1 else skeleton_nodes[0].GetName()

    names: dict[str, list[object]] = {}
    for node in skeleton_nodes:
        names.setdefault(node.GetName(), []).append(node)
    duplicates = sorted(name for name, nodes in names.items() if len(nodes) > 1)
    if duplicates:
        raise ValueError(
            "FBX skeleton node names must be unique for deterministic retargeting; "
            f"duplicate names: {duplicates[:20]}."
        )

    mesh_nodes = []
    for node in all_nodes:
        attribute = node.GetNodeAttribute()
        if not attribute or attribute.GetAttributeType() != fbx.FbxNodeAttribute.EType.eMesh:
            continue
        if attribute.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eSkin) > 0:
            mesh_nodes.append(node)
    if not mesh_nodes:
        raise ValueError(
            "The target FBX has no skinned mesh. Motius animates an existing rig; "
            "it does not auto-rig a static mesh."
        )
    return names, skeleton_nodes, mesh_nodes, armature_name


def _axis_description(axis_system) -> dict[str, object]:
    up_axis, up_sign = axis_system.GetUpVector()
    front_axis, front_sign = axis_system.GetFrontVector()
    return {
        "up_axis": str(up_axis),
        "up_sign": int(up_sign),
        "front_axis": str(front_axis),
        "front_sign": int(front_sign),
        "coordinate_system": str(axis_system.GetCoorSystem()),
    }


def _clear_animations(scene) -> None:
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
    stacks = [scene.GetSrcObject(criteria, index) for index in range(scene.GetSrcObjectCount(criteria))]
    for stack in stacks:
        if stack is not None:
            stack.Destroy()


def _depth(node) -> int:
    depth = 0
    parent = node.GetParent()
    while parent is not None:
        depth += 1
        parent = parent.GetParent()
    return depth


def _skeleton_height(points: np.ndarray) -> float:
    extent = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    if extent <= 1e-6:
        raise ValueError("Cannot infer root-motion scale from a zero-height skeleton.")
    return extent


def _channel_sequence(node, channel_matrix: np.ndarray) -> str:
    order = node.GetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot)
    base = _ROTATION_ORDER.get(order)
    if base is None:
        raise ValueError(f"Unsupported FBX rotation order on {node.GetName()!r}: {order}.")
    rest = node.LclRotation.Get()
    euler = np.asarray([rest[0], rest[1], rest[2]], dtype=np.float64)
    candidates = (base, base.upper())
    errors = [
        np.linalg.norm(Rotation.from_euler(sequence, euler, degrees=True).as_matrix() - channel_matrix)
        for sequence in candidates
    ]
    return candidates[int(np.argmin(errors))]


def _set_curve(property_value, layer, component: str, values: np.ndarray, fps: float) -> None:
    curve = property_value.GetCurve(layer, component, True)
    curve.KeyModifyBegin()
    time = fbx.FbxTime()
    component_index = {"X": 0, "Y": 1, "Z": 2}[component]
    for frame, vector in enumerate(values):
        time.SetSecondDouble(frame / fps)
        key = curve.KeyAdd(time)[0]
        curve.KeySetValue(key, float(vector[component_index]))
        curve.KeySetInterpolation(
            key, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
        )
    curve.KeyModifyEnd()


def _animate(
    scene,
    payload,
    nodes_by_name,
    bone_map,
    fps: float,
    root_scale,
    *,
    align_root_start: bool = True,
):
    source_names = [str(value) for value in payload["joint_names"]]
    source_index = {name: index for index, name in enumerate(source_names)}
    joints = np.asarray(payload["joints"], dtype=np.float64)
    source_global = np.asarray(payload["global_rotations"], dtype=np.float64)
    frames = len(joints)
    source_rest = np.asarray(payload["rest_joints"], dtype=np.float64)

    target_nodes = {source: nodes_by_name[target][0] for source, target in bone_map.items()}
    reverse_map = {target: source for source, target in bone_map.items()}
    mapped_nodes = list(target_nodes.values())
    relevant_nodes = {int(node.GetUniqueID()): node for node in mapped_nodes}
    for node in tuple(mapped_nodes):
        parent = node.GetParent()
        while parent is not None:
            relevant_nodes[int(parent.GetUniqueID())] = parent
            parent = parent.GetParent()

    rest_global_matrix = {
        node_id: _matrix4(node.EvaluateGlobalTransform())
        for node_id, node in relevant_nodes.items()
    }
    rest_global_rotation = {
        node_id: _proper_rotation(node.EvaluateGlobalTransform())
        for node_id, node in relevant_nodes.items()
    }
    rest_global_position = {
        node_id: _translation(node.EvaluateGlobalTransform())
        for node_id, node in relevant_nodes.items()
    }

    target_points = np.stack(
        [rest_global_position[int(node.GetUniqueID())] for node in mapped_nodes]
    )
    if root_scale == "auto":
        root_scale = _skeleton_height(target_points) / _skeleton_height(source_rest[: len(source_names)])
    root_scale = float(root_scale)

    desired_global: dict[str, np.ndarray] = {}
    for source_name, node in target_nodes.items():
        source_joint = source_index[source_name]
        node_id = int(node.GetUniqueID())
        desired_global[source_name] = np.einsum(
            "fij,jk->fik", source_global[:, source_joint], rest_global_rotation[node_id]
        )

    for source_name, child_name in _ARM_DIRECTION_CHILD.items():
        if source_name not in target_nodes or child_name not in target_nodes:
            continue
        source_joint = source_index[source_name]
        child_joint = source_index[child_name]
        target_axis = (
            rest_global_position[int(target_nodes[child_name].GetUniqueID())]
            - rest_global_position[int(target_nodes[source_name].GetUniqueID())]
        )
        for frame in range(frames):
            current = source_global[frame, source_joint] @ target_axis
            wanted = joints[frame, child_joint] - joints[frame, source_joint]
            desired_global[source_name][frame] = (
                _rotation_between(current, wanted) @ desired_global[source_name][frame]
            )

    ordered = sorted(target_nodes.items(), key=lambda item: _depth(item[1]))
    rotation_values: dict[str, np.ndarray] = {}
    rest_channel_errors = {}
    for source_name, node in ordered:
        parent = node.GetParent()
        parent_source = reverse_map.get(parent.GetName()) if parent is not None else None
        if parent_source in desired_global:
            parent_rotations = desired_global[parent_source]
        else:
            parent_id = int(parent.GetUniqueID()) if parent is not None else None
            parent_rest = (
                rest_global_rotation[parent_id]
                if parent_id in rest_global_rotation
                else np.eye(3)
            )
            parent_rotations = np.broadcast_to(parent_rest, (frames, 3, 3))
        effective_local = np.einsum(
            "fji,fjk->fik", parent_rotations, desired_global[source_name]
        )

        if node.GetRotationActive():
            pre_value = node.GetPreRotation(fbx.FbxNode.EPivotSet.eSourcePivot)
            post_value = node.GetPostRotation(fbx.FbxNode.EPivotSet.eSourcePivot)
            pre = _fbx_euler_matrix(pre_value)
            post = _fbx_euler_matrix(post_value)
        else:
            pre = np.eye(3, dtype=np.float64)
            post = np.eye(3, dtype=np.float64)
        channels = np.einsum("ij,fjk,kl->fil", pre.T, effective_local, post)

        parent_id = int(parent.GetUniqueID()) if parent is not None else None
        node_id = int(node.GetUniqueID())
        parent_rest_rotation = (
            rest_global_rotation[parent_id]
            if parent_id in rest_global_rotation
            else np.eye(3)
        )
        rest_effective = parent_rest_rotation.T @ rest_global_rotation[node_id]
        rest_channel_matrix = pre.T @ rest_effective @ post
        sequence = _channel_sequence(node, rest_channel_matrix)
        rest_value = node.LclRotation.Get()
        rest_euler = np.asarray([rest_value[0], rest_value[1], rest_value[2]])
        reconstructed = Rotation.from_euler(sequence, rest_euler, degrees=True).as_matrix()
        rest_channel_errors[source_name] = float(
            np.linalg.norm(reconstructed - rest_channel_matrix)
        )
        if rest_channel_errors[source_name] > 2e-3:
            raise ValueError(
                f"Cannot solve FBX rotation basis for {node.GetName()!r}; "
                f"rest-matrix error={rest_channel_errors[source_name]:.6f}."
            )

        with np.errstate(invalid="ignore"):
            eulers = Rotation.from_matrix(channels).as_euler(sequence, degrees=False)
        eulers = np.unwrap(eulers, axis=0)
        rotation_values[source_name] = np.rad2deg(eulers)

    root_source = "Pelvis"
    root_node = target_nodes[root_source]
    root_parent = root_node.GetParent()
    root_parent_id = int(root_parent.GetUniqueID()) if root_parent is not None else None
    parent_matrix = (
        rest_global_matrix[root_parent_id]
        if root_parent_id in rest_global_matrix
        else np.eye(4)
    )
    parent_inverse = np.linalg.inv(parent_matrix)
    root_rest_global = rest_global_position[int(root_node.GetUniqueID())]
    root_rest_local_effective = _matrix4(root_node.EvaluateLocalTransform())[:3, 3]
    root_lcl = root_node.LclTranslation.Get()
    root_lcl_rest = np.asarray([root_lcl[0], root_lcl[1], root_lcl[2]], dtype=np.float64)
    if align_root_start:
        root_delta = (
            joints[:, source_index[root_source]]
            - joints[0, source_index[root_source]]
        )
    else:
        root_delta = (
            joints[:, source_index[root_source]]
            - source_rest[source_index[root_source]]
        )
    desired_root_global = root_rest_global + root_delta * root_scale
    homogeneous = np.concatenate(
        [desired_root_global, np.ones((frames, 1), dtype=np.float64)], axis=1
    )
    desired_root_local_effective = (parent_inverse @ homogeneous.T).T[:, :3]
    root_translation = root_lcl_rest + (
        desired_root_local_effective - root_rest_local_effective
    )

    _clear_animations(scene)
    stack = fbx.FbxAnimStack.Create(scene, "Motius_Retargeted_Animation")
    layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)
    time_mode = fbx.FbxTime.ConvertFrameRateToTimeMode(fps)
    scene.GetGlobalSettings().SetTimeMode(time_mode)

    for source_name, node in ordered:
        values = rotation_values[source_name]
        for component in "XYZ":
            _set_curve(node.LclRotation, layer, component, values, fps)
    for component in "XYZ":
        _set_curve(root_node.LclTranslation, layer, component, root_translation, fps)

    start = fbx.FbxTime()
    stop = fbx.FbxTime()
    start.SetSecondDouble(0.0)
    stop.SetSecondDouble((frames - 1) / fps)
    stack.LocalStart.Set(start)
    stack.LocalStop.Set(stop)

    direction_errors = []
    time = fbx.FbxTime()
    for frame in range(frames):
        time.SetSecondDouble(frame / fps)
        for source_name, child_name in _ARM_DIRECTION_CHILD.items():
            if source_name not in target_nodes or child_name not in target_nodes:
                continue
            source_axis = (
                joints[frame, source_index[child_name]]
                - joints[frame, source_index[source_name]]
            )
            target_axis = (
                _translation(target_nodes[child_name].EvaluateGlobalTransform(time))
                - _translation(target_nodes[source_name].EvaluateGlobalTransform(time))
            )
            if np.linalg.norm(source_axis) > 1e-8 and np.linalg.norm(target_axis) > 1e-8:
                cosine = float(
                    np.clip(
                        np.dot(source_axis, target_axis)
                        / (np.linalg.norm(source_axis) * np.linalg.norm(target_axis)),
                        -1.0,
                        1.0,
                    )
                )
                direction_errors.append(math.degrees(math.acos(cosine)))

    diagnostics = {
        "arm_chain_direction_error_deg_mean": float(np.mean(direction_errors)),
        "arm_chain_direction_error_deg_p95": float(np.percentile(direction_errors, 95)),
        "arm_chain_direction_error_deg_max": float(np.max(direction_errors)),
        "rest_channel_matrix_error_max": float(max(rest_channel_errors.values())),
    }
    return stack, root_scale, diagnostics


def _create_smpl_scene(scene, payload):
    vertices = np.asarray(payload["vertices"], dtype=np.float64)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    weights = np.asarray(payload["weights"], dtype=np.float64)
    rest_joints = np.asarray(payload["rest_joints"], dtype=np.float64)
    parents = np.asarray(payload["parents"], dtype=np.int64)
    joint_names = [str(value) for value in payload["joint_names"]]

    scene.GetGlobalSettings().SetAxisSystem(fbx.FbxAxisSystem.MayaZUp)
    scene.GetGlobalSettings().SetSystemUnit(fbx.FbxSystemUnit.m)
    skeleton_nodes = []
    for joint, name in enumerate(joint_names):
        attribute = fbx.FbxSkeleton.Create(scene, f"{name}_Skeleton")
        skeleton_type = (
            fbx.FbxSkeleton.EType.eRoot
            if parents[joint] < 0
            else fbx.FbxSkeleton.EType.eLimbNode
        )
        attribute.SetSkeletonType(skeleton_type)
        node = fbx.FbxNode.Create(scene, name)
        node.SetNodeAttribute(attribute)
        parent = int(parents[joint])
        offset = rest_joints[joint] if parent < 0 else rest_joints[joint] - rest_joints[parent]
        node.LclTranslation.Set(
            fbx.FbxDouble3(float(offset[0]), float(offset[1]), float(offset[2]))
        )
        if parent < 0:
            scene.GetRootNode().AddChild(node)
        else:
            skeleton_nodes[parent].AddChild(node)
        skeleton_nodes.append(node)

    mesh = fbx.FbxMesh.Create(scene, "SMPL_Mesh")
    mesh.InitControlPoints(len(vertices))
    for index, vertex in enumerate(vertices):
        mesh.SetControlPointAt(
            fbx.FbxVector4(float(vertex[0]), float(vertex[1]), float(vertex[2])),
            index,
        )
    for face in faces:
        mesh.BeginPolygon(0)
        for index in face:
            mesh.AddPolygon(int(index))
        mesh.EndPolygon()

    face_vertices = vertices[faces]
    face_normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    vertex_normals = np.zeros_like(vertices)
    for corner in range(faces.shape[1]):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(lengths, 1e-12)
    normal_element = fbx.FbxLayerElementNormal.Create(mesh, "SMPL_Normals")
    normal_element.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByControlPoint)
    normal_element.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
    for normal in vertex_normals:
        normal_element.GetDirectArray().Add(
            fbx.FbxVector4(float(normal[0]), float(normal[1]), float(normal[2]))
        )
    if mesh.GetLayer(0) is None:
        mesh.CreateLayer()
    mesh.GetLayer(0).SetNormals(normal_element)

    mesh_node = fbx.FbxNode.Create(scene, "SMPL_Mesh")
    mesh_node.SetNodeAttribute(mesh)
    material = fbx.FbxSurfacePhong.Create(scene, "SMPL_Material")
    material.Diffuse.Set(fbx.FbxDouble3(0.72, 0.46, 0.34))
    material.Specular.Set(fbx.FbxDouble3(0.08, 0.08, 0.08))
    material.Shininess.Set(12.0)
    mesh_node.AddMaterial(material)
    scene.GetRootNode().AddChild(mesh_node)

    skin = fbx.FbxSkin.Create(scene, "SMPL_Skin")
    mesh_transform = mesh_node.EvaluateGlobalTransform()
    for joint, node in enumerate(skeleton_nodes):
        cluster = fbx.FbxCluster.Create(scene, f"{joint_names[joint]}_Cluster")
        cluster.SetLink(node)
        cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)
        indices = np.flatnonzero(weights[:, joint] > 1e-8)
        for vertex in indices:
            cluster.AddControlPointIndex(int(vertex), float(weights[vertex, joint]))
        cluster.SetTransformMatrix(mesh_transform)
        cluster.SetTransformLinkMatrix(node.EvaluateGlobalTransform())
        skin.AddCluster(cluster)
    mesh.AddDeformer(skin)

    bind_pose = fbx.FbxPose.Create(scene, "SMPL_BindPose")
    bind_pose.SetIsBindPose(True)
    bind_pose.Add(mesh_node, fbx.FbxMatrix(mesh_node.EvaluateGlobalTransform()))
    for node in skeleton_nodes:
        bind_pose.Add(node, fbx.FbxMatrix(node.EvaluateGlobalTransform()))
    scene.AddPose(bind_pose)

    nodes_by_name = {node.GetName(): [node] for node in skeleton_nodes}
    return nodes_by_name, skeleton_nodes, [mesh_node], "SMPL_Armature"


def _save_scene(manager, scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = manager.GetIOSettings()
    settings.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
    settings.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
    settings.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
    exporter = fbx.FbxExporter.Create(manager, "")
    writer = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    if not exporter.Initialize(str(output), writer, settings):
        message = exporter.GetStatus().GetErrorString()
        exporter.Destroy()
        raise RuntimeError(f"FBX exporter initialization failed: {message}")
    if not exporter.Export(scene):
        message = exporter.GetStatus().GetErrorString()
        exporter.Destroy()
        raise RuntimeError(f"FBX export failed: {message}")
    exporter.Destroy()


def _main() -> None:
    args = _parse_args()
    job = json.loads(args.job.read_text())
    payload = np.load(job["payload_path"], allow_pickle=False)
    manager, scene = FbxCommon.InitializeSdkObjects()
    try:
        if job["mode"] == "character_retarget":
            character = Path(job["character_fbx"])
            if not FbxCommon.LoadScene(manager, scene, str(character)):
                raise RuntimeError(f"Failed to load target character FBX: {character}.")
            original_axis = scene.GetGlobalSettings().GetAxisSystem()
            original_axis_description = _axis_description(original_axis)
            fbx.FbxAxisSystem.MayaZUp.ConvertScene(scene)
            nodes_by_name, skeleton_nodes, mesh_nodes, armature_name = _collect_scene(
                scene, job.get("target_armature")
            )
            bone_map = MAPPING.resolve_bone_map(
                [node.GetName() for node in skeleton_nodes],
                job.get("bone_map"),
                strict=bool(job.get("strict_bone_map", True)),
            )
            if "Pelvis" not in bone_map:
                raise ValueError("The target bone map must include Pelvis for root motion.")
            _, root_scale, diagnostics = _animate(
                scene,
                payload,
                nodes_by_name,
                bone_map,
                float(job["fps"]),
                job.get("root_motion_scale", "auto"),
            )
            output_axis = original_axis
            character_value = str(character)
            skin_description = (
                "target character mesh, materials, hierarchy, and skin weights preserved"
            )
        elif job["mode"] == "smpl_export":
            nodes_by_name, skeleton_nodes, mesh_nodes, armature_name = _create_smpl_scene(
                scene, payload
            )
            bone_map = {
                name: name
                for name in MAPPING.SMPL22_BONE_NAMES
                if name in nodes_by_name
            }
            _, root_scale, diagnostics = _animate(
                scene,
                payload,
                nodes_by_name,
                bone_map,
                float(job["fps"]),
                1.0,
                align_root_start=False,
            )
            output_axis = fbx.FbxAxisSystem.MayaYUp
            original_axis_description = None
            character_value = None
            skin_description = "SMPL mesh, skeleton, bind pose, and model skin weights"
        else:
            raise ValueError(f"Unsupported FBX SDK mode: {job['mode']!r}.")
        output = Path(job["output_path"])
        output_axis.ConvertScene(scene)
        _save_scene(manager, scene, output)
        manifest = {
            "schema_version": 1,
            "backend": "fbxsdk",
            "mode": job["mode"],
            "output_path": str(output),
            "source_model_path": job["source_model_path"],
            "model_type": job["model_type"],
            "gender": job["gender"],
            "frames": int(job["frames"]),
            "fps": float(job["fps"]),
            "armature_name": armature_name,
            "mesh_names": [node.GetName() for node in mesh_nodes],
            "character_fbx": character_value,
            "bone_map": bone_map,
            "root_motion_scale": root_scale,
            "retarget_diagnostics": diagnostics,
            "rest_pose_alignment": "target-rest-basis global rotation transfer with arm-chain alignment",
            "motion_source": job.get("source_metadata", {}),
            "coordinates": {
                "input": "SMPL Y-up, +Z forward",
                "retarget_internal": "FBX SDK Maya Z-up, right-handed",
                "target_original_axis": original_axis_description,
                "output_axis": _axis_description(output_axis),
                "fbx_export": (
                    "target character original FBX axis system"
                    if job["mode"] == "character_retarget"
                    else "FBX SDK Maya Y-up, right-handed"
                ),
            },
            "skin": skin_description,
        }
        Path(job["manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        manager.Destroy()


if __name__ == "__main__":
    try:
        _main()
    except Exception as error:
        print(f"FBX SDK export failed: {error}", file=sys.stderr)
        raise
