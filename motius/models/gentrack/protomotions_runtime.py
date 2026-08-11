#!/usr/bin/env python3
"""G1 Robot RL Tracker Export: motion_135 NPZ → G1 robot body poses JSON.

End-to-end pipeline:
  1. Retarget motion_135 NPZ → ProtoMotions .motion file (via pipeline_motion_to_robot.py)
  2. Run G1 ONNX policy in MuJoCo simulation
  3. Export per-frame body positions (xpos) and quaternions (xquat) as JSON
     for Three.js visualization

Output JSON format:
    {
        "type": "robot_frames",
        "robot": "g1",
        "fps": 50,
        "num_frames": N,
        "bodies": [
            {"name": "pelvis", "meshes": ["pelvis.stl", "pelvis_contour_link.stl"]},
            {"name": "left_hip_pitch_link", "meshes": ["left_hip_pitch_link.stl"]},
            ...
        ],
        "frames": [
            {
                "body_pos": [[x,y,z], ...],   # per-body world position (num_bodies x 3)
                "body_quat": [[w,x,y,z], ...] # per-body world quaternion wxyz (num_bodies x 4)
            },
            ...
        ]
    }

Usage:
    # Single file
    python3 tools/gentrack/run_protomotions_reward.py \
        --input outputs/evaluation/gentrack/candidates/example.npz \
        --output-dir outputs/evaluation/gentrack/protomotions/

    # Batch (all NPZ in a directory)
    python3 tools/gentrack/run_protomotions_reward.py \
        --input-dir outputs/evaluation/gentrack/candidates/ \
        --output-dir outputs/evaluation/gentrack/protomotions/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from math import ceil, cos, sin

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from .tracker_paths import (
        PROTOMOTIONS_G1_MJCF,
        PROTOMOTIONS_G1_ONNX,
        PROTOMOTIONS_G1_URDF,
        PROTOMOTIONS_ROOT,
    )
except ImportError:
    from motius.models.gentrack.tracker_paths import (
        PROTOMOTIONS_G1_MJCF,
        PROTOMOTIONS_G1_ONNX,
        PROTOMOTIONS_G1_URDF,
        PROTOMOTIONS_ROOT,
    )

# Add ProtoMotions to path for deployment utilities
_PROTO_ROOT = str(PROTOMOTIONS_ROOT)
if _PROTO_ROOT not in sys.path:
    sys.path.insert(0, _PROTO_ROOT)

from deployment.state_utils import (
    mujoco_wxyz_to_xyzw,
    compute_anchor_rot_np,
    compute_yaw_offset_np,
    apply_heading_offset_np,
)
from deployment.motion_utils import MotionPlayer

# Default paths
DEFAULT_ONNX = PROTOMOTIONS_G1_ONNX
DEFAULT_MJCF = PROTOMOTIONS_G1_MJCF
DEFAULT_SMPL_MODEL = PROJECT_ROOT / "checkpoints" / "smpl_models"
DEFAULT_URDF = PROTOMOTIONS_G1_URDF

# Retargeting scripts
PIPELINE_SCRIPT = SCRIPT_DIR / "pipeline_motion_to_robot.py"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Heavy-asset caches
# ---------------------------------------------------------------------------
# When this module is used as the per-candidate scorer inside the PhysFlow
# online loop, ``simulate_and_export`` is called many times per training step
# (best-of-N candidates x judges). Re-creating the ONNX session and re-parsing
# the MJCF on every call dominated step time (~11-13s/step, with the MuJoCo
# rollout itself only ~0.6s). These process-level caches load each heavy asset
# once and reuse it; only the lightweight per-rollout ``MjData`` is recreated.
_ONNX_SESSION_CACHE: dict = {}
_MJMODEL_CACHE: dict = {}
_ONNX_SESSION_CACHE_LOCK = threading.Lock()
_MJMODEL_CACHE_LOCK = threading.Lock()


def _get_onnx_session(onnx_path: str):
    sess = _ONNX_SESSION_CACHE.get(onnx_path)
    if sess is None:
        with _ONNX_SESSION_CACHE_LOCK:
            sess = _ONNX_SESSION_CACHE.get(onnx_path)
            if sess is None:
                log.info(f"Loading ONNX (cache miss): {onnx_path}")
                session_options = ort.SessionOptions()
                intra_threads = int(os.environ.get("PHYSFLOW_ORT_INTRA_OP_THREADS", "0"))
                inter_threads = int(os.environ.get("PHYSFLOW_ORT_INTER_OP_THREADS", "0"))
                if intra_threads > 0:
                    session_options.intra_op_num_threads = intra_threads
                if inter_threads > 0:
                    session_options.inter_op_num_threads = inter_threads
                sess = ort.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                _ONNX_SESSION_CACHE[onnx_path] = sess
    return sess


def _get_cached_mujoco_model(mjcf_path: str, stiffness, damping, physics_dt: float):
    """Return a fully-configured (cached) MjModel; create fresh MjData per call.

    The model is configured deterministically from (mjcf, stiffness, damping,
    physics_dt) and is never mutated during a rollout (the sim only writes to
    MjData), so it is safe to share across candidate rollouts.
    """
    key = (str(mjcf_path), tuple(stiffness), tuple(damping), float(physics_dt))
    model = _MJMODEL_CACHE.get(key)
    if model is None:
        with _MJMODEL_CACHE_LOCK:
            model = _MJMODEL_CACHE.get(key)
            if model is None:
                model, _ = load_mujoco_model(mjcf_path, stiffness, damping, physics_dt)
                _MJMODEL_CACHE[key] = model
    return model


# ---------------------------------------------------------------------------
# MJCF parsing: body→mesh mapping
# ---------------------------------------------------------------------------

def parse_body_mesh_mapping(mjcf_path: pathlib.Path) -> list:
    """Parse MJCF XML to extract body name -> raw STL visual mapping.

    The browser viewer loads the original STL files directly. MuJoCo's compiled
    ``geom_pos``/``geom_quat`` are for compiler-normalized mesh vertices and
    must not be applied to raw STL vertices, otherwise visual meshes are offset
    and rotated twice. For web visualization we therefore preserve the XML geom
    transform only.
    """
    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()

    # Build mesh_name → file mapping from <asset>
    mesh_name_to_file = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh_elem in asset.findall("mesh"):
            name = mesh_elem.get("name", "")
            filename = mesh_elem.get("file", "")
            if name and filename:
                mesh_name_to_file[name] = filename

    def parse_float_list(value: str | None, default: list[float]) -> list[float]:
        if value is None:
            return list(default)
        return [float(x) for x in value.split()]

    def quat_from_axis_angle(axis_angle: str | None) -> list[float] | None:
        if axis_angle is None:
            return None
        values = [float(x) for x in axis_angle.split()]
        if len(values) != 4:
            return None
        axis = np.asarray(values[:3], dtype=np.float64)
        angle = values[3]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return [1.0, 0.0, 0.0, 0.0]
        axis = axis / norm
        half = 0.5 * angle
        xyz = axis * sin(half)
        return [float(cos(half)), float(xyz[0]), float(xyz[1]), float(xyz[2])]

    def quat_from_euler(euler: str | None) -> list[float] | None:
        if euler is None:
            return None
        values = [float(x) for x in euler.split()]
        if len(values) != 3:
            return None
        cx, cy, cz = (cos(v * 0.5) for v in values)
        sx, sy, sz = (sin(v * 0.5) for v in values)
        return [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ]

    def geom_mesh_record(geom: ET.Element) -> dict | None:
        if geom.get("type") not in (None, "mesh") and not geom.get("mesh"):
            return None
        mesh_name = geom.get("mesh", "")
        if mesh_name not in mesh_name_to_file:
            return None
        quat = (
            parse_float_list(geom.get("quat"), [1.0, 0.0, 0.0, 0.0])
            if geom.get("quat") is not None
            else quat_from_axis_angle(geom.get("axisangle"))
            or quat_from_euler(geom.get("euler"))
            or [1.0, 0.0, 0.0, 0.0]
        )
        return {
            "file": mesh_name_to_file[mesh_name],
            "pos": parse_float_list(geom.get("pos"), [0.0, 0.0, 0.0]),
            "quat": quat,
        }

    bodies = []
    def walk_body(elem: ET.Element) -> None:
        body_name = elem.get("name", "unnamed")
        mesh_records = []
        seen_files = set()
        for geom in elem.findall("geom"):
            mesh_record = geom_mesh_record(geom)
            if mesh_record is None:
                continue
            stl_file = mesh_record["file"]
            if stl_file in seen_files:
                continue
            seen_files.add(stl_file)
            mesh_records.append(mesh_record)
        bodies.append({"name": body_name, "meshes": mesh_records})
        for child in elem.findall("body"):
            walk_body(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            walk_body(top_body)
    return bodies


# ---------------------------------------------------------------------------
# MuJoCo model loading (from test_tracker_mujoco.py)
# ---------------------------------------------------------------------------

def _patch_mjcf_xml(xml_path: pathlib.Path) -> str:
    """Patch MJCF for standalone MuJoCo use (strip sensors, add ground)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or "ground" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("rgba", "0.7 0.7 0.7 1")

    return ET.tostring(root, encoding="unicode")


def load_mujoco_model(mjcf_path: str, stiffness: list, damping: list, physics_dt: float):
    """Load MuJoCo model configured for G1 policy deployment."""
    mjcf_file = pathlib.Path(mjcf_path)
    if not mjcf_file.is_absolute():
        candidates = [
            PROTOMOTIONS_ROOT / mjcf_path,
            PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / mjcf_path,
        ]
        for c in candidates:
            if c.exists():
                mjcf_file = c
                break

    if not mjcf_file.exists():
        raise FileNotFoundError(f"Cannot find MJCF: {mjcf_path}")

    log.info(f"Loading MuJoCo model: {mjcf_file}")
    patched_xml = _patch_mjcf_xml(mjcf_file)

    asset_dir = str(mjcf_file.parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)

    # Set physics timestep
    model.opt.timestep = physics_dt
    log.info(f"  Physics timestep: {physics_dt}s ({1.0/physics_dt:.0f}Hz)")

    # Zero passive forces
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Configure implicit PD actuators
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator mismatch: nu={num_actuators}, stiff={len(stiffness)}, damp={len(damping)}"
    )
    for i in range(num_actuators):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0

    log.info(f"  {num_actuators} actuators, {model.nbody} bodies, {model.nq} qpos, {model.nv} qvel")
    return model, data


# ---------------------------------------------------------------------------
# Robot state reading (from test_tracker_mujoco.py)
# ---------------------------------------------------------------------------

def read_robot_state(data, anchor_body_index: int, root_body_index: int = 0):
    """Read robot state from MuJoCo buffers."""
    body_pos = data.xpos[1:].copy()
    body_rot_wxyz = data.xquat[1:].copy()
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)

    # For root body, use canonical free-joint quaternion
    root_rot_wxyz = data.qpos[3:7].copy()
    body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz)

    root_local_ang_vel = data.qvel[3:6].copy().astype(np.float32)

    return {
        "dof_pos":            data.qpos[7:].copy().astype(np.float32),
        "dof_vel":            data.qvel[6:].copy().astype(np.float32),
        "body_pos":           body_pos.astype(np.float32),
        "body_rot":           body_rot.astype(np.float32),
        "root_local_ang_vel": root_local_ang_vel,
    }


def set_initial_pose(model, data, motion_player):
    """Initialize robot at first frame of motion."""
    frame0 = motion_player.get_state_at_frame(0)
    root_pos = frame0["body_pos"][0]
    root_quat = frame0["body_rot"][0]  # xyzw

    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat[[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:] = frame0["dof_pos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def build_onnx_inputs(
    robot_state,
    ref_state,
    future_refs,
    onnx_name_to_key,
    anchor_body_index,
    num_dofs,
    prev_actions=None,
):
    """Assemble ONNX input dict from robot state + motion futures."""
    dof_pos = robot_state["dof_pos"]
    dof_vel = robot_state["dof_vel"]
    body_pos = robot_state["body_pos"]
    body_rot = robot_state["body_rot"]
    root_local_ang_vel = robot_state["root_local_ang_vel"]

    anchor_pos = body_pos[anchor_body_index]
    anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)

    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]
    future_anchor_pos = future_refs["body_pos"][:, anchor_body_index, :]
    future_anchor_vel = future_refs["body_vel"][:, anchor_body_index, :]
    future_anchor_ang_vel = future_refs["body_ang_vel"][:, anchor_body_index, :]
    ref_anchor_pos = ref_state["body_pos"][anchor_body_index]

    key_to_array = {
        "current.dof_pos":             dof_pos[None],
        "current.dof_vel":             dof_vel[None],
        "current.anchor_pos":          anchor_pos[None],
        "current.anchor_rot":          anchor_rot[None],
        "current.root_local_ang_vel":  root_local_ang_vel[None],
        "historical.processed_actions": prev_actions[None, None],
        "mimic.ref_anchor_pos":        ref_anchor_pos[None],
        "mimic.future_anchor_pos":     future_anchor_pos[None],
        "mimic.future_anchor_rot":     future_anchor_rot[None],
        "mimic.future_anchor_vel":     future_anchor_vel[None],
        "mimic.future_anchor_ang_vel": future_anchor_ang_vel[None],
        "mimic.future_pos":            future_refs["body_pos"][None],
        "mimic.future_rot":            future_refs["body_rot"][None],
        "mimic.future_vel":            future_refs["body_vel"][None],
        "mimic.future_ang_vel":        future_refs["body_ang_vel"][None],
        "mimic.future_dof_pos":        future_refs["dof_pos"][None],
        "mimic.future_dof_vel":        future_refs["dof_vel"][None],
    }

    onnx_inputs = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

    return onnx_inputs


def apply_heading_offset_to_positions(offset_quat_xyzw: np.ndarray, body_pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Rotate world positions around an origin by a yaw-only heading offset."""
    x, y, z, w = offset_quat_xyzw
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c = np.cos(yaw)
    s = np.sin(yaw)
    shifted = body_pos - origin
    aligned = shifted.copy()
    aligned[..., 0] = c * shifted[..., 0] - s * shifted[..., 1]
    aligned[..., 1] = s * shifted[..., 0] + c * shifted[..., 1]
    return aligned + origin


def apply_heading_offset_to_vectors(offset_quat_xyzw: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate world velocity vectors by a yaw-only heading offset."""
    x, y, z, w = offset_quat_xyzw
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c = np.cos(yaw)
    s = np.sin(yaw)
    aligned = vectors.copy()
    aligned[..., 0] = c * vectors[..., 0] - s * vectors[..., 1]
    aligned[..., 1] = s * vectors[..., 0] + c * vectors[..., 1]
    return aligned


def _compute_root_tracking_stats(ref_root_positions, sim_root_positions) -> dict:
    """Compare simulated root XY motion against the reference trajectory."""
    if not ref_root_positions or not sim_root_positions:
        return {
            "root_displacement_ref_m": 0.0,
            "root_displacement_track_m": 0.0,
            "root_displacement_error_m": 0.0,
            "root_trajectory_error_mean_m": 0.0,
            "root_trajectory_error_final_m": 0.0,
        }

    ref = np.asarray(ref_root_positions, dtype=np.float64)
    sim = np.asarray(sim_root_positions, dtype=np.float64)
    n = min(len(ref), len(sim))
    ref = ref[:n]
    sim = sim[:n]
    if n < 2:
        return {
            "root_displacement_ref_m": 0.0,
            "root_displacement_track_m": 0.0,
            "root_displacement_error_m": 0.0,
            "root_trajectory_error_mean_m": 0.0,
            "root_trajectory_error_final_m": 0.0,
        }

    ref_xy = ref[:, :2] - ref[0, :2]
    sim_xy = sim[:, :2] - sim[0, :2]
    traj_error = np.linalg.norm(sim_xy - ref_xy, axis=-1)
    ref_disp = ref_xy[-1]
    sim_disp = sim_xy[-1]

    return {
        "root_displacement_ref_m": float(np.linalg.norm(ref_disp)),
        "root_displacement_track_m": float(np.linalg.norm(sim_disp)),
        "root_displacement_error_m": float(np.linalg.norm(sim_disp - ref_disp)),
        "root_trajectory_error_mean_m": float(np.mean(traj_error)),
        "root_trajectory_error_final_m": float(traj_error[-1]),
    }


def is_reference_conditioned_fall(
    sim_root_height: float,
    ref_root_height: float,
    *,
    ref_up_axis_cosine: float = 1.0,
    root_orientation_error_rad: float = 0.0,
    min_ref_root_height_m: float = 0.5,
    min_ref_up_axis_cosine: float = 0.5,
    low_root_threshold_m: float = 0.3,
    root_height_deficit_m: float = 0.25,
    root_orientation_error_threshold_rad: float = 1.0,
) -> bool:
    """Return one-frame evidence for a reference-conditioned collapse.

    The reference must itself be upright. A final Unexpected Fall additionally
    requires this evidence to persist for 0.20 seconds in the rollout loop.
    """
    return (
        float(ref_root_height) >= float(min_ref_root_height_m)
        and float(ref_up_axis_cosine) >= float(min_ref_up_axis_cosine)
        and (
            (
                float(sim_root_height) < float(low_root_threshold_m)
                and float(ref_root_height) - float(sim_root_height)
                > float(root_height_deficit_m)
            )
            or float(root_orientation_error_rad)
            > float(root_orientation_error_threshold_rad)
        )
    )


def _quat_angle_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    dot = float(np.clip(abs(np.dot(a, b)) / denom, 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _up_axis_cosine_xyzw(quat: np.ndarray) -> float:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    x, y, _, _ = quat
    return float(1.0 - 2.0 * (x * x + y * y))


# ---------------------------------------------------------------------------
# Simulation + Export
# ---------------------------------------------------------------------------

def simulate_and_export(
    onnx_path: str,
    motion_file: str,
    output_json_path: str,
    mjcf_path: str,
    body_mesh_mapping: list,
    subsample_factor: int = 1,
    terminate_on_unexpected_fall: bool = True,
    export_frames: bool = True,
    collect_tracking_trajectories: bool = False,
) -> dict:
    """Run G1 ONNX policy simulation and export body poses as JSON.

    Parameters
    ----------
    onnx_path: Path to unified_pipeline.onnx
    motion_file: Path to .motion file (ProtoMotions format)
    output_json_path: Where to save the output JSON
    mjcf_path: Path to G1 MJCF XML
    body_mesh_mapping: List of {name, meshes} from parse_body_mesh_mapping
    subsample_factor: Export every Nth frame (1=all frames at 50fps)
    export_frames: Write the visualization JSON. Reward-only rollouts disable
        this while retaining exactly the same simulation and tracking stats.
    collect_tracking_trajectories: Return aligned reference/execution body
        trajectories for public tracking-kernel reward computation.

    Returns
    -------
    stats dict with simulation metrics
    """
    yaml_path = onnx_path.replace(".onnx", ".yaml")

    # Load YAML metadata
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    robot_meta = meta["robot"]
    timing = meta["timing"]
    motion_meta = meta["motion"]
    control = meta["control"]
    runtime = meta["_runtime"]

    anchor_body_index = robot_meta["anchor_body_index"]
    root_body_index = robot_meta["root_body_index"]
    num_bodies = robot_meta["num_bodies"]
    num_dofs = robot_meta["num_dofs"]
    control_dt = timing["control_dt"]
    decimation = timing["decimation"]
    physics_dt = timing["physics_dt"]
    future_step_indices = motion_meta["future_step_indices"]
    stiffness = control["stiffness"]
    damping = control["damping"]
    pd_target_max_accel = control.get("pd_target_max_accel")
    action_ema_alpha = control.get("action_ema_alpha", 1.0)
    onnx_name_to_key = runtime["onnx_name_to_in_key"]

    session = _get_onnx_session(onnx_path)
    actual_out_names = [out.name for out in session.get_outputs()]

    # Load motion
    log.info(f"Loading motion: {motion_file}")
    player = MotionPlayer(motion_file, control_dt=control_dt)
    log.info(f"  Motion: {player.total_frames} frames @ {1.0/control_dt:.0f}Hz "
             f"(duration={player.total_frames * control_dt:.2f}s)")

    # Cached, fully-configured MuJoCo model; fresh per-rollout state buffer.
    model = _get_cached_mujoco_model(mjcf_path, stiffness, damping, physics_dt)
    data = mujoco.MjData(model)

    # Verify body count matches
    mj_num_bodies = model.nbody - 1  # exclude world body
    if mj_num_bodies != num_bodies:
        log.warning(f"Body count mismatch: MuJoCo={mj_num_bodies}, YAML={num_bodies}")

    # Initialize
    set_initial_pose(model, data, player)

    # EMA state
    use_ema = action_ema_alpha < 1.0
    ema_prev_targets = None
    prev_pd = None
    prev_prev_pd = None
    prev_actions = None
    heading_offset = None

    # Collect frames
    frames_data = []
    ref_root_positions = []
    sim_root_positions = []
    ref_body_positions = []
    sim_body_positions = []
    ref_body_quaternions = []
    sim_body_quaternions = []
    total_steps = 0
    max_pd_diff = 0.0
    fall_detected = False
    absolute_low_root_detected = False
    max_root_height_deficit_m = 0.0
    fall_evidence_frames = 0
    required_fall_evidence_frames = max(1, int(ceil(0.20 / control_dt)))

    t_start = time.perf_counter()
    initial_ref_pos = player.get_state_at_frame(0)["body_pos"][anchor_body_index]

    for frame_idx in range(player.total_frames):
        # Read robot state
        robot_state = read_robot_state(data, anchor_body_index, root_body_index)

        # Heading offset on first step
        if heading_offset is None:
            robot_anchor_rot = robot_state["body_rot"][anchor_body_index]
            motion_anchor_rot = player.get_state_at_frame(0)["body_rot"][anchor_body_index]
            heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)

        # Get future motion references
        ref_state = player.get_state_at_frame(frame_idx)
        ref_state = dict(ref_state)
        ref_state["body_pos"] = apply_heading_offset_to_positions(
            heading_offset,
            ref_state["body_pos"],
            initial_ref_pos,
        )
        ref_state["body_vel"] = apply_heading_offset_to_vectors(heading_offset, ref_state["body_vel"])
        ref_state["body_ang_vel"] = apply_heading_offset_to_vectors(heading_offset, ref_state["body_ang_vel"])
        future_refs = player.get_future_references(frame_idx, future_step_indices)
        future_refs["body_pos"] = apply_heading_offset_to_positions(
            heading_offset,
            future_refs["body_pos"],
            initial_ref_pos,
        )
        future_refs["body_vel"] = apply_heading_offset_to_vectors(heading_offset, future_refs["body_vel"])
        future_refs["body_ang_vel"] = apply_heading_offset_to_vectors(heading_offset, future_refs["body_ang_vel"])
        future_refs["body_rot"] = apply_heading_offset_np(heading_offset, future_refs["body_rot"])

        # Build ONNX inputs
        onnx_inputs = build_onnx_inputs(
            robot_state=robot_state,
            ref_state=ref_state,
            future_refs=future_refs,
            onnx_name_to_key=onnx_name_to_key,
            anchor_body_index=anchor_body_index,
            num_dofs=num_dofs,
            prev_actions=prev_actions,
        )

        # ONNX inference
        ort_out = session.run(actual_out_names, onnx_inputs)
        pd_targets = ort_out[1].squeeze().copy()  # joint_pos_targets

        # PD target acceleration clamp
        if pd_target_max_accel is not None and prev_pd is not None and prev_prev_pd is not None:
            delta = pd_targets - prev_pd
            prev_delta = prev_pd - prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
            pd_targets = prev_pd + prev_delta + clamped_accel

        prev_prev_pd = prev_pd
        prev_pd = pd_targets.copy()

        # EMA filter
        if use_ema:
            if ema_prev_targets is None:
                ema_prev_targets = pd_targets.copy()
            pd_targets = action_ema_alpha * pd_targets + (1.0 - action_ema_alpha) * ema_prev_targets
            ema_prev_targets = pd_targets.copy()

        prev_actions = pd_targets.copy()

        # Write PD targets and step physics
        data.ctrl[:] = pd_targets
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # A low root can be the requested motion (fall/get-up, floor sit/lie).
        # Only treat it as a failure when it is low relative to this reference
        # frame. Formal reward evaluation disables early termination so a
        # tracker can recover later in the clip.
        root_height = float(data.qpos[2])
        ref_root_height = float(ref_state["body_pos"][root_body_index, 2])
        root_height_deficit = ref_root_height - root_height
        ref_root_quat = ref_state["body_rot"][root_body_index]
        sim_root_quat = mujoco_wxyz_to_xyzw(data.qpos[3:7].copy())
        ref_up_axis_cosine = _up_axis_cosine_xyzw(ref_root_quat)
        root_orientation_error = _quat_angle_xyzw(sim_root_quat, ref_root_quat)
        max_root_height_deficit_m = max(max_root_height_deficit_m, root_height_deficit)
        absolute_low_root_detected = absolute_low_root_detected or root_height < 0.3
        fall_evidence = is_reference_conditioned_fall(
            root_height,
            ref_root_height,
            ref_up_axis_cosine=ref_up_axis_cosine,
            root_orientation_error_rad=root_orientation_error,
        )
        fall_evidence_frames = fall_evidence_frames + 1 if fall_evidence else 0
        if fall_evidence_frames == required_fall_evidence_frames:
            log.warning(
                "  Unexpected fall at frame %d, root_h=%.3f ref_root_h=%.3f "
                "deficit=%.3f root_ori_err=%.3f persisted=%d",
                frame_idx,
                root_height,
                ref_root_height,
                root_height_deficit,
                root_orientation_error,
                fall_evidence_frames,
            )
            fall_detected = True
            if terminate_on_unexpected_fall:
                break

        # Export frame (subsample)
        ref_root_positions.append(ref_state["body_pos"][0].copy())
        sim_root_positions.append(data.xpos[1].copy())
        if collect_tracking_trajectories:
            ref_body_positions.append(ref_state["body_pos"].copy())
            sim_body_positions.append(data.xpos[1:num_bodies + 1].copy())
            ref_body_quaternions.append(ref_state["body_rot"].copy())
            sim_body_quaternions.append(
                mujoco_wxyz_to_xyzw(data.xquat[1:num_bodies + 1].copy())
            )
        if export_frames and frame_idx % subsample_factor == 0:
            # body_pos: data.xpos[1:] (skip world body), shape [num_bodies, 3]
            body_pos = data.xpos[1:num_bodies+1].copy()
            # body_quat: data.xquat[1:] in wxyz format, shape [num_bodies, 4]
            body_quat = data.xquat[1:num_bodies+1].copy()

            frames_data.append({
                "body_pos": body_pos.tolist(),
                "body_quat": body_quat.tolist(),  # wxyz for Three.js
            })

        # Track ref error
        ref_dof_pos = ref_state["dof_pos"]
        diff = float(np.abs(data.qpos[7:] - ref_dof_pos).max())
        if diff > max_pd_diff:
            max_pd_diff = diff

        total_steps += 1

        if frame_idx % 200 == 0:
            log.info(f"  frame={frame_idx:4d}/{player.total_frames}  "
                     f"root_h={root_height:.3f}  max_err={max_pd_diff:.4f}")

    elapsed = time.perf_counter() - t_start
    log.info(f"  Simulation done: {total_steps} steps in {elapsed:.1f}s "
             f"({total_steps/max(elapsed,1e-6):.0f} steps/s)")

    # Determine output fps
    output_fps = 50.0 / subsample_factor  # control at 50Hz

    if export_frames:
        output_data = {
            "type": "robot_frames",
            "robot": "g1",
            "fps": output_fps,
            "num_frames": len(frames_data),
            "num_bodies": num_bodies,
            "bodies": body_mesh_mapping[:num_bodies],
            "frames": frames_data,
        }
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output_data, f)

        file_size_mb = os.path.getsize(output_json_path) / 1e6
        log.info(
            f"  Saved: {output_json_path} "
            f"({file_size_mb:.1f} MB, {len(frames_data)} frames)"
        )

    root_stats = _compute_root_tracking_stats(ref_root_positions, sim_root_positions)

    stats = {
        "total_steps": total_steps,
        "total_frames_exported": len(frames_data),
        "fall_detected": fall_detected,
        "fall_protocol": "reference_conditioned_v012_persistent",
        "absolute_low_root_detected": absolute_low_root_detected,
        "max_root_height_deficit_m": float(max_root_height_deficit_m),
        "terminate_on_unexpected_fall": bool(terminate_on_unexpected_fall),
        "fall_persistence_frames": int(required_fall_evidence_frames),
        "max_joint_error_rad": max_pd_diff,
        "root_height_final": float(data.qpos[2]),
        "elapsed_seconds": elapsed,
        "output_fps": output_fps,
        **root_stats,
    }
    if collect_tracking_trajectories:
        stats.update({
            "tracking_fps": float(1.0 / control_dt),
            "reference_body_pos": np.asarray(
                ref_body_positions, dtype=np.float32
            ),
            "execution_body_pos": np.asarray(
                sim_body_positions, dtype=np.float32
            ),
            "reference_body_quat": np.asarray(
                ref_body_quaternions, dtype=np.float32
            ),
            "execution_body_quat": np.asarray(
                sim_body_quaternions, dtype=np.float32
            ),
        })
    return stats


# ---------------------------------------------------------------------------
# Retargeting step
# ---------------------------------------------------------------------------

def retarget_npz_to_motion(
    npz_path: pathlib.Path,
    output_dir: pathlib.Path,
    smpl_model_path: str = None,
    urdf_path: str = None,
    fps: int = 30,
    pyroki_max_iterations: int = 800,
    subsample_factor: int = 1,
    target_raw_frames: int = 450,
) -> pathlib.Path:
    """Run retargeting pipeline: motion_135 NPZ → .motion file.

    Uses pipeline_motion_to_robot.py as subprocess.
    Returns path to generated .motion file.
    """
    if smpl_model_path is None:
        smpl_model_path = str(DEFAULT_SMPL_MODEL)
    if urdf_path is None:
        urdf_path = str(DEFAULT_URDF)

    motion_output_dir = output_dir / "motion_files"
    motion_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if .motion file already exists
    stem = npz_path.stem
    existing_motion = motion_output_dir / f"{stem}.motion"
    if existing_motion.exists():
        log.info(f"  Retargeted .motion already exists: {existing_motion}")
        return existing_motion

    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--input", str(npz_path),
        "--output", str(motion_output_dir),
        "--smpl-model-path", smpl_model_path,
        "--urdf", urdf_path,
        "--fps", str(fps),
        "--keep-intermediates",
        "--pyroki-max-iterations", str(pyroki_max_iterations),
        "--subsample-factor", str(subsample_factor),
        "--target-raw-frames", str(target_raw_frames),
        "--output-fps", str(max(1, int(round(fps / max(subsample_factor, 1))))),
    ]

    log.info(f"  Running retargeting: {npz_path.name}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error(f"  Retargeting failed for {npz_path.name}:")
        log.error(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Retargeting failed: {npz_path.name}")

    # Find the generated .motion file
    motion_files = list(motion_output_dir.glob(f"{stem}*.motion"))
    if not motion_files:
        # Also check in subdirectories
        motion_files = list(motion_output_dir.glob(f"**/{stem}*.motion"))

    if not motion_files:
        raise FileNotFoundError(f"No .motion file found after retargeting {npz_path.name}")

    return motion_files[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G1 Robot RL Tracker Export: NPZ → robot body poses JSON"
    )
    parser.add_argument("--input", type=str, help="Single input motion_135 NPZ")
    parser.add_argument("--input-dir", type=str, help="Directory of NPZ files to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for robot JSON files")
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX),
                        help="Path to G1 ONNX model")
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF),
                        help="Path to G1 MJCF XML")
    parser.add_argument("--smpl-model-path", type=str, default=str(DEFAULT_SMPL_MODEL),
                        help="Path to SMPL model directory")
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF),
                        help="Path to G1 URDF for retargeting")
    parser.add_argument("--fps", type=int, default=30,
                        help="Input motion FPS")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Export every Nth control frame (default 2 → 25fps output)")
    parser.add_argument("--skip-retarget", action="store_true",
                        help="Skip retargeting, assume .motion files exist")
    parser.add_argument("--motion-dir", type=str,
                        help="Directory containing pre-generated .motion files")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect NPZ files
    npz_files = []
    if args.input:
        npz_files = [pathlib.Path(args.input)]
    elif args.input_dir:
        npz_files = sorted(pathlib.Path(args.input_dir).glob("*.npz"))
    else:
        parser.error("Must specify --input or --input-dir")

    if not npz_files:
        log.error("No NPZ files found!")
        sys.exit(1)

    log.info(f"Processing {len(npz_files)} NPZ files")
    log.info(f"Output: {output_dir}")

    # Parse body-mesh mapping from MJCF
    mjcf_path = pathlib.Path(args.mjcf)
    body_mesh_mapping = parse_body_mesh_mapping(mjcf_path)
    log.info(f"Parsed {len(body_mesh_mapping)} bodies from MJCF")

    # Verify ONNX exists
    onnx_path = args.onnx
    if not pathlib.Path(onnx_path).exists():
        log.error(f"ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Process each NPZ
    all_stats = []
    for i, npz_path in enumerate(npz_files):
        log.info(f"\n{'='*60}")
        log.info(f"  [{i+1}/{len(npz_files)}] {npz_path.name}")
        log.info(f"{'='*60}")

        stem = npz_path.stem
        output_json = output_dir / f"{stem}.json"

        # Skip if already exists
        if output_json.exists():
            log.info(f"  Output already exists, skipping: {output_json}")
            all_stats.append({"name": stem, "skipped": True})
            continue

        try:
            # Step 1: Retarget NPZ → .motion
            if args.skip_retarget and args.motion_dir:
                motion_dir = pathlib.Path(args.motion_dir)
                motion_candidates = list(motion_dir.glob(f"{stem}*.motion"))
                if not motion_candidates:
                    log.error(f"  No .motion file found for {stem} in {motion_dir}")
                    all_stats.append({"name": stem, "error": "no_motion_file"})
                    continue
                motion_path = motion_candidates[0]
            else:
                motion_path = retarget_npz_to_motion(
                    npz_path, output_dir,
                    smpl_model_path=args.smpl_model_path,
                    urdf_path=args.urdf,
                    fps=args.fps,
                )

            log.info(f"  Motion file: {motion_path}")

            # Step 2: Simulate + Export
            stats = simulate_and_export(
                onnx_path=onnx_path,
                motion_file=str(motion_path),
                output_json_path=str(output_json),
                mjcf_path=args.mjcf,
                body_mesh_mapping=body_mesh_mapping,
                subsample_factor=args.subsample,
            )
            stats["name"] = stem
            all_stats.append(stats)

        except Exception as e:
            log.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({"name": stem, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"  G1 RL Tracker Export Complete")
    print(f"{'='*60}")
    successes = [s for s in all_stats if "error" not in s and not s.get("skipped")]
    failures = [s for s in all_stats if "error" in s]
    skipped = [s for s in all_stats if s.get("skipped")]
    print(f"  Total: {len(npz_files)}")
    print(f"  Success: {len(successes)}")
    print(f"  Skipped (already exist): {len(skipped)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print(f"\n  Failures:")
        for f in failures:
            print(f"    - {f['name']}: {f['error']}")

    if successes:
        avg_steps = np.mean([s["total_steps"] for s in successes])
        avg_error = np.mean([s["max_joint_error_rad"] for s in successes])
        falls = sum(1 for s in successes if s.get("fall_detected"))
        print(f"\n  Stats (successful):")
        print(f"    Avg steps: {avg_steps:.0f}")
        print(f"    Avg max joint error: {avg_error:.4f} rad")
        print(f"    Falls: {falls}/{len(successes)}")

    # Save summary
    summary_path = output_dir / "_export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
