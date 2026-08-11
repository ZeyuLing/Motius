#!/usr/bin/env python3
"""Build engine-isolated tracking result packs and all-case G1 web assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/leaderboards"
DEFAULT_GENTRACK_ROOT = Path(
    "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
)
DEFAULT_MUJOCO_ROOT = (
    ROOT / "outputs/motion_tracking/benchmarks/mujoco_lafan1_g1_v1"
)
ASSET_DATASET = (
    "https://huggingface.co/datasets/ZeyuLing/"
    "Motius-Leaderboard-Cases/resolve/main/motion-tracking-g1/v1"
)
MESH_DATASET = (
    "https://huggingface.co/datasets/ZeyuLing/"
    "Motius-Leaderboard-Cases/resolve/main/t2m-unitree-g1/meshes"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gentrack-root", type=Path, default=DEFAULT_GENTRACK_ROOT)
    parser.add_argument("--mujoco-root", type=Path, default=DEFAULT_MUJOCO_ROOT)
    parser.add_argument(
        "--asset-output",
        type=Path,
        default=ROOT / "outputs/publication/motion_tracking_g1_v1",
    )
    parser.add_argument("--skip-assets", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_recorded_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    text = candidate.as_posix()
    marker = "/outputs/"
    if marker in text:
        remapped = repo_root / "outputs" / text.split(marker, 1)[1]
        if remapped.exists():
            return remapped
    raise FileNotFoundError(f"Recorded GenTrack path is unavailable: {path}")


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["case_id"])] = row
    return rows


def _safe_name(case_id: str, index: int) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", case_id).strip("_")[:44]
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]
    return f"{index:04d}_{prefix}_{digest}.bin"


def _qpos(path: Path, key: str = "qpos") -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive[key], dtype=np.float64)
        if "fps" in archive:
            fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        elif "frequency" in archive:
            fps = float(np.asarray(archive["frequency"]).reshape(-1)[0])
        else:
            raise ValueError(f"{path} has no fps/frequency field")
    if values.ndim != 2 or values.shape[1] != 36:
        raise ValueError(f"{path}: expected [T, 36] G1 qpos, got {values.shape}")
    return values, fps


class G1TransformExporter:
    """Convert qpos to compact body transforms using Motius' packaged MJCF."""

    def __init__(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mujoco = mujoco
        scene = (
            ROOT
            / "motius/simulators/mujoco/assets/unitree_g1/"
            "scene_mjx_flat_terrain.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.body_ids = np.arange(1, self.model.nbody, dtype=np.int32)
        model_xml = scene.with_name("g1_mjx.xml")
        model_tree = ET.parse(model_xml)
        mesh_assets = {
            mesh.attrib["name"]: mesh.attrib
            for mesh in model_tree.findall(".//asset/mesh")
        }
        self.body_meshes: dict[str, list[dict[str, Any]]] = {}
        for body in model_tree.findall(".//worldbody//body"):
            body_name = body.attrib["name"]
            meshes = []
            seen = set()
            for geom in body.findall("geom"):
                mesh_name = geom.attrib.get("mesh")
                if not mesh_name:
                    continue
                asset = mesh_assets.get(mesh_name)
                if asset is None:
                    raise ValueError(f"No MJCF asset mapping for mesh {mesh_name!r}")
                pos = tuple(float(value) for value in geom.attrib.get("pos", "0 0 0").split())
                quat = tuple(float(value) for value in geom.attrib.get("quat", "1 0 0 0").split())
                scale = tuple(float(value) for value in asset.get("scale", "1 1 1").split())
                key = (mesh_name, pos, quat, scale)
                if key in seen:
                    continue
                seen.add(key)
                meshes.append(
                    {
                        "file": Path(asset["file"]).name.lower(),
                        "pos": list(pos),
                        "quat": list(quat),
                        "scale": list(scale),
                    }
                )
            self.body_meshes[body_name] = meshes

    def robot_metadata(self) -> dict[str, Any]:
        mujoco = self.mujoco
        bodies = []
        for body_id in self.body_ids:
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
            )
            bodies.append(
                {"name": body_name, "meshes": self.body_meshes.get(body_name, [])}
            )
        return {
            "schema_version": 1,
            "format": "float32_le_body_pos_xyz_quat_wxyz",
            "stride": 7,
            "body_count": len(bodies),
            "bodies": bodies,
        }

    def export(self, qpos: np.ndarray, output: Path) -> None:
        frames = np.empty(
            (len(qpos), len(self.body_ids), 7), dtype=np.float32
        )
        for frame_index, pose in enumerate(qpos):
            self.data.qpos[:] = pose
            self.data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            frames[frame_index, :, :3] = self.data.xpos[self.body_ids]
            frames[frame_index, :, 3:] = self.data.xquat[self.body_ids]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(frames.astype("<f4", copy=False).tobytes())


def _metric_definitions(engine: str) -> list[dict[str, str]]:
    common = [
        {"key": "success_rate", "label": "Success", "unit": "%", "direction": "higher"},
        {"key": "completion_rate", "label": "Completion", "unit": "%", "direction": "higher"},
        {"key": "local_mpjpe_mm", "label": "Local MPJPE", "unit": "mm", "direction": "lower"},
    ]
    if engine == "mujoco":
        return common + [
            {"key": "joint_mae_rad", "label": "Joint MAE", "unit": "rad", "direction": "lower"},
            {"key": "root_drift_mm", "label": "Root drift", "unit": "mm", "direction": "lower"},
            {"key": "foot_slip_m_s", "label": "Foot slip", "unit": "m/s", "direction": "lower"},
            {"key": "mechanical_power_w", "label": "Mechanical power", "unit": "W", "direction": "lower"},
        ]
    return common + [
        {"key": "global_mpjpe_mm", "label": "Global MPJPE", "unit": "mm", "direction": "lower"},
        {"key": "joint_mae_rad", "label": "Joint MAE", "unit": "rad", "direction": "lower"},
        {"key": "mpjve_m_s", "label": "MPJVE", "unit": "m/s", "direction": "lower"},
        {"key": "root_velocity_error_m_s", "label": "Root velocity error", "unit": "m/s", "direction": "lower"},
        {"key": "unexpected_fall_rate", "label": "Unexpected fall", "unit": "%", "direction": "lower"},
    ]


def _mujoco_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "success_rate": float(raw["success_rate"]),
        "completion_rate": float(raw["survival_rate"]),
        "local_mpjpe_mm": float(raw["local_body_mpjpe_m"]) * 1000.0,
        "joint_mae_rad": float(raw["joint_position_mae_rad"]),
        "root_drift_mm": float(raw["root_position_error_m"]) * 1000.0,
        "foot_slip_m_s": float(raw.get("foot_slip_m_s", 0.0)),
        "mechanical_power_w": float(raw.get("mechanical_power_w", 0.0)),
    }


def _isaac_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "success_rate": float(raw["success_rate_unified"]),
        "completion_rate": float(raw["completion"]),
        "local_mpjpe_mm": float(raw["mpjpe_mm"]),
        "global_mpjpe_mm": float(raw["eg_mpjpe_mm"]),
        "joint_mae_rad": float(raw["e_joint_rad"]),
        "mpjve_m_s": float(raw["mpjve_mps"]),
        "root_velocity_error_m_s": float(raw["root_vel_err_mps"]),
        "unexpected_fall_rate": float(raw["unexpected_fall_rate"]),
    }


def _isaac_case_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "success_rate": float(bool(raw["success_unified"])),
        "completion_rate": float(raw["completion"]),
        "local_mpjpe_mm": float(raw["mpjpe_mm"]),
        "global_mpjpe_mm": float(raw["eg_mpjpe_mm"]),
        "joint_mae_rad": float(raw["e_joint_rad"]),
        "mpjve_m_s": float(raw["mpjve_mps"]),
        "root_velocity_error_m_s": float(raw["root_vel_err_mps"]),
        "unexpected_fall_rate": float(bool(raw["unexpected_fall"])),
    }


def _zero_metrics(definitions: Iterable[dict[str, str]]) -> dict[str, float]:
    return {
        item["key"]: 1.0 if item["key"] in {"success_rate", "completion_rate"} else 0.0
        for item in definitions
    }


def _mujoco_result_pack(
    any2track: dict[str, Any],
    protomotions: dict[str, Any],
    humanoid_gpt: dict[str, Any],
) -> dict[str, Any]:
    definitions = _metric_definitions("mujoco")
    rows = [
        {
            "method": "GT reference",
            "kind": "reference",
            "rankable": False,
            "checkpoint": None,
            "splits": {
                "lafan1_g1": {
                    "coverage": {"evaluated": 40, "population": 40},
                    "metrics": _zero_metrics(definitions),
                }
            },
            "note": "Kinematic calibration; excluded from controller ranking.",
        }
    ]
    for method, checkpoint, payload in (
        (
            "Any2Track",
            "ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2",
            any2track,
        ),
        (
            "ProtoMotions",
            "ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED",
            protomotions,
        ),
        (
            "HumanoidGPT",
            "ZeyuLing/Motius-HumanoidGPT-G1",
            humanoid_gpt,
        ),
    ):
        rows.append(
            {
                "method": method,
                "kind": "controller",
                "rankable": True,
                "checkpoint": checkpoint,
                "splits": {
                    "lafan1_g1": {
                        "coverage": {"evaluated": 40, "population": 40},
                        "metrics": _mujoco_metrics(payload["aggregate"]),
                    }
                },
            }
        )
    return {
        "schema_version": 2,
        "benchmark_id": "motion_tracking_mujoco_lafan1_g1",
        "status": "complete",
        "generated_at": "2026-07-29",
        "engine": {"name": "MuJoCo", "version": "3.11.0"},
        "protocol_id": "mujoco-g1-reference-tracking-50hz-v1",
        "control_hz": 50,
        "splits": [
            {
                "id": "lafan1_g1",
                "label": "LAFAN1-G1",
                "population": 40,
                "description": "All 40 OpenTrack G1 motions; fixed first 1,000 control steps.",
            }
        ],
        "metric_definitions": definitions,
        "rows": rows,
        "provenance": {
            "result_root": "outputs/motion_tracking/benchmarks/mujoco_lafan1_g1_v1",
            "aggregation": "Episode success; completed-step-weighted physical errors.",
        },
    }


def _gen_summary_root(repo_root: Path) -> Path:
    return (
        repo_root
        / "outputs/evaluation/physflow/gentrack_aaai2027/"
        "protocol_v020_paper_fall_only"
    )


def _isaac_result_pack(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = _metric_definitions("isaaclab")
    summary_root = _gen_summary_root(repo_root)
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    case_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for method in ("sonic_released", "beyondmimic"):
        summaries[method] = {}
        case_metrics[method] = {}
        for split in ("lafan1_g1", "amass_test_g1"):
            summary = _read_json(summary_root / method / split / "summary.json")
            if not summary.get("final_table_eligible"):
                raise ValueError(f"{method}/{split} is not final-table eligible")
            summaries[method][split] = summary
            metric_path = _resolve_recorded_path(
                summary["source_case_metrics"], repo_root
            )
            case_metrics[method][split] = _read_jsonl(metric_path)

    split_populations = {"lafan1_g1": 40, "amass_test_g1": 138}
    rows = [
        {
            "method": "GT reference",
            "kind": "reference",
            "rankable": False,
            "checkpoint": None,
            "splits": {
                split: {
                    "coverage": {"evaluated": population, "population": population},
                    "metrics": _zero_metrics(definitions),
                }
                for split, population in split_populations.items()
            },
            "note": "Kinematic reference; excluded from physical-policy ranking.",
        },
        {
            "method": "SONIC",
            "kind": "controller",
            "rankable": True,
            "checkpoint": "ZeyuLing/Motius-SONIC-G1",
            "splits": {
                split: {
                    "coverage": {
                        "evaluated": int(summaries["sonic_released"][split]["num_cases"]),
                        "population": population,
                    },
                    "metrics": _isaac_metrics(summaries["sonic_released"][split]),
                }
                for split, population in split_populations.items()
            },
        },
        {
            "method": "BeyondMimic",
            "kind": "specialist",
            "rankable": False,
            "checkpoint": None,
            "splits": {
                split: {
                    "coverage": {
                        "evaluated": int(summaries["beyondmimic"][split]["num_cases"]),
                        "population": population,
                    },
                    "metrics": _isaac_metrics(summaries["beyondmimic"][split]),
                }
                for split, population in split_populations.items()
            },
            "note": "Per-reference optimization upper bound; unranked and coverage-disclosed.",
        },
    ]
    result_pack = {
        "schema_version": 2,
        "benchmark_id": "motion_tracking_isaaclab_g1",
        "status": "complete",
        "generated_at": "2026-07-29",
        "engine": {"name": "Isaac Lab", "version": "2.3.2"},
        "protocol_id": "gentrack-fall-only-30fps-v020",
        "control_hz": 50,
        "evaluation_fps": 30,
        "splits": [
            {
                "id": "lafan1_g1",
                "label": "LAFAN1-G1",
                "population": 40,
                "description": "All 40 public LAFAN1 G1 references.",
            },
            {
                "id": "amass_test_g1",
                "label": "AMASS-test-G1",
                "population": 138,
                "description": "Frozen licensed AMASS-G1 test split; BeyondMimic retains 100 per-reference fits.",
            },
        ],
        "metric_definitions": definitions,
        "rows": rows,
        "provenance": {
            "project": "GenTrack",
            "result_root": "outputs/evaluation/physflow/gentrack_aaai2027/protocol_v020_paper_fall_only",
            "success": "Complete rollout, one 30 FPS resample, reference-relative pelvis-height fall-only gate.",
            "aggregation": "Continuous errors include successful and failed trajectories.",
        },
    }
    return result_pack, case_metrics


def _case_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for case in payload["cases"]:
        case_id = str(case["id"])
        base = case_id.split("__f", 1)[0]
        result[base] = case
    return result


def _asset_entry(path: str, qpos: np.ndarray, fps: float) -> dict[str, Any]:
    return {"path": path, "frames": int(len(qpos)), "fps": float(fps)}


def _export_mujoco_assets(
    exporter: G1TransformExporter,
    asset_root: Path,
    result_root: Path,
    any2track: dict[str, Any],
    protomotions: dict[str, Any],
    humanoid_gpt: dict[str, Any],
) -> dict[str, Any]:
    any_cases = _case_lookup(any2track)
    proto_cases = _case_lookup(protomotions)
    humanoid_gpt_cases = _case_lookup(humanoid_gpt)
    ids = sorted(set(any_cases) & set(proto_cases) & set(humanoid_gpt_cases))
    if len(ids) != 40:
        raise ValueError(f"MuJoCo visual population is {len(ids)}, expected 40")
    rollouts = {
        "any2track": result_root / "any2track/rollouts",
        "protomotions": result_root / "protomotions/rollouts",
        "humanoid_gpt": result_root / "humanoid_gpt/rollouts",
    }
    cases = []
    for index, case_id in enumerate(ids):
        filename = _safe_name(case_id, index)
        any_path = next(rollouts["any2track"].glob(f"{case_id}__f*.npz"))
        proto_path = next(rollouts["protomotions"].glob(f"{case_id}__f*.npz"))
        humanoid_gpt_path = next(
            rollouts["humanoid_gpt"].glob(f"{case_id}__f*.npz")
        )
        with np.load(any_path, allow_pickle=False) as archive:
            gt = np.asarray(archive["reference_qpos"], dtype=np.float64)
            any_qpos = np.asarray(archive["qpos"], dtype=np.float64)
            fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        with np.load(proto_path, allow_pickle=False) as archive:
            proto_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        with np.load(humanoid_gpt_path, allow_pickle=False) as archive:
            humanoid_gpt_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        assets = {}
        for key, values in (
            ("gt", gt),
            ("any2track", any_qpos),
            ("protomotions", proto_qpos),
            ("humanoid_gpt", humanoid_gpt_qpos),
        ):
            relative = f"frames/{key}/{filename}"
            exporter.export(values, asset_root / "mujoco" / relative)
            assets[key] = _asset_entry(relative, values, fps)
        cases.append(
            {
                "case_id": case_id,
                "split": "lafan1_g1",
                "label": case_id.replace("_", " "),
                "assets": assets,
                "metrics": {
                    "any2track": _mujoco_metrics(any_cases[case_id]["metrics"]),
                    "protomotions": _mujoco_metrics(proto_cases[case_id]["metrics"]),
                    "humanoid_gpt": _mujoco_metrics(
                        humanoid_gpt_cases[case_id]["metrics"]
                    ),
                },
            }
        )
    return {
        "schema_version": 2,
        "benchmark_id": "motion_tracking_mujoco_lafan1_g1",
        "title": "Motion Tracking · MuJoCo · LAFAN1-G1",
        "population": len(cases),
        "asset_base_url": f"{ASSET_DATASET}/mujoco/",
        "mesh_base_url": f"{MESH_DATASET}/",
        "robot": "robot.json",
        "columns": [
            {"key": "gt", "label": "GT reference"},
            {"key": "any2track", "label": "Any2Track"},
            {"key": "protomotions", "label": "ProtoMotions"},
            {"key": "humanoid_gpt", "label": "HumanoidGPT"},
        ],
        "splits": [{"id": "lafan1_g1", "label": "LAFAN1-G1", "population": 40}],
        "cases": cases,
    }


def _npz_files(path: Path) -> dict[str, Path]:
    return {item.stem: item for item in path.glob("*.npz") if item.is_file()}


def _export_isaac_assets(
    exporter: G1TransformExporter,
    asset_root: Path,
    repo_root: Path,
    results: dict[str, Any],
    case_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> dict[str, Any]:
    summary_root = _gen_summary_root(repo_root)
    cases = []
    for split_index, split in enumerate(("lafan1_g1", "amass_test_g1")):
        sonic_summary = _read_json(summary_root / "sonic_released" / split / "summary.json")
        beyond_summary = _read_json(summary_root / "beyondmimic" / split / "summary.json")
        directories = {
            "gt": _resolve_recorded_path(sonic_summary["reference_qpos_dir"], repo_root),
            "sonic": _resolve_recorded_path(sonic_summary["execution_qpos_dir"], repo_root),
            "beyondmimic": _resolve_recorded_path(beyond_summary["execution_qpos_dir"], repo_root),
        }
        files = {key: _npz_files(path) for key, path in directories.items()}
        ids = sorted(set(files["gt"]) & set(files["sonic"]))
        expected = next(item["population"] for item in results["splits"] if item["id"] == split)
        if len(ids) != expected:
            raise ValueError(f"{split}: found {len(ids)} GT/SONIC cases, expected {expected}")
        for local_index, case_id in enumerate(ids):
            filename = _safe_name(case_id, local_index)
            assets = {}
            for key in ("gt", "sonic", "beyondmimic"):
                source = files[key].get(case_id)
                if source is None:
                    continue
                values, fps = _qpos(source)
                relative = f"frames/{split}/{key}/{filename}"
                exporter.export(values, asset_root / "isaaclab" / relative)
                assets[key] = _asset_entry(relative, values, fps)
            metrics = {}
            for source_key, public_key in (
                ("sonic_released", "sonic"),
                ("beyondmimic", "beyondmimic"),
            ):
                raw = case_metrics[source_key][split].get(case_id)
                if raw is not None:
                    metrics[public_key] = _isaac_case_metrics(raw)
            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "label": case_id.replace("__", " · ").replace("_", " "),
                    "assets": assets,
                    "metrics": metrics,
                }
            )
    return {
        "schema_version": 2,
        "benchmark_id": "motion_tracking_isaaclab_g1",
        "title": "Motion Tracking · Isaac Lab · Unitree G1",
        "population": len(cases),
        "asset_base_url": f"{ASSET_DATASET}/isaaclab/",
        "mesh_base_url": f"{MESH_DATASET}/",
        "robot": "robot.json",
        "columns": [
            {"key": "gt", "label": "GT reference"},
            {"key": "sonic", "label": "SONIC"},
            {"key": "beyondmimic", "label": "BeyondMimic"},
        ],
        "splits": [
            {"id": "lafan1_g1", "label": "LAFAN1-G1", "population": 40},
            {"id": "amass_test_g1", "label": "AMASS-test-G1", "population": 138},
        ],
        "cases": cases,
        "provenance": {
            "source": "GenTrack canonical qpos30 exports",
            "beyondmimic": "Per-reference optimization; unavailable panels are shown explicitly.",
        },
    }


def main() -> None:
    args = _arguments()
    any_path = args.mujoco_root / "any2track/results.json"
    proto_path = args.mujoco_root / "protomotions/results.json"
    any2track = _read_json(any_path)
    protomotions = _read_json(proto_path)
    humanoid_gpt = _read_json(args.mujoco_root / "humanoid_gpt/results.json")
    mujoco_results = _mujoco_result_pack(
        any2track,
        protomotions,
        humanoid_gpt,
    )
    isaac_results, case_metrics = _isaac_result_pack(args.gentrack_root)

    mujoco_docs = DOCS / "hf_space_motion_tracking_mujoco"
    isaac_docs = DOCS / "hf_space_motion_tracking_isaaclab"
    _write_json(mujoco_docs / "motion_tracking_results.json", mujoco_results)
    _write_json(isaac_docs / "motion_tracking_results.json", isaac_results)

    if args.skip_assets:
        return
    exporter = G1TransformExporter()
    robot = exporter.robot_metadata()
    mujoco_manifest = _export_mujoco_assets(
        exporter,
        args.asset_output,
        args.mujoco_root,
        any2track,
        protomotions,
        humanoid_gpt,
    )
    isaac_manifest = _export_isaac_assets(
        exporter,
        args.asset_output,
        args.gentrack_root,
        isaac_results,
        case_metrics,
    )
    for docs, manifest in (
        (mujoco_docs, mujoco_manifest),
        (isaac_docs, isaac_manifest),
    ):
        _write_json(docs / "cases/manifest.json", manifest)
        _write_json(docs / "cases/robot.json", robot)
    print(
        json.dumps(
            {
                "mujoco_cases": mujoco_manifest["population"],
                "isaaclab_cases": isaac_manifest["population"],
                "asset_output": str(args.asset_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
