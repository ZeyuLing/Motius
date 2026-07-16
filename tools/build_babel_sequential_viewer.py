#!/usr/bin/env python3
"""Build a Three.js audit viewer for captioned BABEL episodes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import canonicalize_smpl22_joints
from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip


DEFAULT_CASES = ("val_919", "val_4869", "val_8738")
COLORS = (
    "#e15d44",
    "#1976d2",
    "#19a974",
    "#d69e18",
    "#8b5cf6",
    "#d9468c",
    "#0f9fa8",
)
METHOD_COLORS = ("#c2412d", "#6d4ea2", "#b46900", "#287147", "#9f3f72")


def _keyframe_indices(frames: int, maximum: int) -> np.ndarray:
    if frames <= maximum:
        return np.arange(frames, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, frames - 1, maximum)).astype(np.int64))


def _quantize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=(0, 1)).astype(np.float32)
    maximum = vertices.max(axis=(0, 1)).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    quantized = np.rint((vertices - minimum) / scale).clip(0, 65535).astype("<u2")
    return quantized, minimum, scale


class _SMPLMeshFitter:
    def __init__(self, model_dir: Path, device: str):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.smpl_rest = load_smpl_rest(model_dir, self.device, gender="neutral")
        self.model = self.smpl_rest[0]
        self.faces = np.asarray(self.model.faces, dtype="<u4")

    def fit(self, joints: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self.torch
        selected = np.asarray(joints[keep], dtype=np.float32)
        fitted = retarget_hml263_clip(
            selected,
            smpl_rest=self.smpl_rest,
            device=self.device,
            gender="neutral",
            source_fps=30.0,
            target_fps=30.0,
            rotation_init="position_ik",
            floor_align=False,
            refine_iters=0,
        )
        global_orient = np.asarray(fitted["global_orient"], dtype=np.float32).reshape(-1, 3)
        body_pose = np.asarray(fitted["body_pose"], dtype=np.float32).reshape(len(keep), -1)
        transl = np.asarray(fitted["transl"], dtype=np.float32).reshape(len(keep), 3)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(keep), 96):
                end = min(start + 96, len(keep))
                count = end - start
                body69 = np.zeros((count, 69), dtype=np.float32)
                body69[:, :63] = body_pose[start:end, :63]
                result = self.model(
                    betas=torch.zeros(count, 10, device=self.device),
                    body_pose=torch.from_numpy(body69).to(self.device),
                    global_orient=torch.from_numpy(global_orient[start:end]).to(self.device),
                    transl=torch.from_numpy(transl[start:end]).to(self.device),
                )
                chunks.append(result.vertices.detach().cpu().numpy().astype(np.float32))
        vertices = np.concatenate(chunks, axis=0)
        vertices[..., 1] -= float(vertices[..., 1].min())
        return vertices, float(np.asarray(fitted["fit_mpjpe_mm"]).mean())


def _load_joints(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim == 2 and value.shape[1] == 66:
        value = value.reshape(-1, 22, 3)
    if value.ndim != 3 or value.shape[1:] != (22, 3):
        raise ValueError(f"Expected (T,22,3) joints at {path}, got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite joints at {path}.")
    return canonicalize_smpl22_joints(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Legacy FlowMDM prediction directory.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Prediction to display; repeat for synchronized multi-method comparison.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--retrieval-audit",
        type=Path,
        help="Optional exact R-Precision ranking export from export_babel_retrieval_audit.py.",
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "body_models",
        help="Directory containing the licensed neutral SMPL body model.",
    )
    parser.add_argument("--mesh-keyframes", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _method_key(label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not key:
        raise ValueError(f"Cannot derive a method key from {label!r}.")
    return key


def _prediction_specs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, Path]] = []
    if args.predictions_dir:
        specs.append(("FlowMDM", args.predictions_dir.resolve()))
    for value in args.prediction:
        if "=" not in value:
            raise ValueError(f"--prediction expects LABEL=DIR, got {value!r}.")
        label, directory = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("--prediction labels must not be empty.")
        specs.append((label, Path(directory).expanduser().resolve()))
    if not specs:
        raise ValueError("Provide --predictions-dir or at least one --prediction LABEL=DIR.")
    keys: set[str] = set()
    result = []
    for label, directory in specs:
        key = _method_key(label)
        if key in keys:
            raise ValueError(f"Duplicate prediction key {key!r} for {label!r}.")
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        keys.add(key)
        result.append((key, label, directory))
    return result


def main() -> None:
    args = parse_args()
    predictions = _prediction_specs(args)
    manifest_path = args.manifest.resolve()
    source = json.loads(manifest_path.read_text())
    cases = {item["case_id"]: item for item in source["cases"]}
    audit = None
    retrieval_records = {}
    if args.retrieval_audit:
        audit = json.loads(args.retrieval_audit.resolve().read_text())
        if audit.get("protocol") != source.get("protocol"):
            raise ValueError("Retrieval audit protocol does not match the sequence manifest.")
        retrieval_records = {
            (item["case_id"], int(item["segment_index"])): item
            for item in audit.get("records", [])
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    mesh_fitter = _SMPLMeshFitter(args.smpl_model_dir.resolve(), args.device)
    faces_file = assets / "smpl_faces.u32"
    mesh_fitter.faces.tofile(faces_file)
    exported = []
    for case_id in args.case_ids:
        case = cases[case_id]
        reference = _load_joints(manifest_path.parent / case["reference_path"])
        method_motions = {
            key: _load_joints(directory / f"{case_id}.npy")
            for key, _label, directory in predictions
        }
        frames = min(
            len(reference),
            *(len(value) for value in method_motions.values()),
            int(case["total_frames"]),
        )
        reference = np.ascontiguousarray(reference[:frames], dtype="<f4")
        reference_file = assets / f"{case_id}_gt_joints.f32"
        reference.tofile(reference_file)
        motion_files = {"gt": reference_file.relative_to(args.output_dir).as_posix()}
        for key, _label, _directory in predictions:
            motion = np.ascontiguousarray(method_motions[key][:frames], dtype="<f4")
            motion_file = assets / f"{case_id}_{key}_joints.f32"
            motion.tofile(motion_file)
            motion_files[key] = motion_file.relative_to(args.output_dir).as_posix()
        mesh_files = {}
        keep = _keyframe_indices(frames, args.mesh_keyframes)
        all_joints = {"gt": reference, **method_motions}
        for key, joints in all_joints.items():
            vertices, fit_mpjpe_mm = mesh_fitter.fit(joints[:frames], keep)
            quantized, minimum, scale = _quantize_vertices(vertices)
            mesh_file = assets / f"{case_id}_{key}_smpl_vertices.u16"
            quantized.tofile(mesh_file)
            mesh_files[key] = {
                "vertices_file": mesh_file.relative_to(args.output_dir).as_posix(),
                "keyframe_indices": keep.astype(int).tolist(),
                "keyframe_count": int(len(keep)),
                "vertex_count": int(vertices.shape[1]),
                "quantization_min": minimum.tolist(),
                "quantization_scale": scale.tolist(),
                "fit_mpjpe_mm": fit_mpjpe_mm,
            }
        segments = []
        for index, segment in enumerate(case["segments"]):
            if int(segment["start_frame"]) >= frames:
                continue
            exported_segment = {
                "caption": segment["caption"],
                "raw_label": segment.get("raw_label", ""),
                "action_categories": segment.get("action_categories", []),
                "action_group_id": segment.get("action_group_id"),
                "start_frame": int(segment["start_frame"]),
                "end_frame": min(int(segment["end_frame"]), frames),
                "color": COLORS[index % len(COLORS)],
            }
            if audit is not None:
                record = retrieval_records.get((case_id, index))
                if record is None:
                    raise ValueError(f"Retrieval audit is missing {case_id} segment {index}.")
                exported_segment["retrieval"] = {
                    "gt": record["gt"],
                    "flowmdm": record["flowmdm"],
                }
            segments.append(exported_segment)
        exported.append(
            {
                "case_id": case_id,
                "frames": frames,
                "fps": float(source.get("fps", 30.0)),
                "motion_files": motion_files,
                "mesh_files": mesh_files,
                "segments": segments,
            }
        )
    viewer_source = Path(__file__).with_name("babel_sequential_viewer.html")
    shutil.copy2(viewer_source, args.output_dir / "index.html")
    payload = {
        "protocol": source["protocol"],
        "methods": [
            {"key": "gt", "label": "BABEL GT", "accent": "#0e7490"},
            *[
                {
                    "key": key,
                    "label": label,
                    "accent": METHOD_COLORS[index % len(METHOD_COLORS)],
                }
                for index, (key, label, _directory) in enumerate(predictions)
            ],
        ],
        "parents": [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
        "mesh": {
            "body_model": "neutral SMPL",
            "vertex_count": 6890,
            "faces_file": faces_file.relative_to(args.output_dir).as_posix(),
            "face_count": int(len(mesh_fitter.faces)),
            "source": "position IK from the canonical SMPL-22 joints used by evaluation",
        },
        "episodes": exported,
    }
    if audit is not None:
        payload["retrieval"] = {
            "evaluator": audit["evaluator"],
            "seed": int(audit["seed"]),
            "chunk_size": int(audit["chunk_size"]),
            "top_k": int(audit["top_k"]),
            "direction_note": audit["direction_note"],
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "cases": args.case_ids,
                "methods": [label for _key, label, _directory in predictions],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
