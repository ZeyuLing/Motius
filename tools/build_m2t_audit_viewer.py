#!/usr/bin/env python3
"""Build a Three.js SMPL-mesh audit viewer for HumanML3D M2T outputs."""

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

from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip


DEFAULT_CASES = (
    "000000",
    "000019",
    "001840",
    "004545",
    "006944",
    "014457",
    "004822",
    "M013344",
)
CASE_TAGS = {
    "000000": "body-part error",
    "000019": "trajectory",
    "001840": "left / right",
    "004545": "high agreement",
    "006944": "fine-grained hand",
    "014457": "object interaction",
    "004822": "paraphrase",
    "M013344": "upper-body action",
}
METHODS = (
    ("tm2t", "TM2T", "#0f887b"),
    ("motiongpt", "MotionGPT (stable)", "#3868a6"),
    ("motiongpt3", "MotionGPT3", "#b44732"),
    ("vermo", "VerMo", "#a96e00"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("outputs/m2t/humanml3d/protocol_manifest.json"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/m2t/humanml3d/full_v1"),
    )
    parser.add_argument(
        "--prediction-dir",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override one method's prediction directory; may be repeated.",
    )
    parser.add_argument(
        "--motiongpt-batch32-dir",
        type=Path,
        help="Optionally show MotionGPT's released batch-32 padding variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m2t/humanml3d/visualization/m2t_smpl_audit"),
    )
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "body_models" / "smpl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh-keyframes", type=int, default=120)
    parser.add_argument("--refine-iters", type=int, default=60)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def keyframe_indices(frames: int, maximum: int) -> np.ndarray:
    if frames <= maximum:
        return np.arange(frames, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, frames - 1, maximum)).astype(np.int64))


def quantize_vertices(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=(0, 1)).astype(np.float32)
    maximum = vertices.max(axis=(0, 1)).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    values = np.rint((vertices - minimum) / scale).clip(0, 65535).astype("<u2")
    return values, minimum, scale


def prediction_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    known = {method for method, _label, _color in METHODS}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or method not in known or not raw_path:
            raise ValueError(
                f"Invalid --prediction-dir {value!r}; expected METHOD=PATH with "
                f"METHOD in {sorted(known)}."
            )
        overrides[method] = Path(raw_path).expanduser().resolve()
    return overrides


def method_prediction(
    root: Path,
    overrides: dict[str, Path],
    method: str,
    sample_id: str,
) -> str:
    method_root = overrides.get(method, root / method)
    path = method_root / "predictions" / f"{sample_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())["prediction"]
    return str(value).strip()


def asset_stem(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def render_vertices(
    model,
    torch,
    converted: dict[str, object],
    keep: np.ndarray,
    device,
) -> np.ndarray:
    global_orient = np.asarray(converted["global_orient"], dtype=np.float32).reshape(-1, 3)
    body_pose = np.asarray(converted["body_pose"], dtype=np.float32).reshape(
        len(global_orient), -1
    )
    transl = np.asarray(converted["transl"], dtype=np.float32).reshape(-1, 3)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(keep), 96):
            indices = keep[start : start + 96]
            count = len(indices)
            body69 = np.zeros((count, 69), dtype=np.float32)
            body69[:, : min(63, body_pose.shape[1])] = body_pose[indices, :63]
            result = model(
                betas=torch.zeros(count, 10, device=device),
                body_pose=torch.from_numpy(body69).to(device),
                global_orient=torch.from_numpy(global_orient[indices]).to(device),
                transl=torch.from_numpy(transl[indices]).to(device),
            )
            chunks.append(result.vertices.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text())
    samples = {item["sample_id"]: item for item in protocol["samples"]}
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else Path(protocol["data_root"]).expanduser().resolve()
    )
    prediction_root = args.prediction_root.resolve()
    overrides = prediction_overrides(args.prediction_dir)
    methods = list(METHODS)
    if args.motiongpt_batch32_dir:
        overrides["motiongpt_batch32"] = (
            args.motiongpt_batch32_dir.expanduser().resolve()
        )
        methods.insert(
            2,
            (
                "motiongpt_batch32",
                "MotionGPT (released batch-32)",
                "#7655a3",
            ),
        )
    output_dir = args.output_dir.resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device(args.device)
    smpl_rest = load_smpl_rest(args.smpl_model_dir.expanduser(), device, gender="neutral")
    model = smpl_rest[0]
    faces = np.asarray(model.faces, dtype="<u4")
    faces_path = assets_dir / "smpl_faces.u32"
    faces.tofile(faces_path)

    exported = []
    for sample_id in args.case_ids:
        if sample_id not in samples:
            raise KeyError(f"Sample {sample_id!r} is not in {protocol_path}.")
        sample = samples[sample_id]
        start = int(sample["start_frame"])
        end = min(int(sample["end_frame"]), start + 196)
        source_path = data_root / sample["motion_path"]
        source = np.asarray(np.load(source_path), dtype=np.float32)[start:end]
        if source.ndim != 2 or source.shape[1] != 263:
            raise ValueError(f"{source_path}: expected (T,263), got {source.shape}.")
        stem = asset_stem(sample_id)
        mesh_path = assets_dir / f"{stem}_smpl_vertices.u16"
        meta_path = assets_dir / f"{stem}_smpl_meta.json"
        if mesh_path.is_file() and meta_path.is_file() and not args.overwrite:
            mesh_meta = json.loads(meta_path.read_text())
        else:
            converted = retarget_hml263_clip(
                source,
                smpl_rest=smpl_rest,
                device=device,
                gender="neutral",
                source_fps=20.0,
                target_fps=20.0,
                target_len=len(source),
                floor_align=True,
                refine_iters=args.refine_iters,
                refine_lr=args.refine_lr,
                rotation_init="position_ik",
            )
            keep = keyframe_indices(len(source), args.mesh_keyframes)
            vertices = render_vertices(model, torch, converted, keep, device)
            vertices[..., 1] -= float(vertices[..., 1].min())
            quantized, minimum, scale = quantize_vertices(vertices)
            quantized.tofile(mesh_path)
            transl = np.asarray(converted["transl"], dtype=np.float32).reshape(-1, 3)
            bounds_min = vertices.min(axis=(0, 1)).astype(np.float32)
            bounds_max = vertices.max(axis=(0, 1)).astype(np.float32)
            errors = np.asarray(converted["fit_mpjpe_mm"], dtype=np.float32)
            mesh_meta = {
                "vertices_file": mesh_path.relative_to(output_dir).as_posix(),
                "keyframe_indices": keep.astype(int).tolist(),
                "keyframe_count": int(len(keep)),
                "vertex_count": int(vertices.shape[1]),
                "quantization_min": minimum.tolist(),
                "quantization_scale": scale.tolist(),
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "root_path": transl[keep].tolist(),
                "fit_mpjpe_mm_mean": float(errors.mean()),
                "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
            }
            meta_path.write_text(json.dumps(mesh_meta, indent=2) + "\n")
        exported.append(
            {
                "sample_id": sample_id,
                "tag": CASE_TAGS.get(sample_id, "audit"),
                "frames": len(source),
                "fps": 20.0,
                "references": [
                    caption["caption"] for caption in sample["captions"][:3]
                ],
                "outputs": {
                    method: method_prediction(
                        prediction_root, overrides, method, sample_id
                    )
                    for method, _label, _color in methods
                },
                "mesh": mesh_meta,
            }
        )
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "frames": len(source),
                    "fit_mpjpe_mm": mesh_meta["fit_mpjpe_mm_mean"],
                }
            ),
            flush=True,
        )

    shutil.copy2(Path(__file__).with_name("m2t_audit_viewer.html"), output_dir / "index.html")
    manifest = {
        "title": "HumanML3D M2T Caption Audit",
        "protocol": protocol["protocol"],
        "methods": [
            {"key": method, "label": label, "accent": color}
            for method, label, color in methods
        ],
        "mesh": {
            "body_model": "neutral SMPL",
            "vertex_count": 6890,
            "faces_file": faces_path.relative_to(output_dir).as_posix(),
            "face_count": int(len(faces)),
        },
        "cases": exported,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"output": str(output_dir), "cases": len(exported)}, indent=2))


if __name__ == "__main__":
    main()
