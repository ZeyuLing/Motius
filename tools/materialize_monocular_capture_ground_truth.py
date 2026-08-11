#!/usr/bin/env python3
"""Materialize licensed 3DPW/EMDB GT with user-supplied SMPL model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.evaluation.monocular_capture import load_monocular_capture_manifest
from motius.evaluation.monocular_ground_truth import (
    GROUND_TRUTH_REVISION,
    OFFICIAL_SOURCE_REVISIONS,
    SMPLGeometry,
    materialize_monocular_ground_truth,
    sha256_file,
)
from motius.motion.representation.monocular_capture import (
    save_monocular_capture_result,
)


_GENDERS = {"male", "female", "neutral"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay user-licensed official annotations through user-supplied "
            "SMPL files. The command never downloads or copies benchmark/model "
            "data, and only writes under this repository's outputs directory."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--smpl-model",
        action="append",
        metavar="GENDER=FILE",
        help=(
            "Exact licensed SMPL .pkl for male, female, or neutral. Repeat for "
            "every gender present in the manifest."
        ),
    )
    parser.add_argument(
        "--smpl-model-version",
        help="User-declared licensed SMPL release recorded in provenance.",
    )
    parser.add_argument(
        "--joint-only-3dpw",
        action="store_true",
        help=(
            "Use the official 3DPW jointPositions field without licensed SMPL. "
            "This disables PVE and is not valid for EMDB."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must resolve below this repository's outputs/ directory.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing materialized artifact with the same sample ID.",
    )
    return parser


def require_repository_output_path(path: Path) -> Path:
    """Resolve and enforce the repository-local outputs boundary."""

    output_root = (ROOT / "outputs").resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Output must live below repository directory {output_root}."
        ) from exc
    return resolved


def _parse_model_files(values: list[str]) -> dict[str, Path]:
    model_files: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--smpl-model must use GENDER=FILE syntax.")
        gender, raw_path = value.split("=", 1)
        gender = gender.strip().lower()
        if gender not in _GENDERS:
            raise ValueError(
                f"Unknown SMPL gender {gender!r}; expected {sorted(_GENDERS)}."
            )
        if gender in model_files:
            raise ValueError(f"Duplicate SMPL model mapping for {gender}.")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SMPL model file does not exist: {path}")
        model_files[gender] = path
    return model_files


class LicensedTorchSMPL:
    """Thin batched adapter around the optional user-installed ``smplx``."""

    def __init__(
        self,
        model_files: Mapping[str, Path],
        *,
        model_version: str,
        device: str,
        batch_size: int,
    ) -> None:
        if not model_version:
            raise ValueError("--smpl-model-version must be non-empty.")
        if batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        try:
            import torch
            import smplx
        except ImportError as exc:
            raise RuntimeError(
                "Materialization requires the optional smplx package; install it "
                "in the private runtime without adding or copying SMPL files."
            ) from exc
        self._torch = torch
        self._smplx = smplx
        self._model_files = dict(model_files)
        self._fingerprints = {
            gender: sha256_file(path)
            for gender, path in self._model_files.items()
        }
        self.model_version = model_version
        self.batch_size = int(batch_size)
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self._models: dict[str, object] = {}

    @property
    def fingerprints(self) -> dict[str, str]:
        return dict(self._fingerprints)

    def fingerprint_for_gender(self, gender: str) -> str:
        try:
            return self._fingerprints[gender]
        except KeyError as exc:
            raise ValueError(
                f"No --smpl-model {gender}=FILE mapping was supplied."
            ) from exc

    def _model(self, gender: str):
        if gender not in self._models:
            try:
                model_path = self._model_files[gender]
            except KeyError as exc:
                raise ValueError(
                    f"No --smpl-model {gender}=FILE mapping was supplied."
                ) from exc
            model = self._smplx.SMPL(
                str(model_path),
                gender=gender,
                num_betas=10,
                create_betas=False,
                create_global_orient=False,
                create_body_pose=False,
                create_transl=False,
            )
            self._models[gender] = model.to(self.device).eval()
        return self._models[gender]

    def materialize(
        self,
        *,
        poses_axis_angle: np.ndarray,
        betas: np.ndarray,
        translation: np.ndarray,
        gender: str,
    ) -> SMPLGeometry:
        torch = self._torch
        model = self._model(gender)
        vertices, joints = [], []
        for start in range(0, len(poses_axis_angle), self.batch_size):
            end = min(start + self.batch_size, len(poses_axis_angle))
            pose = torch.as_tensor(
                poses_axis_angle[start:end],
                dtype=torch.float32,
                device=self.device,
            )
            shape = torch.as_tensor(
                betas[start:end],
                dtype=torch.float32,
                device=self.device,
            )
            transl = torch.as_tensor(
                translation[start:end],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                output = model(
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:],
                    betas=shape,
                    transl=transl,
                    return_verts=True,
                )
            if output.joints.shape[1] < 24:
                raise RuntimeError(
                    "Licensed SMPL runtime emitted fewer than 24 kinematic joints."
                )
            vertices.append(output.vertices.detach().cpu().numpy())
            # smplx appends optional vertex-selected landmarks after the 24
            # official SMPL kinematic joints; the benchmark contract uses 0:24.
            joints.append(output.joints[:, :24].detach().cpu().numpy())
        return SMPLGeometry(
            vertices=np.concatenate(vertices, axis=0).astype(
                np.float32,
                copy=False,
            ),
            joints=np.concatenate(joints, axis=0).astype(
                np.float32,
                copy=False,
            ),
        )


def _artifact_name(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._")
    suffix = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}--{suffix}.npz"


def main() -> None:
    args = _parser().parse_args()
    output_dir = require_repository_output_path(args.output_dir)
    samples = load_monocular_capture_manifest(args.manifest)
    if args.joint_only_3dpw:
        if any(sample.protocol != "3dpw_test_camera_v1" for sample in samples):
            raise ValueError("--joint-only-3dpw accepts only a 3DPW manifest.")
        if args.smpl_model:
            raise ValueError(
                "--joint-only-3dpw must not be combined with --smpl-model."
            )
        body_model = None
        model_files = {}
    else:
        if not args.smpl_model or not args.smpl_model_version:
            raise ValueError(
                "Licensed replay requires --smpl-model and "
                "--smpl-model-version."
            )
        model_files = _parse_model_files(args.smpl_model)
        body_model = LicensedTorchSMPL(
            model_files,
            model_version=args.smpl_model_version,
            device=args.device,
            batch_size=args.batch_size,
        )
    data_root = args.data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "ground_truth_index.json"
    if index_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace {index_path}; pass --overwrite explicitly."
        )

    records = []
    seen_artifacts: set[Path] = set()
    for sample in samples:
        _, annotation_path = sample.resolved(data_root)
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Licensed annotation does not exist: {annotation_path}"
            )
        artifact = output_dir / sample.protocol / _artifact_name(sample.sample_id)
        if artifact in seen_artifacts:
            raise ValueError(f"Duplicate artifact path generated for {sample.sample_id}.")
        seen_artifacts.add(artifact)
        if artifact.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to replace {artifact}; pass --overwrite explicitly."
            )
        result = materialize_monocular_ground_truth(
            sample,
            annotation_path,
            body_model,
        )
        save_monocular_capture_result(result, artifact)
        records.append(
            {
                "sample_id": sample.sample_id,
                "protocol": sample.protocol,
                "artifact": artifact.relative_to(output_dir).as_posix(),
                "annotation_sha256": result.metadata["annotation_sha256"],
                "model_sha256": result.metadata["model_sha256"],
                "coordinate_space": result.metadata["coordinate_space"],
                "valid_frames": int(result.tracks[0].valid.sum()),
                "total_frames": result.tracks[0].num_frames,
                "public_manifest": result.public_manifest(),
            }
        )

    index = {
        "schema_revision": GROUND_TRUTH_REVISION,
        "manifest_sha256": sha256_file(args.manifest),
        "model_version": args.smpl_model_version,
        "model_fingerprints": (
            {} if body_model is None else body_model.fingerprints
        ),
        "geometry_source": (
            "official_3dpw_jointPositions"
            if body_model is None
            else "licensed_smpl_replay"
        ),
        "official_source_revisions": dict(OFFICIAL_SOURCE_REVISIONS),
        "population": len(records),
        "artifacts": records,
        "private_data_policy": {
            "downloads_performed": False,
            "annotations_copied": False,
            "smpl_models_copied": False,
            "output_boundary": "repository_outputs_only",
        },
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(
        f"Materialized {len(records)} licensed GT tracks under "
        f"{output_dir.relative_to((ROOT / 'outputs').resolve())}."
    )


if __name__ == "__main__":
    main()
