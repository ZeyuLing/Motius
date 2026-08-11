"""Licensed-data-safe manifests for monocular motion-capture evaluation."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np


MANIFEST_REVISION = "motius_monocular_capture_v1"
SUPPORTED_PROTOCOLS = {
    "3dpw_test_camera_v1",
    "emdb_1_camera_v1",
    "emdb_2_global_v1",
}


def _safe_relative(path: Path, *, field_name: str) -> Path:
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{field_name} must be a safe path relative to the private data root."
        )
    return path


@dataclass(frozen=True)
class MonocularCaptureSample:
    """One person track in a licensed benchmark sequence."""

    dataset: str
    split: str
    protocol: str
    sequence_id: str
    track_id: str
    input_path: Path
    input_kind: str
    annotation_path: Path
    fps: float
    start_frame: int
    end_frame: int
    expected_body_model: str = "smpl"
    metadata: Mapping[str, object] = None

    def __post_init__(self) -> None:
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported monocular protocol {self.protocol!r}.")
        if self.input_kind not in {"video", "image_sequence"}:
            raise ValueError("input_kind must be video or image_sequence.")
        if not all(
            (
                self.dataset,
                self.split,
                self.sequence_id,
                self.track_id,
                self.expected_body_model,
            )
        ):
            raise ValueError("Sample identity and body model fields are required.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("Frame interval must be non-empty and half-open.")
        object.__setattr__(
            self,
            "input_path",
            _safe_relative(self.input_path, field_name="input_path"),
        )
        object.__setattr__(
            self,
            "annotation_path",
            _safe_relative(self.annotation_path, field_name="annotation_path"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def sample_id(self) -> str:
        return f"{self.protocol}:{self.sequence_id}:{self.track_id}"

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def resolved(self, data_root: Path) -> tuple[Path, Path]:
        root = Path(data_root).resolve()
        input_path = (root / self.input_path).resolve()
        annotation_path = (root / self.annotation_path).resolve()
        for value in (input_path, annotation_path):
            if root not in value.parents and value != root:
                raise ValueError("Resolved benchmark path escaped data_root.")
        return input_path, annotation_path

    def as_record(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "protocol": self.protocol,
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "input_path": self.input_path.as_posix(),
            "input_kind": self.input_kind,
            "annotation_path": self.annotation_path.as_posix(),
            "fps": self.fps,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "expected_body_model": self.expected_body_model,
            "metadata": dict(self.metadata),
        }


def write_monocular_capture_manifest(
    samples: Iterable[MonocularCaptureSample],
    path: Path,
    *,
    dataset_license: str,
    source: str,
) -> dict:
    """Write an index containing no benchmark media or absolute private paths."""

    records = list(samples)
    if not records:
        raise ValueError("Refusing to write an empty monocular manifest.")
    sample_ids = [sample.sample_id for sample in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Monocular manifest sample IDs must be unique.")
    protocols = sorted({sample.protocol for sample in records})
    payload = {
        "schema_revision": MANIFEST_REVISION,
        "source": source,
        "dataset_license": dataset_license,
        "protocols": protocols,
        "population": len(records),
        "total_frames": sum(sample.num_frames for sample in records),
        "private_data_policy": {
            "paths": "relative_to_user_supplied_data_root",
            "contains_media": False,
            "publishable": True,
        },
        "samples": [sample.as_record() for sample in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_monocular_capture_manifest(
    path: Path,
    *,
    data_root: Optional[Path] = None,
    require_files: bool = False,
) -> tuple[MonocularCaptureSample, ...]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_revision") != MANIFEST_REVISION:
        raise ValueError(
            f"Expected {MANIFEST_REVISION}, got {payload.get('schema_revision')!r}."
        )
    records = tuple(
        MonocularCaptureSample(
            dataset=record["dataset"],
            split=record["split"],
            protocol=record["protocol"],
            sequence_id=record["sequence_id"],
            track_id=record["track_id"],
            input_path=Path(record["input_path"]),
            input_kind=record["input_kind"],
            annotation_path=Path(record["annotation_path"]),
            fps=float(record["fps"]),
            start_frame=int(record["start_frame"]),
            end_frame=int(record["end_frame"]),
            expected_body_model=record.get("expected_body_model", "smpl"),
            metadata=record.get("metadata", {}),
        )
        for record in payload["samples"]
    )
    if payload.get("population") != len(records):
        raise ValueError("Manifest population does not match sample records.")
    if require_files:
        if data_root is None:
            raise ValueError("require_files=True requires data_root.")
        missing = [
            str(path)
            for sample in records
            for path in sample.resolved(data_root)
            if not path.exists()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing licensed benchmark files: {preview}")
    return records


def _load_annotation(path: Path) -> dict:
    """Load an official benchmark pickle supplied by the licensed user."""

    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary annotation in {path}.")
    return payload


def build_3dpw_test_samples(data_root: Path) -> tuple[MonocularCaptureSample, ...]:
    """Index the official 3DPW ``sequenceFiles/test`` split by person track."""

    root = Path(data_root).resolve()
    annotation_dir = root / "sequenceFiles" / "test"
    annotation_paths = sorted(annotation_dir.glob("*.pkl"))
    if not annotation_paths:
        raise FileNotFoundError(
            f"No 3DPW test annotations found under {annotation_dir}."
        )
    samples = []
    for annotation_path in annotation_paths:
        payload = _load_annotation(annotation_path)
        sequence_id = str(payload.get("sequence", annotation_path.stem))
        poses = payload.get("poses")
        if not isinstance(poses, (list, tuple)):
            raise ValueError(f"3DPW poses are missing in {annotation_path}.")
        image_path = root / "imageFiles" / sequence_id
        if not image_path.is_dir():
            raise FileNotFoundError(f"Missing 3DPW image sequence {image_path}.")
        validity = payload.get("campose_valid")
        for person_index, person_poses in enumerate(poses):
            frames = int(np.asarray(person_poses).shape[0])
            if frames <= 0:
                continue
            if isinstance(validity, (list, tuple)) and person_index < len(validity):
                valid_count = int(np.asarray(validity[person_index], dtype=bool).sum())
            else:
                valid_count = frames
            samples.append(
                MonocularCaptureSample(
                    dataset="3DPW",
                    split="test",
                    protocol="3dpw_test_camera_v1",
                    sequence_id=sequence_id,
                    track_id=f"person_{person_index}",
                    input_path=image_path.relative_to(root),
                    input_kind="image_sequence",
                    annotation_path=annotation_path.relative_to(root),
                    fps=float(payload.get("fps", 30.0)),
                    start_frame=0,
                    end_frame=frames,
                    expected_body_model="smpl",
                    metadata={
                        "person_index": person_index,
                        "valid_frames": valid_count,
                        "annotation_format": "official_3dpw_sequence_pickle",
                    },
                )
            )
    if not samples:
        raise ValueError("3DPW test annotations contained no person tracks.")
    return tuple(samples)


def build_emdb_samples(data_root: Path) -> tuple[MonocularCaptureSample, ...]:
    """Index EMDB-1 camera and EMDB-2 global protocols from official flags."""

    root = Path(data_root).resolve()
    annotation_paths = sorted(root.glob("P*/*/*_data.pkl"))
    if not annotation_paths:
        annotation_paths = sorted(root.rglob("*_data.pkl"))
    if not annotation_paths:
        raise FileNotFoundError(f"No EMDB *_data.pkl files found under {root}.")
    samples = []
    for annotation_path in annotation_paths:
        payload = _load_annotation(annotation_path)
        sequence_id = str(payload.get("name", annotation_path.stem.removesuffix("_data")))
        frames = int(payload.get("n_frames", 0))
        if frames <= 0:
            raise ValueError(f"Invalid EMDB n_frames in {annotation_path}.")
        image_path = annotation_path.parent / "images"
        if not image_path.is_dir():
            raise FileNotFoundError(f"Missing EMDB image sequence {image_path}.")
        valid_count = int(
            np.asarray(
                payload.get("good_frames_mask", np.ones(frames)),
                dtype=bool,
            ).sum()
        )
        protocols = []
        if bool(payload.get("emdb1")):
            protocols.append("emdb_1_camera_v1")
        if bool(payload.get("emdb2")):
            protocols.append("emdb_2_global_v1")
        for protocol in protocols:
            samples.append(
                MonocularCaptureSample(
                    dataset="EMDB",
                    split="test",
                    protocol=protocol,
                    sequence_id=sequence_id,
                    track_id="person_0",
                    input_path=image_path.relative_to(root),
                    input_kind="image_sequence",
                    annotation_path=annotation_path.relative_to(root),
                    fps=float(payload.get("fps", 30.0)),
                    start_frame=0,
                    end_frame=frames,
                    expected_body_model="smpl",
                    metadata={
                        "valid_frames": valid_count,
                        "gender": str(payload.get("gender", "neutral")),
                        "annotation_format": "official_emdb_sequence_pickle",
                    },
                )
            )
    if not samples:
        raise ValueError("EMDB annotations did not select any EMDB-1/EMDB-2 sequences.")
    return tuple(samples)


__all__ = [
    "MANIFEST_REVISION",
    "SUPPORTED_PROTOCOLS",
    "MonocularCaptureSample",
    "build_3dpw_test_samples",
    "build_emdb_samples",
    "load_monocular_capture_manifest",
    "write_monocular_capture_manifest",
]
