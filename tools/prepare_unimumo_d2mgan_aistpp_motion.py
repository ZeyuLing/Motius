#!/usr/bin/env python3
"""Fill filtered UniMuMo AIST++ motions from the official SMPL release."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import motius.models.flowmdm.network.data_loaders.humanml.scripts.motion_process as hml
from motius.evaluation.protocols import d2mgan_aistpp_test_segments
from motius.models.flowmdm.network.data_loaders.humanml.common.skeleton import (
    Skeleton,
)
from motius.models.flowmdm.network.data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain,
    t2m_raw_offsets,
)
from motius.motion.retarget._hml263_smpl_impl import recover_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", required=True, type=Path)
    parser.add_argument("--aistpp-motion-zip", required=True, type=Path)
    parser.add_argument(
        "--web-rig-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "leaderboards"
            / "hf_space_music_to_dance"
            / "cases"
            / "smpl_model"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_motion(root: Path, motion_id: str) -> Path | None:
    matches = [
        root / split / "joint_vecs" / f"{motion_id}.npy"
        for split in ("train", "val", "test")
    ]
    existing = [path for path in matches if path.is_file()]
    if len(existing) > 1:
        raise RuntimeError(f"Duplicate motion {motion_id}: {existing}")
    return existing[0] if existing else None


def configure_humanml(reference: np.ndarray) -> None:
    hml.l_idx1, hml.l_idx2 = 5, 8
    hml.fid_r, hml.fid_l = [8, 11], [7, 10]
    hml.face_joint_indx = [2, 1, 17, 16]
    hml.r_hip, hml.l_hip = 2, 1
    hml.joints_num = 22
    hml.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    hml.kinematic_chain = t2m_kinematic_chain
    recovered = recover_from_ric(np.asarray(reference, dtype=np.float32))
    skeleton = Skeleton(hml.n_raw_offsets, t2m_kinematic_chain, "cpu")
    hml.tgt_offsets = skeleton.get_offsets_joints(
        torch.from_numpy(recovered[0])
    )


def load_web_rig(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
    joints = np.fromfile(path / "joints.f32", dtype="<f4").reshape(-1, 3)
    parents = np.asarray(metadata["parents"], dtype=np.int64)
    return joints[:22].astype(np.float32), parents[:22]


def aistpp_smpl_to_joints(
    payload: dict,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(payload["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3)
    local = Rotation.from_rotvec(poses[:, :22].reshape(-1, 3)).as_matrix()
    local = local.reshape(len(poses), 22, 3, 3).astype(np.float32)
    global_rotation = np.empty_like(local)
    global_rotation[:, 0] = local[:, 0]
    joints = np.empty((len(poses), 22, 3), dtype=np.float32)
    scaling = float(np.asarray(payload["smpl_scaling"]).reshape(-1)[0])
    translation = np.asarray(payload["smpl_trans"], dtype=np.float32) / scaling
    joints[:, 0] = rest_joints[0] + translation
    for joint in range(1, 22):
        parent = int(parents[joint])
        global_rotation[:, joint] = (
            global_rotation[:, parent] @ local[:, joint]
        )
        offset = rest_joints[joint] - rest_joints[parent]
        joints[:, joint] = joints[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], offset
        )
    return joints


def required_frames_by_motion() -> dict[str, int]:
    result: dict[str, int] = {}
    for segment in d2mgan_aistpp_test_segments():
        result[segment.source_motion_id] = max(
            result.get(segment.source_motion_id, 0),
            (segment.segment_index - 1) * 120 + 114,
        )
    return result


def main() -> None:
    args = parse_args()
    root = args.motion_root.expanduser().resolve()
    required = required_frames_by_motion()
    existing = {
        motion_id: find_motion(root, motion_id) for motion_id in required
    }
    reference_path = next((path for path in existing.values() if path is not None), None)
    if reference_path is None:
        raise FileNotFoundError("At least one published UniMuMo AIST++ motion is required")
    configure_humanml(np.load(reference_path))
    rest_joints, parents = load_web_rig(args.web_rig_dir.expanduser().resolve())
    output_root = root / "train" / "joint_vecs"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(args.aistpp_motion_zip.expanduser().resolve()) as archive:
        for motion_id, minimum_frames in sorted(required.items()):
            current = existing[motion_id]
            destination = output_root / f"{motion_id}.npy"
            if current is not None and not args.overwrite:
                motion = np.load(current, mmap_mode="r")
                rows.append(
                    {
                        "motion_id": motion_id,
                        "source": "UniMuMo published HML263",
                        "path": str(current.relative_to(root)),
                        "frames": int(len(motion)),
                    }
                )
                continue
            member = f"motions/{motion_id}.pkl"
            try:
                payload = pickle.loads(archive.read(member))
            except KeyError as exc:
                raise FileNotFoundError(member) from exc
            joints = aistpp_smpl_to_joints(payload, rest_joints, parents)
            features, _, _, _ = hml.process_file(joints, 0.002)
            features = np.asarray(features, dtype=np.float32)
            if len(features) < minimum_frames:
                raise ValueError(
                    f"{motion_id} has {len(features)} frames, needs {minimum_frames}"
                )
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite HML263 features for {motion_id}")
            np.save(destination, features)
            rows.append(
                {
                    "motion_id": motion_id,
                    "source": "AIST++ official SMPL reconstructed to HML263",
                    "path": str(destination.relative_to(root)),
                    "frames": int(len(features)),
                    "smpl_loss": float(payload.get("smpl_loss", np.nan)),
                }
            )
    manifest = {
        "schema_version": 1,
        "protocol": "D2M-GAN public AIST++ 86-segment split",
        "motion_ids": len(required),
        "published_unimumo": sum(
            row["source"] == "UniMuMo published HML263" for row in rows
        ),
        "reconstructed_aistpp": sum(
            row["source"].startswith("AIST++") for row in rows
        ),
        "reconstruction_note": (
            "The UniMuMo public motion archive omits AIST++ sequences on its "
            "filter list. Missing protocol inputs are reconstructed from the "
            "official AIST++ SMPL release with the public neutral web rig; "
            "shape/gender differences are not retained."
        ),
        "motions": rows,
    }
    path = root / "d2mgan_official86_motion_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
