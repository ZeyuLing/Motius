from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.add_smpl_gallery_method import add_method


def test_add_method_writes_chunked_descriptors_and_asset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    descriptors = source / "descriptors"
    descriptors.mkdir(parents=True)
    cases = [
        {"case_id": "a", "references": ["first"]},
        {"case_id": "b", "references": ["second"]},
    ]
    manifest = {
        "motion_methods": [{"key": "gt", "label": "GT", "accent": "#000"}],
        "cases": cases,
        "case_descriptor_chunks": {
            "size": 2,
            "path": "descriptors/{chunk}.json",
        },
    }
    (descriptors / "000.json").write_text(
        json.dumps(
            {
                "start": 0,
                "motions": [
                    {"gt": {"asset": "assets/gt_000.smpl"}},
                    {"gt": {"asset": "assets/gt_000.smpl"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    motions = tmp_path / "motions"
    motions.mkdir()
    for index, case_id in enumerate(("a", "b")):
        motion = np.zeros((4, 135), dtype=np.float32)
        motion[:, 0] = np.arange(4, dtype=np.float32) + index
        np.savez_compressed(motions / f"{case_id}.npz", motion_135=motion)

    output = tmp_path / "output"
    summary = add_method(
        manifest,
        manifest_root=source,
        method_key="new",
        method_label="New",
        accent="#123456",
        insert_after="gt",
        motion_dir=motions,
        output_dir=output,
        fps=30.0,
        stride=2,
    )

    assert summary["cases"] == 2
    assert summary["assets"] == 1
    written_manifest = json.loads((output / "manifest.json").read_text())
    assert [method["key"] for method in written_manifest["motion_methods"]] == [
        "gt",
        "new",
    ]
    payload = json.loads((output / "descriptors/000.json").read_text())
    assert all("new" in motions for motions in payload["motions"])
    assert (output / "assets/new_000.smpl").stat().st_size > 0


def test_add_method_can_build_one_descriptor_shard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    descriptors = source / "descriptors"
    descriptors.mkdir(parents=True)
    cases = [{"case_id": name} for name in ("a", "b", "c")]
    manifest = {
        "motion_methods": [{"key": "gt", "label": "GT", "accent": "#000"}],
        "cases": cases,
        "case_descriptor_chunks": {
            "size": 2,
            "path": "descriptors/{chunk}.json",
        },
    }
    for chunk, start, size in (("000", 0, 2), ("001", 2, 1)):
        (descriptors / f"{chunk}.json").write_text(
            json.dumps(
                {
                    "start": start,
                    "motions": [
                        {"gt": {"asset": f"assets/gt_{chunk}.smpl"}}
                        for _ in range(size)
                    ],
                }
            ),
            encoding="utf-8",
        )
    motions = tmp_path / "motions"
    motions.mkdir()
    for case in cases:
        np.savez_compressed(
            motions / f"{case['case_id']}.npz",
            motion_135=np.zeros((4, 135), dtype=np.float32),
        )

    output = tmp_path / "output"
    summary = add_method(
        manifest,
        manifest_root=source,
        method_key="new",
        method_label="New",
        accent="#123456",
        insert_after="gt",
        motion_dir=motions,
        output_dir=output,
        fps=30.0,
        stride=2,
        shard_index=1,
        num_shards=2,
    )

    assert summary["cases"] == 1
    assert summary["assets"] == 1
    assert not (output / "manifest.json").exists()
    assert not (output / "descriptors/000.json").exists()
    assert (output / "descriptors/001.json").is_file()
    assert (output / "assets/new_001.smpl").stat().st_size > 0
