from pathlib import Path

from tools.build_3dpw_monocular_shard_plan import (
    SCHEMA_REVISION,
    build_shard_plan,
)


def _artifact(sequence_id: str) -> dict:
    return {
        "public_manifest": {
            "metadata": {
                "sequence_id": sequence_id,
            }
        }
    }


def test_shard_plan_balances_person_frames_and_skips_completed(tmp_path: Path):
    videos = {
        "videos": [
            {"sequence_id": "a", "frames": 100},
            {"sequence_id": "b", "frames": 80},
            {"sequence_id": "c", "frames": 60},
            {"sequence_id": "d", "frames": 40},
        ]
    }
    index = {
        "artifacts": [
            _artifact("a"),
            _artifact("a"),
            _artifact("b"),
            _artifact("c"),
            _artifact("c"),
            _artifact("d"),
        ]
    }
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "b.motius.npz").touch()

    plan = build_shard_plan(
        video_manifest=videos,
        ground_truth_index=index,
        prediction_dir=prediction_dir,
        num_shards=2,
    )

    assert plan["schema_revision"] == SCHEMA_REVISION
    assert plan["completed_at_plan_time"] == 1
    assert plan["pending"] == 3
    assert sorted(
        sequence_id
        for assigned in plan["assignments"].values()
        for sequence_id in assigned
    ) == ["a", "c", "d"]
    assert plan["estimated_person_frames"] == [200, 160]


def test_shard_plan_honors_max_sequences(tmp_path: Path):
    videos = {
        "videos": [
            {"sequence_id": "a", "frames": 100},
            {"sequence_id": "b", "frames": 80},
        ]
    }
    index = {"artifacts": [_artifact("a"), _artifact("b")]}

    plan = build_shard_plan(
        video_manifest=videos,
        ground_truth_index=index,
        prediction_dir=tmp_path,
        num_shards=2,
        max_sequences=1,
    )

    assert plan["population"] == 1
    assert plan["pending"] == 1
    assert plan["assignments"] == {"0": ["a"], "1": []}
