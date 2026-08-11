import numpy as np
import pytest

from motius.motion.retarget.smpl_soma import SOMA30_IN_SOMA77
from tools.generate_kimodo_native_demos import _joints as _kimodo_native_joints
from tools.render_model_card_threejs_media import (
    _jobs,
    _music_to_dance_jobs,
    _native_jobs,
    _sidecar_jobs,
    _t2m_jobs,
)


def test_render_sources_are_unique():
    jobs = _jobs()
    assert len(jobs) == len({job.source for job in jobs})


def test_t2m_jobs_use_full_frame_threejs_capture():
    jobs = _t2m_jobs()
    assert len(jobs) >= 60
    assert all(job.viewer.name == "index.html" for job in jobs)
    assert all(job.viewer.is_file() for job in jobs)
    assert not any(job.include_audio for job in jobs)


def test_task_sidecars_resolve_to_leaderboard_viewers():
    jobs = _sidecar_jobs()
    assert len(jobs) >= 9
    assert all(job.case_id and job.method and job.viewer for job in jobs)


def test_music_to_dance_jobs_require_audio_and_native_overlay():
    jobs = _music_to_dance_jobs()
    assert len(jobs) == 6
    assert all(job.include_audio for job in jobs)
    assert all(job.layout == "stage" for job in jobs)
    assert all(
        job.representation == "smpl-plus-native-skeleton"
        for job in jobs
    )


def test_native_jobs_do_not_claim_an_unvalidated_smpl_bridge():
    jobs = _native_jobs()
    assert len(jobs) >= 15
    assert all(job.viewer.is_file() for job in jobs)
    ardy = [job for job in jobs if job.method == "ardy"]
    assert ardy
    assert all(job.representation == "ardy-330-native-mesh" for job in ardy)
    assert all(job.fps == 20 for job in ardy)


def test_temporal_previews_prefer_native_viewers():
    jobs = {job.method: job for job in _native_jobs() if "temporal" in job.source}
    for method in {"kimodo", "maskcontrol", "motionstreamer", "omnicontrol", "prism"}:
        if method not in jobs:
            continue
        assert "model_card_native_viewers" in str(jobs[method].viewer)
        assert "native" in jobs[method].representation


def test_kimodo_native_preview_subsets_expanded_soma77_joints():
    joints77 = np.arange(2 * 77 * 3, dtype=np.float32).reshape(2, 77, 3)
    actual = _kimodo_native_joints({"posed_joints": joints77[None]})

    np.testing.assert_array_equal(actual, joints77[:, SOMA30_IN_SOMA77])


def test_kimodo_native_preview_rejects_unknown_topology():
    with pytest.raises(ValueError, match="SOMA-30 or expanded SOMA-77"):
        _kimodo_native_joints(
            {"posed_joints": np.zeros((2, 31, 3), dtype=np.float32)}
        )
