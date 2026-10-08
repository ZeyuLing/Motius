import numpy as np
import pytest

from motius.motion.retarget.smpl_soma import SOMA30_IN_SOMA77
from tools.generate_kimodo_native_demos import _joints as _kimodo_native_joints
from tools.render_model_card_threejs_media import (
    _jobs,
    _music_to_dance_jobs,
    _native_jobs,
    _native_job_specs,
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
    assert all(job.layout == "tile" for job in jobs)
    assert all(
        job.representation == "smpl-plus-native-skeleton"
        for job in jobs
    )


def test_native_jobs_do_not_claim_an_unvalidated_smpl_bridge():
    jobs = _native_job_specs()
    assert len(jobs) >= 15
    ardy = [job for job in jobs if job.method == "ardy"]
    assert ardy
    assert all(job.representation == "ardy-330-native-mesh" for job in ardy)
    assert all(job.fps == 20 for job in ardy)


def test_temporal_previews_prefer_native_viewers():
    jobs = {job.method: job for job in _native_job_specs() if "temporal" in job.source}
    for method in {"kimodo", "maskcontrol", "motionstreamer", "omnicontrol", "prism"}:
        assert method in jobs
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


def test_native_jobs_only_schedule_published_sources_with_local_viewers(tmp_path, monkeypatch):
    import json
    from tools import render_model_card_threejs_media as media

    monkeypatch.setattr(media, "ROOT", tmp_path)
    specs = media._native_job_specs()
    published, missing_viewer, unpublished = specs[:3]
    for job in (published, unpublished):
        job.viewer.parent.mkdir(parents=True, exist_ok=True)
        job.viewer.write_text("<html></html>")
    attachments = tmp_path / "attachments.json"
    attachments.write_text(json.dumps({"videos": {
        published.source: {}, missing_viewer.source: {},
    }}))
    monkeypatch.setattr(media, "ATTACHMENTS", attachments)
    assert media._native_jobs() == [published]


@pytest.mark.parametrize("representation, expected", [("smpl", "rendered"), ("native-joints", "cached")])
def test_render_cache_requires_the_requested_representation(tmp_path, monkeypatch, representation, expected):
    from tools import render_model_card_threejs_media as media

    job = media.RenderJob(source="assets/model_zoo/demo/preview.gif", method="demo", label="Demo", case_id="case", representation="native-joints")
    output = job.output(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"cached video")
    output.with_suffix(".render.json").write_text("{}")
    monkeypatch.setattr(media, "_audit_one", lambda *args: (None, {
        "method": "demo", "case_id": "case", "fps": 30,
        "representation": representation,
    }))
    commands = []
    monkeypatch.setattr(media.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert media._render(job, tmp_path, False, False) == (job.source, expected)
    assert bool(commands) == (expected == "rendered")
