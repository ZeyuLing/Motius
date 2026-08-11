from pathlib import Path

import pytest

from tools.build_temporal_case_explorer import (
    condition_intervals,
    display_references,
)
from tools.smpl_gallery_assets import load_motion135, write_chunked_manifest
from tools.build_smpl_motion_gallery import (
    load_motion_record,
    load_motion_record_with_retry,
    load_skeleton_record,
    parse_asset_bases,
)
from tools.build_body_part_case_explorer import (
    condition_atoms_from_mask,
    condition_intervals_from_mask,
)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GALLERY_TEMPLATE = ROOT / "tools" / "leaderboard_smpl_gallery.html"


@pytest.mark.parametrize(
    ("setting", "length", "expected"),
    [
        ("start_1f", 100, [[0, 1]]),
        ("pre20", 101, [[0, 20]]),
        ("pre20_uncond", 101, [[0, 20]]),
        ("both_1f", 100, [[0, 1], [99, 100]]),
        ("both_1f", 1, [[0, 1]]),
        ("mid80", 100, [[0, 10], [90, 100]]),
        ("mid80_uncond", 101, [[0, 10], [91, 101]]),
    ],
)
def test_temporal_condition_intervals(setting, length, expected):
    assert condition_intervals(setting, length) == expected


def test_adaptive_keyframes_are_rescaled_to_display_frames():
    assert condition_intervals(
        "adaptive_keyframes",
        51,
        case_id="case",
        keyframes={
            "case": {
                "T": 101,
                "keyframe_indices": [0, 50, 100],
            }
        },
    ) == [[0, 1], [25, 26], [50, 51]]


def test_display_references_replace_stale_condition_label():
    references = ["a person walks", "Condition: Prediction: first frame"]

    assert display_references(references, "Keyframe: adaptive sparse frames") == [
        "a person walks",
        "Condition: Keyframe: adaptive sparse frames",
    ]


def test_chunked_manifest_moves_only_motion_descriptors(tmp_path):
    manifest = {
        "cases": [
            {"case_id": "a", "references": ["walk"], "motions": {"gt": {"frames": 10}}},
            {"case_id": "b", "references": ["run"], "motions": {"gt": {"frames": 20}}},
        ]
    }

    write_chunked_manifest(tmp_path, manifest, chunk_size=1)

    assert manifest["schema_version"] == 3
    assert manifest["cases"][0] == {"case_id": "a", "references": ["walk"]}
    assert (tmp_path / "descriptors" / "000.json").read_text().startswith(
        '{"start":0,"motions":[{"gt":{"frames":10}}]}'
    )


def test_body_part_condition_intervals_are_recovered_from_source_mask():
    mask = np.ones((8, 12), dtype=np.float32)
    mask[0, :3] = 0
    mask[3:5, 4:7] = 0
    mask[7, 1] = 0

    assert condition_intervals_from_mask(mask) == [[0, 1], [3, 5], [7, 8]]


def test_body_part_condition_atoms_decode_rotation_and_position_blocks():
    mask = np.ones((8, 198), dtype=np.float32)
    mask[[0, 7], 20 * 6 : 21 * 6] = 0
    position_start = 22 * 6 + 20 * 3
    mask[[0, 7], position_start : position_start + 3] = 0

    assert condition_atoms_from_mask(mask) == {
        "rotation_joint_indices": [20],
        "position_joint_indices": [20],
        "position_axes": ["x", "y", "z"],
    }


def test_motion_gallery_retry_preserves_valid_records(tmp_path):
    motion = np.zeros((8, 135), dtype=np.float32)
    motion[:, 3:135:6] = 1.0
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, motion_135=motion)

    loaded, fit = load_motion_record_with_retry(path, max_frames=6)

    assert fit is None
    assert loaded.shape == (6, 135)


def test_motion_gallery_accepts_per_method_asset_bases():
    assert parse_asset_bases(
        ["gt=https://example.test/gt"], {"gt", "method"}
    ) == {"gt": "https://example.test/gt/"}


def test_smpl_gallery_uses_rigid_mesh_floor_alignment():
    page = GALLERY_TEMPLATE.read_text()

    assert "computeGroundOffset(view)" in page
    assert "view.groundOffset=Number.isFinite(meshMin)?-meshMin+.002:0" in page
    assert "view.motion.trans[t0+1]*scale[1]" in page
    assert "per-frame ground" not in page


def test_smpl_gallery_overlays_native_skeleton_on_matching_mesh():
    page = GALLERY_TEMPLATE.read_text()

    assert 'views=manifest.motion_methods.map((method,index)=>makeView(method,index))' in page
    assert 'skeletonMethod=(manifest.skeleton_methods||[]).find(value=>value.key===method.key)' in page
    assert "alignSkeletonToMesh(view)" in page
    assert "meshRoot.x-nativeRoot[0]" in page
    assert "computeSkeletonGroundOffset" not in page
    assert "-minimum+.04" not in page


def test_gallery_uses_npz_native_skeleton_fps(tmp_path):
    path = tmp_path / "edge.npz"
    joints = np.arange(6 * 24 * 3, dtype=np.float32).reshape(6, 24, 3)
    np.savez(path, joints=joints, fps=np.float32(30.0))

    loaded = load_skeleton_record(
        path,
        source_fps=60.0,
        target_fps=30.0,
        target_frames=6,
    )

    np.testing.assert_array_equal(loaded, joints)


def test_gallery_loads_motionstreamer_272_through_public_conversion(tmp_path):
    from motius.motion.representation.motion272 import motion135_to_272

    motion = np.zeros((5, 135), dtype=np.float32)
    identity_6d = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
    motion[:, 3:] = np.tile(identity_6d, 22)
    source = tmp_path / "motionstreamer.npz"
    np.savez(source, motion_272=motion135_to_272(motion))

    loaded = load_motion135(source)

    assert loaded.shape == motion.shape
    np.testing.assert_allclose(loaded[:, [0, 2]], motion[:, [0, 2]], atol=1e-5)
    np.testing.assert_allclose(loaded[:, 3:], motion[:, 3:], atol=1e-5)
    assert np.isfinite(loaded[:, 1]).all()


def test_gallery_motion_loader_keeps_full_clip_without_source_descriptor(tmp_path):
    motion = np.zeros((17, 135), dtype=np.float32)
    motion[:, 3:] = np.tile(
        np.array([1, 0, 0, 1, 0, 0], dtype=np.float32), 22
    )
    source = tmp_path / "full.npz"
    np.savez(source, motion_135=motion)

    loaded, fit_mean = load_motion_record(source, max_frames=None)

    assert loaded.shape == (17, 135)
    assert fit_mean is None


def test_smpl_gallery_exposes_condition_colors_and_local_exports():
    page = GALLERY_TEMPLATE.read_text()

    assert "condition_intervals" in page
    assert "isConditionFrame(currentItem,frame)" in page
    assert "buildConditionMarkers" in page
    assert "condition.position_joint_indices" in page
    assert 'data-format="npz"' in page
    assert 'data-format="fbx"' in page
    assert "motion_135.npy" in page
    assert "condition_mask.npy" in page
    assert "new FBXExporter().parseSync" in page
    assert 'headers:{Range:' in page
    assert 'await import("fflate")' in page
    assert "views.slice(0,eager)" in page
    assert "mesh.frustumCulled=false" in page


def test_smpl_gallery_retries_throttled_assets_without_poisoning_cache():
    page = GALLERY_TEMPLATE.read_text()

    assert "response.status===429||response.status>=500" in page
    assert "retry-after" in page
    assert "MAX_FETCHES=1" in page
    assert "MIN_FETCH_INTERVAL_MS=350" in page
    assert "fetchWithSlot(path,request)" in page
    assert 'cache:attempt===0?(init.cache||"default"):"reload"' in page
    assert "if(assetCache.get(key)===pending)assetCache.delete(key)" in page
    assert "Promise.allSettled" in page


def test_smpl_gallery_falls_back_to_cached_full_asset_after_range_failure():
    page = GALLERY_TEMPLATE.read_text()

    assert "async function fetchAssetSlice(path,start,end)" in page
    assert 'headers:{Range:`bytes=${start}-${end-1}`}' in page
    assert "await fetchBuffer(path),partial:false" in page
    assert "loaded=await fetchAssetSlice(path,start,end)" in page
    assert "Range request failed; using cached full asset" in page


def test_smpl_gallery_lazily_hydrates_and_isolates_failed_tiles():
    page = GALLERY_TEMPLATE.read_text()

    assert "new IntersectionObserver" in page
    assert 'rootMargin:"80px 0px"' in page
    assert "for(const view of views.slice(eager))observer.observe(view.tile)" in page
    assert 'view.tile.dataset.loadState="error"' in page
    assert 'view.retryButton.hidden=false' in page
    assert "Motion load failed for" in page
    assert "for(let start=eager" not in page


def test_smpl_gallery_applies_low_rank_smpl_pose_correctives():
    page = GALLERY_TEMPLATE.read_text()

    assert "meta.pose_correctives" in page
    assert "THREE.HalfFloatType" in page
    assert "motiusPoseCorrective" in page
    assert "updatePoseCorrectives(view)" in page
    assert "applyPoseCorrective(view,vertex,position)" in page
