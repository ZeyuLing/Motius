from pathlib import Path

import pytest

from tools.build_temporal_case_explorer import condition_intervals


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


def test_smpl_gallery_uses_rigid_mesh_floor_alignment():
    page = GALLERY_TEMPLATE.read_text()

    assert "computeGroundOffset(view)" in page
    assert "view.groundOffset=Number.isFinite(meshMin)?-meshMin+.002:0" in page
    assert "view.motion.trans[t0+1]*scale[1]" in page
    assert "per-frame ground" not in page


def test_smpl_gallery_exposes_condition_colors_and_local_exports():
    page = GALLERY_TEMPLATE.read_text()

    assert "condition_intervals" in page
    assert "isConditionFrame(currentItem,frame)" in page
    assert 'data-format="npz"' in page
    assert 'data-format="fbx"' in page
    assert "motion_135.npy" in page
    assert "condition_mask.npy" in page
    assert "new FBXExporter().parseSync" in page
