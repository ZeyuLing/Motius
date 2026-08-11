import argparse
import json
from pathlib import Path

import numpy as np

from tools.build_native_mesh_viewer import build


ROOT = Path(__file__).resolve().parents[1]


def test_build_native_mesh_viewer_aligns_floor_and_preserves_clock(tmp_path):
    source = tmp_path / "motion.npz"
    vertices = np.array(
        [
            [[0.0, -0.2, 0.0], [1.0, 0.8, 0.0], [0.0, -0.2, 1.0]],
            [[0.1, -0.1, 0.0], [1.1, 0.9, 0.0], [0.1, -0.1, 1.0]],
        ],
        dtype=np.float32,
    )
    np.savez(
        source,
        core_vertices=vertices,
        core_faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    output = tmp_path / "viewer"
    viewer = build(
        argparse.Namespace(
            input=[source],
            output_dir=output,
            vertices_key="core_vertices",
            faces_key="core_faces",
            faces_input=None,
            trajectory_key=None,
            trajectory_joint=0,
            condition_root_key=None,
            segment_frames_key=None,
            label="Native",
            case="case",
            caption="walk",
            segment_caption=[],
            representation="native",
            fps=20,
        )
    )
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert viewer == output / "index.html"
    assert manifest["cases"][0]["frames"] == 2
    assert manifest["cases"][0]["fps"] == 20
    assert manifest["representation"] == "native"
    assert (output / "vertices.u16").is_file()


def test_build_native_mesh_viewer_preserves_joint_trajectory_height(tmp_path):
    source = tmp_path / "motion.npz"
    vertices = np.array(
        [
            [[0.0, -0.2, 0.0], [1.0, 0.8, 0.0], [0.0, -0.2, 1.0]],
            [[0.1, -0.1, 0.0], [1.1, 0.9, 0.0], [0.1, -0.1, 1.0]],
        ],
        dtype=np.float32,
    )
    condition_joints = np.zeros((2, 2, 3), dtype=np.float32)
    condition_joints[:, 1, 1] = [1.0, 1.2]
    np.savez(
        source,
        vertices=vertices,
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        condition_joints=condition_joints,
    )
    output = tmp_path / "viewer"
    build(
        argparse.Namespace(
            input=[source],
            output_dir=output,
            vertices_key="vertices",
            faces_key="faces",
            faces_input=None,
            trajectory_key="condition_joints",
            trajectory_joint=1,
            condition_root_key=None,
            segment_frames_key=None,
            label="Native",
            case="case",
            caption="wave",
            segment_caption=[],
            representation="native",
            fps=20,
        )
    )
    item = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )["cases"][0]
    assert item["trajectory_on_floor"] is False
    assert np.allclose(
        np.asarray(item["trajectory"])[:, 1],
        [1.2, 1.4],
    )


def test_native_viewers_render_conditions_as_visible_3d_tubes():
    for template in (
        ROOT / "tools/templates/native_mesh_viewer.html",
        ROOT / "tools/templates/native_skeleton_viewer.html",
    ):
        text = template.read_text(encoding="utf-8")
        assert "new THREE.TubeGeometry(" in text
        assert "new THREE.Line(" not in text


def test_native_skeleton_viewer_tracks_root_without_resetting_orbit():
    text = (
        ROOT / "tools/templates/native_skeleton_viewer.html"
    ).read_text(encoding="utf-8")

    assert "item.skeleton.parents.findIndex(parent => parent < 0)" in text
    assert "cameraOffset.copy(camera.position).sub(controls.target)" in text
    assert "camera.position.copy(focus).add(cameraOffset)" in text
