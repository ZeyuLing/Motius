import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motius.motion.fbx._mapping import SMPL22_BONE_NAMES
from motius.motion.rigging import (
    SMPL22_RIG_NAMES,
    SMPL22_RIG_PARENTS,
    TemplateRiggingConfig,
    compute_skin_weights,
    fit_humanoid_skeleton,
)
from motius.motion.rigging.api import auto_rig_character
from motius.motion.skeleton.names import SMPL22_PARENTS


def _synthetic_t_pose(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def segment(start, end, radius: float, count: int) -> np.ndarray:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        centers = start + rng.random((count, 1)) * (end - start)
        return centers + rng.normal(size=(count, 3)) * radius

    return np.concatenate(
        [
            segment((0, 0, 0.48), (0, 0, 0.96), 0.075, 600),
            segment((0, 0, 0.79), (+0.50, 0, 0.79), 0.030, 320),
            segment((0, 0, 0.79), (-0.50, 0, 0.79), 0.030, 320),
            segment((+0.08, 0, 0.52), (+0.08, 0, 0.02), 0.040, 350),
            segment((-0.08, 0, 0.52), (-0.08, 0, 0.02), 0.040, 350),
        ],
        axis=0,
    )


def test_template_rig_matches_canonical_smpl22_contract() -> None:
    skeleton = fit_humanoid_skeleton(_synthetic_t_pose())
    assert skeleton.names == SMPL22_BONE_NAMES == SMPL22_RIG_NAMES
    np.testing.assert_array_equal(skeleton.parents, SMPL22_PARENTS)
    np.testing.assert_array_equal(SMPL22_RIG_PARENTS, SMPL22_PARENTS)
    assert skeleton.joints.shape == (22, 3)
    assert skeleton.diagnostics["inferred_pose"] == "T"
    assert skeleton.diagnostics["quality_score"] > 0.7
    assert not skeleton.diagnostics["warnings"]
    neck = skeleton.names.index("Neck")
    head = skeleton.names.index("Head")
    assert skeleton.joints[head, 1] < skeleton.joints[neck, 1]


def test_template_skin_weights_are_sparse_normalized_and_side_aware() -> None:
    vertices = _synthetic_t_pose()
    skeleton = fit_humanoid_skeleton(vertices)
    probes = np.vstack(
        [
            vertices,
            skeleton.joints[20] + (0.005, 0.0, 0.0),
            skeleton.joints[21] - (0.005, 0.0, 0.0),
        ]
    )
    result = compute_skin_weights(
        probes,
        skeleton,
        config=TemplateRiggingConfig(top_k=4, chunk_size=113),
    )
    assert result.weights.shape == (len(probes), 22)
    assert np.all(result.weights >= 0)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=2e-7)
    assert np.max(np.count_nonzero(result.weights, axis=1)) <= 4
    left_wrist = skeleton.names.index("L_Wrist")
    right_wrist = skeleton.names.index("R_Wrist")
    assert result.weights[-2, left_wrist] > result.weights[-2, right_wrist]
    assert result.weights[-1, right_wrist] > result.weights[-1, left_wrist]
    assert result.diagnostics["unbound_vertices"] == 0
    assert result.diagnostics["active_joints"] == 22


def test_terminal_regions_are_owned_by_terminal_joints() -> None:
    skeleton = fit_humanoid_skeleton(_synthetic_t_pose())
    names = skeleton.names
    probes = np.asarray(
        [
            skeleton.joints[names.index("Head")] + (0.0, 0.0, 0.03),
            skeleton.joints[names.index("L_Wrist")] + (0.03, 0.0, 0.0),
            skeleton.joints[names.index("R_Wrist")] - (0.03, 0.0, 0.0),
            skeleton.joints[names.index("L_Foot")] + (0.0, -0.02, 0.0),
            skeleton.joints[names.index("R_Foot")] + (0.0, -0.02, 0.0),
        ],
        dtype=np.float64,
    )
    expected = [
        names.index("Head"),
        names.index("L_Wrist"),
        names.index("R_Wrist"),
        names.index("L_Foot"),
        names.index("R_Foot"),
    ]
    result = compute_skin_weights(probes, skeleton)
    np.testing.assert_array_equal(np.argmax(result.weights, axis=1), expected)


def test_limb_segments_are_owned_by_the_joint_that_rotates_them() -> None:
    skeleton = fit_humanoid_skeleton(_synthetic_t_pose())
    names = skeleton.names
    probes = []
    expected = []
    for joint_name, child_name in (
        ("L_Hip", "L_Knee"),
        ("L_Knee", "L_Ankle"),
        ("L_Shoulder", "L_Elbow"),
        ("L_Elbow", "L_Wrist"),
    ):
        joint = names.index(joint_name)
        child = names.index(child_name)
        probes.append((skeleton.joints[joint] + skeleton.joints[child]) * 0.5)
        expected.append(joint)

    result = compute_skin_weights(np.asarray(probes), skeleton)
    np.testing.assert_array_equal(np.argmax(result.weights, axis=1), expected)


def test_template_fit_reports_narrow_silhouette_risk() -> None:
    vertices = _synthetic_t_pose().copy()
    vertices[:, 0] *= 0.22
    skeleton = fit_humanoid_skeleton(vertices)
    assert any("narrow" in warning for warning in skeleton.diagnostics["warnings"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 5}, "top_k"),
        ({"weight_falloff": 0}, "weight_falloff"),
        ({"side_penalty": 0}, "side_penalty"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_template_config_rejects_invalid_skinning_controls(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        TemplateRiggingConfig(**kwargs)


def test_auto_rig_api_builds_blender_job_and_reads_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    from motius.motion.rigging import api

    source = tmp_path / "character.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"test")
    output = tmp_path / "rigged.glb"
    captured = {}

    monkeypatch.setattr(api, "resolve_blender_executable", lambda value: blender)

    def fake_run(command, **kwargs):
        job_path = Path(command[-1])
        job = json.loads(job_path.read_text())
        captured.update(job)
        Path(job["output_path"]).write_bytes(b"glTF-test")
        manifest = {
            "method": "template",
            "armature_name": "Motius_SMPL22_Rig",
            "mesh_names": ["Body", "Eyes"],
            "joint_names": list(SMPL22_RIG_NAMES),
            "warnings": ["inspect wrists"],
        }
        Path(job["manifest_path"]).write_text(json.dumps(manifest))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    result = auto_rig_character(
        source,
        output,
        blender_executable=blender,
        up_axis="Y",
        top_k=3,
        weight_method="capsules",
        replace_existing_rig=True,
    )

    assert result.output_path == output.resolve()
    assert result.armature_name == "Motius_SMPL22_Rig"
    assert result.mesh_names == ("Body", "Eyes")
    assert result.warnings == ("inspect wrists",)
    assert captured["up_axis"] == "Y"
    assert captured["config"]["top_k"] == 3
    assert captured["weight_method"] == "capsules"
    assert captured["replace_existing_rig"] is True
    assert Path(captured["template_module"]).name == "template.py"


def test_auto_rig_api_rejects_bad_paths_formats_and_axis(tmp_path: Path) -> None:
    source = tmp_path / "character.glb"
    source.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="Unsupported rig output"):
        auto_rig_character(source, tmp_path / "rigged.obj")
    with pytest.raises(ValueError, match="declare their coordinate system"):
        auto_rig_character(source, tmp_path / "rigged.glb", up_axis="Y")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        auto_rig_character(tmp_path / "missing.glb", tmp_path / "rigged.glb")


def test_blender_backend_is_valid_python_and_covers_import_formats() -> None:
    root = Path(__file__).resolve().parents[1]
    backend = root / "motius/motion/rigging/_blender.py"
    text = backend.read_text()
    ast.parse(text)
    for suffix in (".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl"):
        assert repr(suffix) in text or f'"{suffix}"' in text


def test_real_public_mesh_demo_has_structural_and_visual_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    demo = root / "assets/motion/auto_rigging_demo"
    manifest = json.loads((demo / "manifest.json").read_text())
    unrigged = json.loads((demo / "unrigged_validation.json").read_text())
    rigged = json.loads((demo / "rigged_validation.json").read_text())
    animated = json.loads((demo / "animation_validation.json").read_text())
    diagnostics = json.loads((demo / "diagnostics.json").read_text())
    animation = json.loads((demo / "animation.json").read_text())

    assert manifest["source"]["license"] == "CC0 1.0"
    assert manifest["source"]["archive_sha256"] == (
        "46a912c0524072ac3b78c35d5d2471df7b8df102394a050ca8cd7184e3393648"
    )
    assert unrigged["verdict"] == "unrigged static mesh"
    assert not any(
        unrigged[key]
        for key in ("armatures", "armature_modifiers", "vertex_groups", "actions")
    )
    assert rigged["canonical_bones"] == 22
    assert rigged["unbound_vertices"] == 0
    assert rigged["max_influences"] <= 4
    assert animated["actions"]
    assert animated["max_vertex_deformation"] > 1e-4
    stretch = diagnostics["edge_stretch"]
    limits = manifest["artifacts"]["deformation_quality_limits"]
    for name, limit in limits.items():
        assert stretch[name] <= limit
    direction = animation["bone_direction_error"]
    assert direction["evaluated_frames"] == manifest["motion"]["frames"]
    for name, limit in manifest["artifacts"][
        "animation_direction_quality_limits"
    ].items():
        assert direction[name] <= limit
    assert (demo / manifest["artifacts"]["visual_qa"]).stat().st_size > 0
    for media in (manifest["artifacts"]["gif"], manifest["artifacts"]["mp4"]):
        assert (demo / media).stat().st_size > 0


def test_auto_rigging_demo_blender_scripts_are_valid_python() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tools/blender_extract_human_base_mesh.py",
        "tools/blender_validate_unrigged_asset.py",
        "tools/blender_retarget_smpl22_joints.py",
        "tools/blender_render_auto_rigging_demo.py",
        "tools/blender_render_rigging_diagnostics.py",
        "tools/build_auto_rigging_demo.py",
    ):
        ast.parse((root / relative).read_text(encoding="utf-8"))


def test_auto_rigging_docs_state_the_observable_orientation_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs/motion/rigging.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "Ankle -> Foot" in guide
    assert "not a sole plane" in guide
    assert "Neck -> Head" in guide
    assert "not the nose or gaze direction" in guide
    assert "rigging and deformation smoke test" in readme
