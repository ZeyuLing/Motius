import ast
import hashlib
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


def test_auto_rig_api_runs_mia_and_normalizes_to_smpl22(
    tmp_path: Path, monkeypatch
) -> None:
    from motius.motion.rigging import api

    source = tmp_path / "stylized.glb"
    source.write_bytes(b"glTF")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"test")
    output = tmp_path / "rigged.fbx"
    captured = {}

    monkeypatch.setattr(api, "resolve_blender_executable", lambda value: blender)

    def fake_mia(character, raw_fbx, **kwargs):
        assert Path(character) == source.resolve()
        Path(raw_fbx).write_bytes(b"FBX-MIA")
        return {
            "name": "Make-It-Animatable",
            "space": kwargs["space"],
            "rest_pose": kwargs["rest_pose"],
            "network_upload": True,
        }

    def fake_run(command, **kwargs):
        assert Path(command[command.index("--python") + 1]).name == "_blender_mia.py"
        job = json.loads(Path(command[-1]).read_text())
        captured.update(job)
        Path(job["output_path"]).write_bytes(b"FBX-normalized")
        Path(job["manifest_path"]).write_text(
            json.dumps(
                {
                    "method": "make_it_animatable",
                    "armature_name": "Motius_SMPL22_Rig",
                    "mesh_names": ["Body"],
                    "joint_names": list(SMPL22_RIG_NAMES),
                    "warnings": [],
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(api, "request_make_it_animatable", fake_mia)
    monkeypatch.setattr(api.subprocess, "run", fake_run)
    result = auto_rig_character(
        source,
        output,
        method="mia",
        blender_executable=blender,
        mia_space="trusted/mia",
        mia_rest_pose="A-pose",
    )

    assert result.method == "make_it_animatable"
    assert result.joint_names == SMPL22_RIG_NAMES
    assert captured["method"] == "make_it_animatable"
    assert captured["backend"]["space"] == "trusted/mia"
    assert captured["backend"]["rest_pose"] == "A-pose"
    assert captured["bone_mapping"]["Hips"] == "Pelvis"


def test_blender_backend_is_valid_python_and_covers_import_formats() -> None:
    root = Path(__file__).resolve().parents[1]
    backend = root / "motius/motion/rigging/_blender.py"
    text = backend.read_text()
    ast.parse(text)
    for suffix in (".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl"):
        assert repr(suffix) in text or f'"{suffix}"' in text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multi_character_demo_has_structural_and_visual_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    demo = root / "assets/motion/auto_rigging_demo"
    manifest = json.loads((demo / "manifest.json").read_text())
    validation = json.loads((demo / "validation.json").read_text())
    render = json.loads((demo / "render.json").read_text())

    assert manifest["autorig_backend"]["motius_method"] == "make_it_animatable"
    assert len(manifest["characters"]) == 3
    assert {item["role"] for item in manifest["characters"]} == {
        "child",
        "big_head",
        "high_weight",
    }
    assert any("BY-NC" in item["license"] for item in manifest["characters"])
    assert manifest["motion"]["synchronized_across_characters"] is True
    assert manifest["motion"]["frames"] == 150

    for character in validation["characters"]:
        assert not any(
            character["input"][key]
            for key in ("armatures", "armature_modifiers", "vertex_groups", "actions")
        )
        assert character["input"]["packed_texture_images"] > 0
        assert character["rig"]["canonical_bones"] == 22
        assert character["rig"]["unbound_vertices"] == 0
        assert character["rig"]["max_influences"] <= 4
        assert character["animation"]["frames"] == 150
        assert character["animation"]["max_sampled_vertex_deformation"] > 1e-4

    assert render["resolution"] == [960, 540]
    assert render["presentation"]["text_overlays"] is False
    assert render["presentation"]["authored_textures"] is True
    assert render["presentation"]["skeleton_overlay"] is True
    assert render["presentation"]["joints_per_character"] == 22

    for name, hash_name in (("mp4", "mp4_sha256"), ("poster", "poster_sha256")):
        media = demo / manifest["artifacts"][name]
        assert media.stat().st_size > 0
        assert _sha256(media) == manifest["artifacts"][hash_name]


def test_auto_rigging_demo_blender_scripts_are_valid_python() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tools/blender_extract_human_base_mesh.py",
        "tools/blender_validate_unrigged_asset.py",
        "tools/blender_retarget_smpl22_joints.py",
        "tools/blender_render_auto_rigging_demo.py",
        "tools/blender_render_rigging_diagnostics.py",
        "tools/blender_render_textured_autorig_video.py",
        "tools/build_auto_rigging_demo.py",
        "motius/motion/rigging/_blender_mia.py",
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
    assert "Make-It-Animatable" in guide
    assert "public Space receives the uploaded character file" in guide
