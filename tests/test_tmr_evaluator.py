import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from motius.evaluation.metrics import aggregate_t2m_metrics, r_precision
from motius.models.tmr import TMRBundle


ROOT = Path(__file__).resolve().parents[1]


def test_tmr_bundle_encodes_text_and_motion() -> None:
    bundle = TMRBundle(
        motion_nfeats=38,
        text_nfeats=16,
        arch={
            "latent_dim": 32,
            "ff_size": 64,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
        },
    ).eval()
    mask = torch.tensor([[True, True, True], [True, True, False]])
    motion = {"x": torch.randn(2, 3, 38), "mask": mask}
    text = {"x": torch.randn(2, 3, 16), "mask": mask}
    assert bundle.encode_motion(motion).shape == (2, 32)
    assert bundle.encode_text(text).shape == (2, 32)


def test_t2m_metrics_report_perfect_aligned_retrieval() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    precision, matching = r_precision(embeddings, embeddings, top_k=3)
    np.testing.assert_array_equal(precision, [4, 4, 4])
    assert matching == 0.0
    metrics = aggregate_t2m_metrics(
        embeddings,
        embeddings,
        embeddings,
        n_repeats=1,
        chunk=4,
    )
    assert metrics["r_precision"] == [1.0, 1.0, 1.0]
    assert abs(metrics["fid"]) < 1e-6


def test_representation_demo_contains_synchronized_routes() -> None:
    source = (ROOT / "assets/motion/representation_demo/data.js").read_text()
    payload = json.loads(source.removeprefix("window.MOTIUS_REPRESENTATION_DEMO=").removesuffix(";\n"))
    assert payload["case_id"] == "004822"
    assert payload["fps"] == 30.0
    assert payload["frames"] == 180
    assert set(payload["representations"]) == {"humanml3d", "smpl", "g1"}
    hml = payload["representations"]["humanml3d"]
    smpl = payload["representations"]["smpl"]
    g1 = payload["representations"]["g1"]
    assert len(hml["positions"]) == payload["frames"]
    assert np.isfinite(np.asarray(hml["positions"])).all()
    for representation in (hml, smpl, g1):
        np.testing.assert_allclose(representation["initial_forward"], [0, 0, 1], atol=1e-5)
    assert g1["forward_basis"] == "MuJoCo pelvis local +X axis"

    asset_dir = ROOT / "assets/motion/representation_demo"
    assert (asset_dir / smpl["vertices_file"]).stat().st_size == (
        payload["frames"] * smpl["vertex_count"] * 3 * 2
    )
    assert (asset_dir / smpl["normals_file"]).stat().st_size == (
        payload["frames"] * smpl["vertex_count"] * 3
    )
    assert (asset_dir / smpl["indices_file"]).stat().st_size == smpl["index_count"] * 4
    assert (asset_dir / g1["vertices_file"]).stat().st_size == g1["vertex_count"] * 3 * 4
    assert (asset_dir / g1["indices_file"]).stat().st_size == g1["index_count"] * 4
    assert (asset_dir / g1["transforms_file"]).stat().st_size == (
        payload["frames"] * g1["geom_count"] * 7 * 4
    )

    viewer = (asset_dir / "index.html").read_text()
    assert "One motion, three representations" not in viewer
    assert "HumanML3D test" not in viewer


def test_two_person_representation_demo_uses_gt_retarget_preview() -> None:
    readme = (ROOT / "README.md").read_text()
    section = readme.split("### Two-Person Representation Demo", 1)[1].split("\n### ", 1)[0]
    assert "interx_smplh_gt_G012T003A016R008_skeleton_smpl_mesh.gif" in section
    assert "assets/motion/interhuman_representation_demo/index.html" in section
    assert "Three.js viewer" in section
    assert "assets/model_zoo/intergen" not in section
    assert "assets/model_zoo/intermask" not in section
    representation_doc = (ROOT / "docs/motion/representations.md").read_text()
    assert "model-generation demo" in representation_doc
    assert "smpl_pair_vertices.u16" in representation_doc

    metadata = json.loads(
        (
            ROOT
            / "assets/motion/interhuman_representation_demo/interx_smplh_gt_G012T003A016R008_skeleton_smpl_mesh.json"
        ).read_text()
    )
    assert metadata["sample_id"] == "G012T003A016R008"
    assert metadata["source"] == "GT InterX smplh_52_2p/P1+P2"
    assert metadata["fps"] == 30
    assert metadata["frames"] == 72
    assert metadata["fit_mpjpe_mm"] == 0.0
    assert "InterX SMPL-H GT" in metadata["route"]
    assert metadata["threejs_viewer"] == "assets/motion/interhuman_representation_demo/index.html"
    assert (ROOT / metadata["gif"]).is_file()
    gif = Image.open(ROOT / metadata["gif"])
    assert gif.size == (1024, 576)
    assert getattr(gif, "n_frames", 1) == metadata["frames"]
    capture_script = (ROOT / "tools/capture_threejs_demo_gif.py").read_text()
    assert "window.__MOTIUS_DEMO__.setFrame" in capture_script

    asset_dir = ROOT / "assets/motion/interhuman_representation_demo"
    data_source = (asset_dir / "data.js").read_text()
    payload = json.loads(
        data_source.removeprefix("window.MOTIUS_TWO_PERSON_REPRESENTATION_DEMO=").removesuffix(";\n")
    )
    assert payload["sample_id"] == "G012T003A016R008"
    assert payload["fps"] == 30
    assert payload["frames"] == 72
    assert set(payload["representations"]) == {"interhuman", "smpl"}
    interhuman = payload["representations"]["interhuman"]
    smpl = payload["representations"]["smpl"]
    assert interhuman["person_count"] == 2
    assert smpl["person_count"] == 2
    assert len(interhuman["positions"]) == payload["frames"]
    interhuman_positions = np.asarray(interhuman["positions"])
    assert np.isfinite(interhuman_positions).all()
    np.testing.assert_allclose(
        interhuman_positions[0].reshape(-1, 3).mean(axis=0)[[0, 2]],
        [0.0, 0.0],
        atol=1e-4,
    )
    assert (asset_dir / smpl["vertices_file"]).stat().st_size == (
        payload["frames"] * smpl["person_count"] * smpl["vertex_count"] * 3 * 2
    )
    assert (asset_dir / smpl["normals_file"]).stat().st_size == (
        payload["frames"] * smpl["person_count"] * smpl["vertex_count"] * 3
    )
    assert (asset_dir / smpl["indices_file"]).stat().st_size == smpl["index_count"] * 4
    viewer = (asset_dir / "index.html").read_text()
    assert "MOTIUS_TWO_PERSON_REPRESENTATION_DEMO" in viewer
    assert "GridHelper" in viewer
    assert "PlaneGeometry" in viewer
    assert "makeSkeletonPair(-1.35)" in viewer
    assert "makeSMPLPair(1.35)" in viewer


def test_gmr_mesh_references_resolve_to_packaged_assets() -> None:
    xml_path = ROOT / "motius/motion/retarget/_gmr/assets/unitree_g1/g1_mocap_29dof.xml"
    xml = xml_path.read_text()
    meshdir = re.search(r'meshdir="([^"]+)"', xml).group(1)
    asset_dir = (xml_path.parent / meshdir).resolve()
    referenced = re.findall(r'file="([^"]+\.STL)"', xml, flags=re.IGNORECASE)
    assert len(referenced) == 35
    assert all((asset_dir / name).is_file() for name in referenced)
