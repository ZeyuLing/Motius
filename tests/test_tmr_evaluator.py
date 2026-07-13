import json
import re
from pathlib import Path

import numpy as np
import torch

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
    for representation in payload["representations"].values():
        assert len(representation["positions"]) == payload["frames"]
        assert np.isfinite(np.asarray(representation["positions"])).all()


def test_gmr_mesh_references_resolve_to_packaged_assets() -> None:
    xml_path = ROOT / "motius/motion/retarget/_gmr/assets/unitree_g1/g1_mocap_29dof.xml"
    xml = xml_path.read_text()
    meshdir = re.search(r'meshdir="([^"]+)"', xml).group(1)
    asset_dir = (xml_path.parent / meshdir).resolve()
    referenced = re.findall(r'file="([^"]+\.STL)"', xml, flags=re.IGNORECASE)
    assert len(referenced) == 35
    assert all((asset_dir / name).is_file() for name in referenced)
