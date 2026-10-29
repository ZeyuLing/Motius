"""Prevent migrated methods from silently regaining an upstream application tree."""
import ast
from pathlib import Path

import pytest

from tools.audit_method_architecture import audit


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def architecture_report():
    return audit(ROOT)


@pytest.mark.parametrize("method", ["flowhmr", "mogendit", "mdm", "condmdi", "omnicontrol", "projflow", "tmr", "intergen", "intermask", "tm2d", "tm2t", "prism", "humanml3d_evaluator", "kimodo", "motionstreamer", "bailando", "momask", "maskcontrol", "flowmdm", "gvhmr", "mogents", "motiongpt", "motiongpt3", "unimumo", "vimogen", "dart", "gentrack", "vermo", "gem_smpl", "gem_x", "motionbricks", "prompthmr", "beyondmimic", "sonic", "protomotions", "motioncanvas", "animatecanvas", "hymotion_t2m", "motion_clip", "edge", "motionclr", "motionlcm", "mld", "motionmillion", "t2mgpt"])
def test_migrated_models_have_no_application_runtime(method, architecture_report):
    entry = next(item for item in architecture_report["methods"] if item["method"] == method)
    assert not entry["findings"], entry["findings"]
    for path in (ROOT / "motius/models" / method).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("motius.pipelines.", "motius.trainers.", "motius.datasets."))


def test_audit_reports_a_vendor_even_if_its_contents_look_like_a_network(tmp_path):
    directory = tmp_path / "motius/models/example/vendor"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text("class Network:\n    def forward(self, x): return x\n")
    findings = audit(tmp_path)["methods"][0]["findings"]
    assert any(item["kind"] == "non_network_file" for item in findings)


def test_audit_detects_training_and_subprocess_inside_a_network_directory(tmp_path):
    directory = tmp_path / "motius/models/example/network"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(
        "import subprocess\nclass Model:\n"
        "    def compute_loss(self):\n        subprocess.run(['python', 'demo.py'])\n"
        "    def forward_with_CFG(self): return None\n")
    findings = audit(tmp_path)["methods"][0]["findings"]
    assert {item["kind"] for item in findings} == {"external_runtime", "orchestration_in_models"}
    assert any(item["evidence"] == "forward_with_CFG" for item in findings)
