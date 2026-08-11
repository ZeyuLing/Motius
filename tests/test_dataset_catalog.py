import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    path = ROOT / "tools" / "audit_datasets.py"
    spec = importlib.util.spec_from_file_location("audit_datasets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_catalog_and_cross_links_are_consistent() -> None:
    audit = _load_audit().run()
    assert audit.errors == []
