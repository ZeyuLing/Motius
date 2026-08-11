import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_leaderboard_catalog_is_complete_and_consistent() -> None:
    audit = _load_tool("audit_leaderboards").run()
    assert audit.errors == []


def test_leaderboard_navigation_is_synchronized() -> None:
    navigation = _load_tool("sync_leaderboard_navigation")
    benchmarks, catalog = navigation._load()
    changed = []
    for benchmark in benchmarks:
        source = benchmark["source"].split("#", 1)[0]
        path = ROOT / source / "index.html"
        if not path.is_file():
            continue
        original = path.read_text()
        if navigation._sync_page(
            path,
            benchmark,
            benchmarks,
            catalog["navigation_target"],
        ):
            changed.append(str(path.relative_to(ROOT)))
            path.write_text(original)
    assert changed == []


def test_leaderboard_navigation_is_safe_inside_huggingface_iframes() -> None:
    catalog = json.loads(
        (ROOT / "docs" / "leaderboards" / "catalog.json").read_text()
    )
    for benchmark in catalog["benchmarks"]:
        source = benchmark["source"].split("#", 1)[0]
        path = ROOT / source / "index.html"
        if not path.is_file():
            continue
        page = path.read_text()
        nav = page.split("<!-- motius-benchmark-nav:start -->", 1)[1].split(
            "<!-- motius-benchmark-nav:end -->", 1
        )[0]
        assert 'target="_top"' not in nav
        assert 'static.hf.space/" target="_self"' in nav
        assert 'motius-benchmark-brand' in nav
        assert 'target="_blank"' in nav


def test_g1_viewer_is_public_and_portable() -> None:
    catalog = json.loads(
        (ROOT / "docs" / "leaderboards" / "catalog.json").read_text()
    )
    benchmark = next(
        item
        for item in catalog["benchmarks"]
        if item["id"] == "text_to_motion_unitree_g1"
    )
    visual = benchmark["visualization"]
    assert visual["status"] == "partial"
    assert visual["population"] == 64
    assert visual["external_assets"] is True

    manifest_path = ROOT / visual["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["population"] == 64
    assert manifest["benchmark_population"] == 1024
    assert len(manifest["columns"]) == 3
    assert manifest["asset_base_url"].startswith("https://huggingface.co/")
    assert "/apdcephfs" not in manifest_path.read_text()


def test_g1_leaderboard_publishes_only_the_final_hymotion_checkpoint() -> None:
    result_path = (
        ROOT
        / "docs/leaderboards/hf_space_t2m_unitree_g1/g1_results.json"
    )
    payload = json.loads(result_path.read_text())
    generated = [
        row
        for row in payload["comparison_snapshot"]["rows"]
        if row["kind"] == "generated"
    ]
    hymotion = [row for row in generated if row["method"] == "HY-Motion G1"]

    assert len(generated) == 2
    assert len(hymotion) == 1
    assert hymotion[0]["variant"] == "released"
    assert hymotion[0]["fid"] == 0.0587378660855683
    assert "iter " not in result_path.read_text()

    catalog = json.loads(
        (ROOT / "docs/leaderboards/catalog.json").read_text()
    )
    setting = next(
        item
        for item in catalog["benchmarks"]
        if item["id"] == "text_to_motion_unitree_g1"
    )
    assert setting["metrics"]["generated_rows"] == len(generated)
    assert setting["visualization"]["method_coverage"] == "GT + 2/2 generated rows"

    page = (
        ROOT / "docs/leaderboards/hf_space_t2m_unitree_g1/index.html"
    ).read_text()
    assert '["released", "reference"].includes(row.variant)' in page
