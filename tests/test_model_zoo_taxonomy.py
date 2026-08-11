import json
from pathlib import Path

from PIL import Image

from tools.audit_model_zoo_release import (
    DEMO_END,
    DEMO_START,
    TASK_LABELS,
    TASK_REGISTRY,
    RESTRICTED_TASK_CELL,
    UNREGISTERED_TASK_CELL,
    _demo_status,
    _parse_task_entries,
    _read_model_rows,
    _task_status,
)
from tools.hf_checkpoint_specs import CHECKPOINT_EXEMPTIONS, CHECKPOINT_SPECS


ROOT = Path(__file__).resolve().parents[1]


def test_model_zoo_uses_canonical_task_labels() -> None:
    rows = _read_model_rows()
    assert len(rows) == 41

    for row in rows:
        card_text = row.card_path.read_text()
        status, note = _task_status(row.task_cell, card_text)
        expected = {
            "MotionBricks": "unregistered",
            "PromptHMR-Video": "restricted",
        }.get(row.method, "valid")
        assert status == expected, f"{row.method}: {note}"


def test_model_zoo_catalog_covers_every_public_artifact_package() -> None:
    rows = _read_model_rows()
    catalog_cards = {row.card_path.stem for row in rows}
    artifact_packages = {
        spec.pipeline_class.split(".")[-2] for spec in CHECKPOINT_SPECS
    }

    catalog_exemptions = {
        package
        for package in CHECKPOINT_EXEMPTIONS
        if (ROOT / "docs/model_zoo" / f"{package}.md").is_file()
    }
    assert catalog_cards == artifact_packages | catalog_exemptions

    motioncanvas = next(
        row for row in rows if row.card_path.name == "motioncanvas.md"
    )
    assert motioncanvas.method == "HYMotion M2M (MotionCanvas)"


def test_release_audit_requires_one_visible_demo_per_supported_task() -> None:
    video = (
        '<video src="https://github.com/user-attachments/assets/'
        '12345678-1234-1234-1234-123456789abc" controls></video>'
    )
    one_demo = f"""
{DEMO_START}
| Task | Input / condition | Rendered output | More |
| --- | --- | --- | --- |
| Monocular Motion Capture | Monocular RGB video | {video} | — |
{DEMO_END}
"""

    assert _demo_status(one_demo, "Monocular Motion Capture")[0] == "present"
    assert (
        _demo_status(one_demo, "Text-to-Motion, Motion-to-Text")[0]
        == "missing"
    )


def test_prompthmr_is_explicitly_restricted_and_not_task_indexed() -> None:
    rows = {row.method: row for row in _read_model_rows()}
    row = rows["PromptHMR-Video"]
    task_index = (ROOT / "docs/model_zoo/README.md").read_text().split(
        "## Task Index", 1
    )[1].split("## Method Catalog", 1)[0]

    assert row.task_cell == RESTRICTED_TASK_CELL
    assert "PromptHMR-Video" not in task_index
    assert _task_status(row.task_cell, row.card_path.read_text())[0] == "restricted"


def test_task_registry_separates_tasks_and_benchmarks() -> None:
    tasks = TASK_REGISTRY["tasks"]
    task_ids = {task["id"] for task in tasks}
    task_labels = {task["label"] for task in tasks}

    assert len(task_ids) == len(tasks)
    assert len(task_labels) == len(tasks)
    assert task_labels == TASK_LABELS

    benchmark_ids = set()
    for benchmark in TASK_REGISTRY["benchmarks"]:
        assert benchmark["id"] not in benchmark_ids
        benchmark_ids.add(benchmark["id"])
        assert benchmark["task"] in task_ids
        task_label = next(
            task["label"] for task in tasks if task["id"] == benchmark["task"]
        )
        assert benchmark["label"].startswith(f"{task_label} · ")


def test_task_registry_is_flat_and_unclassified() -> None:
    assert "families" not in TASK_REGISTRY
    assert len(TASK_REGISTRY["tasks"]) == 15
    assert all("family" not in task for task in TASK_REGISTRY["tasks"])


def test_every_task_has_linked_readme_and_registry_resources() -> None:
    root_readme = (ROOT / "README.md").read_text()
    task_system = root_readme.split("## Task System", 1)[1].split(
        "\n## ", 1
    )[0]
    task_registry = (ROOT / "docs/tasks/README.md").read_text()
    task_matrix = task_registry.split("## Task Matrix", 1)[1].split(
        "\n## ", 1
    )[0]
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    task_index = model_zoo.split("## Task Index", 1)[1].split(
        "## Method Catalog", 1
    )[0]

    for task in TASK_REGISTRY["tasks"]:
        label = task["label"]
        for surface, table in {
            "root Task System": task_system,
            "Task Matrix": task_matrix,
            "Model Zoo Task Index": task_index,
        }.items():
            matching_rows = [
                line
                for line in table.splitlines()
                if line.startswith(f"| [{label}](")
            ]
            assert len(matching_rows) == 1, (
                f"{surface} has no unique {label} row"
            )
            task_cell = matching_rows[0].strip().strip("|").split("|")[0]
            task_target = task_cell.split("](", 1)[1].split(")", 1)[0]
            assert "leaderboard" in task_target
            assert "tasks/" not in task_target

        assert "leaderboard" in task["model_zoo_target"]
        assert "../tasks/" not in task["model_zoo_target"]


def test_release_manifest_task_labels_are_canonical() -> None:
    manifest = json.loads(
        (ROOT / "docs/model_zoo/release_manifest.json").read_text()
    )
    for model in manifest["models"].values():
        assert "task" not in model
        values = model["tasks"]
        assert isinstance(values, list)
        assert values
        assert set(values).issubset(TASK_LABELS)


def test_documentation_uses_one_information_architecture() -> None:
    readme = (ROOT / "README.md").read_text()
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    benchmark_hub = (ROOT / "docs/leaderboards/README.md").read_text()

    assert "## Task System" in readme
    assert "## Models And Benchmarks" in readme
    assert "## Motion Interoperability" in readme
    assert "## Canonical Tasks" not in readme
    assert "## Model Zoo" not in readme
    assert "## Leaderboards" not in readme

    assert "## Task Index" in model_zoo
    assert "## Method Catalog" in model_zoo
    assert "## Text And Motion" not in model_zoo
    assert "## Temporal, Editing, And Control" not in model_zoo

    benchmark_labels = {
        benchmark["label"] for benchmark in TASK_REGISTRY["benchmarks"]
    }
    assert len(benchmark_labels) == 15
    grouped_labels = {
        benchmark["label"]
        for benchmark in TASK_REGISTRY["benchmarks"]
        if benchmark.get("leaderboard")
    }
    for label in benchmark_labels - grouped_labels:
        assert f"**{label}**" in benchmark_hub
    assert benchmark_hub.count("**Text-to-Motion**") == 1
    assert "SMPL: 26 rows" in benchmark_hub
    assert "G1: 2 rows" in benchmark_hub
    assert "### T2M HumanML3D" not in benchmark_hub
    assert "### M2T HumanML3D" not in benchmark_hub
    assert "### BABEL Sequential Generation" not in benchmark_hub


def test_model_zoo_task_index_covers_every_release_capability() -> None:
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    task_index = model_zoo.split("## Task Index", 1)[1].split(
        "## Method Catalog", 1
    )[0]

    for row in _read_model_rows():
        if row.task_cell in {UNREGISTERED_TASK_CELL, RESTRICTED_TASK_CELL}:
            continue
        for task_label, _ in _parse_task_entries(row.task_cell):
            matching_lines = [
                line
                for line in task_index.splitlines()
                if f"[{task_label}](" in line
            ]
            assert len(matching_lines) == 1, (
                f"Task Index has no unique {task_label} entry"
            )
            assert f"({row.card_path.name})" in matching_lines[0], (
                f"{row.method} is missing from the {task_label} index"
            )


def test_documentation_uses_scan_friendly_tables_and_navigation() -> None:
    readme = (ROOT / "README.md").read_text()
    task_registry = (ROOT / "docs/tasks/README.md").read_text()
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    benchmark_hub = (ROOT / "docs/leaderboards/README.md").read_text()
    evaluator_zoo = (ROOT / "docs/evaluator_zoo/README.md").read_text()

    assert "| Layer | Owns | Source of truth |" in readme
    assert "| Goal | Guide |" in readme
    assert "🧭 Tasks" in readme
    assert "📦 Models" in readme
    assert "📊 Benchmarks" in readme

    assert (
        "| Task | Condition → output | Principal scope / tracks | "
        "Leaderboard settings |"
    ) in task_registry
    assert "| Method | Task coverage | Native space | Artifacts |" in model_zoo
    assert model_zoo.count("| Task | Contract | Integrated methods |") == 1
    assert (
        benchmark_hub.count(
            "| Benchmark | Status | Metrics | Visualization | Resources |"
        )
        == 1
    )
    assert "| Evaluator | Native input | Principal metrics | Artifact |" in evaluator_zoo


def test_motionbricks_is_model_zoo_method_without_task_registration() -> None:
    readme = (ROOT / "README.md").read_text()
    task_registry = (ROOT / "docs/tasks/README.md").read_text()
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    method_catalog = model_zoo.split("## Method Catalog", 1)[1].split(
        "\n## ", 1
    )[0]
    motion_toolkit = (ROOT / "docs/motion/README.md").read_text()

    assert "Robot Motion Control" not in TASK_LABELS
    assert "families" not in TASK_REGISTRY
    assert "Robot Motion Control" not in readme
    assert "## Embodied Motion" not in task_registry
    assert "### Embodied Motion" not in model_zoo
    assert (
        "| [MotionBricks](motionbricks.md) | **Not registered** |"
        in method_catalog
    )
    assert "MotionBricks runtime integration" not in motion_toolkit
    assert (ROOT / "docs/model_zoo/motionbricks.md").is_file()


def test_text_to_motion_includes_unitree_g1_setting() -> None:
    benchmarks = {
        benchmark["id"]: benchmark for benchmark in TASK_REGISTRY["benchmarks"]
    }
    setting = benchmarks["text_to_motion_unitree_g1"]
    assert setting["task"] == "text_to_motion"
    assert setting["label"] == "Text-to-Motion · Unitree G1 Skeleton"
    assert setting["leaderboard"] == {
        "id": "text_to_motion",
        "label": "Text-to-Motion",
        "target": (
            "https://huggingface.co/spaces/"
            "ZeyuLing/t2m-humanml3d-leaderboard"
        ),
    }
    assert setting["setting"] == {
        "id": "unitree_g1_skeleton",
        "label": "Unitree G1 Skeleton",
        "detail": "Native G1",
    }

    assert setting["target"].endswith("/t2m-unitree-g1-leaderboard")
    protocol_path = ROOT / "docs/leaderboards/t2m_unitree_g1.md"
    assert protocol_path.is_file()
    protocol = protocol_path.read_text()
    for expected in ("KIMODO", "HY-Motion G1", "`g1_38`", "TMR-G1"):
        assert expected in protocol


def test_local_benchmark_pages_use_canonical_titles() -> None:
    local_sources = {
        "text_to_motion_humanml3d": "hf_space_t2m_humanml3d",
        "text_to_motion_unitree_g1": "hf_space_t2m_unitree_g1",
        "motion_to_text_humanml3d": "hf_space_m2t_humanml3d",
        "sequential_text_to_motion_babel": "hf_space_babel_sequential",
        "temporal_motion_completion_humanml3d": "hf_space_temporal_condition",
        "music_to_dance_aistpp": "hf_space_music_to_dance",
        "dance_to_music_aistpp": "hf_space_dance_to_music",
        "monocular_motion_capture_3dpw_test": "hf_space_monocular_capture",
    }
    benchmarks = {
        benchmark["id"]: benchmark for benchmark in TASK_REGISTRY["benchmarks"]
    }

    for benchmark_id, directory in local_sources.items():
        label = benchmarks[benchmark_id]["label"]
        source = ROOT / "docs/leaderboards" / directory
        assert f"title: {label}" in (source / "README.md").read_text()
        page = (source / "index.html").read_text()
        assert f"<h1" in page
        assert label in page


def test_t2m_pages_share_one_leaderboard_navigation() -> None:
    sources = {
        "smpl": ROOT / "docs/leaderboards/hf_space_t2m_humanml3d/index.html",
        "g1": ROOT / "docs/leaderboards/hf_space_t2m_unitree_g1/index.html",
    }
    for setting, source in sources.items():
        page = source.read_text()
        assert "Text-to-Motion Leaderboard" in page
        assert 'data-leaderboard-id="text_to_motion"' in page
        assert page.count("<span>SMPL Skeleton</span>") == 1
        assert page.count("<span>Unitree G1 Skeleton</span>") == 1
        assert (
            'href="https://zeyuling-t2m-humanml3d-leaderboard.'
            'static.hf.space/" target="_self"'
        ) in page
        assert (
            'href="https://zeyuling-t2m-unitree-g1-leaderboard.'
            'static.hf.space/" target="_self"'
        ) in page
        assert 'target="_top"' not in page
        assert page.count('aria-current="page"') >= 2
        assert f'data-benchmark-id="text_to_motion_{"humanml3d" if setting == "smpl" else "unitree_g1"}"' in page


def test_g1_setting_has_the_shared_t2m_page_contract() -> None:
    page = (
        ROOT / "docs/leaderboards/hf_space_t2m_unitree_g1/index.html"
    ).read_text()
    for required in (
        'id="radar-chart"',
        'id="metric-chart"',
        'id="leaderboard-body"',
        '<iframe src="cases/index.html"',
        "All-case Motion Comparison",
        "Evaluation protocol",
    ):
        assert required in page
    assert '<nav class="nav"' not in page


def test_root_readme_uses_only_representation_conversion_visuals() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "<table>" not in readme
    assert "assets/model_zoo/" not in readme
    assert "### Two-Person Representation Demo" not in readme
    assert "(T, A, D)" in readme
    assert "004822_hml_smpl_soma_core_g1_1920_30fps.gif" in readme
    assert "interx_smplh_gt_G021T002A012R014_skeleton_smpl_mesh.gif" in readme
    assert "004822_skeleton_smpl_mixamo_1440_readme_30fps.gif" in readme
    assert "004822_skeleton_smpl_mixamo_1440_30fps.gif" in readme


def test_root_readme_conversion_visuals_are_high_resolution() -> None:
    assets = {
        "assets/motion/representation_demo/"
        "004822_hml_smpl_soma_core_g1_1920_30fps.gif": ((1920, 1080), 180),
        "assets/motion/fbx_character_demo/"
        "004822_skeleton_smpl_mixamo_1440_readme_30fps.gif": ((1440, 900), 90),
    }
    for relative_path, (expected_size, expected_frames) in assets.items():
        with Image.open(ROOT / relative_path) as image:
            assert image.size == expected_size
            assert image.n_frames == expected_frames
