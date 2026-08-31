from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from tools import publish_hf_spaces as publisher

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


def _make_manifest(root: Path) -> Path:
    source = root / "docs" / "leaderboards" / "hf_space_alpha"
    source.mkdir(parents=True)
    taxonomy_dir = root / "docs" / "tasks"
    taxonomy_dir.mkdir(parents=True)
    catalog = {
        "schema_version": 1,
        "benchmarks": [
            {
                "id": "alpha_benchmark",
                "source": "docs/leaderboards/hf_space_alpha",
            }
        ],
    }
    taxonomy = {
        "schema_version": 2,
        "benchmarks": [
            {
                "id": "alpha_benchmark",
                "target": "https://huggingface.co/spaces/Example/alpha-space",
            }
        ],
    }
    (root / "docs" / "leaderboards" / "catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )
    (taxonomy_dir / "taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    return source


def _make_git_fixture(root: Path) -> tuple[publisher.SpaceSpec, Path]:
    source = _make_manifest(root)
    (source / "index.html").write_text("<h1>old</h1>\n", encoding="utf-8")
    (source / "leaderboard.js").write_text("old();\n", encoding="utf-8")
    (source / "payload.json").write_text('{"value": 1}\n', encoding="utf-8")
    (source / "README.md").write_text("# Alpha\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Motius Tests")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "fixture")

    (source / "index.html").write_text("<h1>new</h1>\n", encoding="utf-8")
    (source / "leaderboard.js").unlink()
    (source / "payload.json").write_text('{"value": 2}\n', encoding="utf-8")
    (source / "theme.css").write_text("body { color: navy; }\n", encoding="utf-8")
    (source / "preview.png").write_bytes(b"not uploaded")
    return publisher.load_space_manifest(root)[0], source


def test_repository_manifest_derives_all_sixteen_space_repo_ids() -> None:
    specs = publisher.load_space_manifest(ROOT)

    assert len(specs) == 16
    assert len({spec.source for spec in specs}) == 16
    assert len({spec.repo_id for spec in specs}) == 16
    by_id = {spec.benchmark_id: spec for spec in specs}
    assert by_id["sequential_text_to_motion_babel"].repo_id == (
        "ZeyuLing/babel-sequential-generation-leaderboard"
    )
    assert by_id["sequential_text_to_motion_babel"].source == PurePosixPath(
        "docs/leaderboards/hf_space_babel_sequential"
    )


def test_since_plan_includes_only_changed_or_new_utf8_interface_files(
    tmp_path: Path,
) -> None:
    spec, _ = _make_git_fixture(tmp_path)

    plan = publisher.build_publish_plans(
        tmp_path, (spec,), since="HEAD", include_all=False
    )[0]

    assert [file.path_in_repo for file in plan.files] == ["index.html", "theme.css"]
    assert {item.repo_path: item.reason for item in plan.skipped} == {
        "docs/leaderboards/hf_space_alpha/leaderboard.js": (
            "delete preserved remotely"
        ),
        "docs/leaderboards/hf_space_alpha/payload.json": "not an interface file",
        "docs/leaderboards/hf_space_alpha/preview.png": "not an interface file",
    }
    assert {file.change for file in plan.files} == {"M", "new"}


def test_all_plan_includes_tracked_and_untracked_interface_files_only(
    tmp_path: Path,
) -> None:
    spec, _ = _make_git_fixture(tmp_path)

    plan = publisher.build_publish_plans(
        tmp_path, (spec,), since=None, include_all=True
    )[0]

    assert [file.path_in_repo for file in plan.files] == [
        "README.md",
        "index.html",
        "theme.css",
    ]


def test_publish_uses_add_operations_and_parent_pinned_space_commit(
    tmp_path: Path,
) -> None:
    spec, _ = _make_git_fixture(tmp_path)
    plan = publisher.build_publish_plans(
        tmp_path, (spec,), since="HEAD", include_all=False
    )[0]
    operation_calls = []

    def operation_factory(**kwargs: object) -> dict[str, object]:
        operation_calls.append(kwargs)
        return dict(kwargs)

    class FakeApi:
        def __init__(self) -> None:
            self.create_call = None

        def repo_info(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs == {"repo_id": spec.repo_id, "repo_type": "space"}
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            self.create_call = kwargs
            return SimpleNamespace(
                oid="b" * 40,
                commit_url="https://huggingface.co/spaces/Example/alpha-space/commit/"
                + "b" * 40,
            )

    api = FakeApi()
    result = publisher.publish_space(
        plan,
        api=api,
        operation_factory=operation_factory,
        commit_message="Refresh interface",
    )

    assert [call["path_in_repo"] for call in operation_calls] == [
        "index.html",
        "theme.css",
    ]
    assert all(isinstance(call["path_or_fileobj"], bytes) for call in operation_calls)
    assert api.create_call == {
        "repo_id": spec.repo_id,
        "repo_type": "space",
        "operations": operation_calls,
        "commit_message": "Refresh interface",
        "parent_commit": "a" * 40,
    }
    assert result.before_sha == "a" * 40
    assert result.after_sha == "b" * 40
    assert result.commit_url.endswith("b" * 40)


def test_publish_refuses_file_changed_after_dry_run(tmp_path: Path) -> None:
    spec, source = _make_git_fixture(tmp_path)
    plan = publisher.build_publish_plans(
        tmp_path, (spec,), since="HEAD", include_all=False
    )[0]
    (source / "index.html").write_text("changed again\n", encoding="utf-8")

    api = SimpleNamespace(
        repo_info=lambda **_: SimpleNamespace(sha="a" * 40),
        create_commit=lambda **_: pytest.fail("create_commit must not be called"),
    )
    with pytest.raises(publisher.PublishError, match="changed after planning"):
        publisher.publish_space(
            plan,
            api=api,
            operation_factory=lambda **kwargs: kwargs,
            commit_message="Refresh interface",
        )


def test_plan_refuses_binary_content_with_an_interface_extension(
    tmp_path: Path,
) -> None:
    spec, source = _make_git_fixture(tmp_path)
    (source / "unsafe.js").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(publisher.PublishError, match="contains NUL bytes"):
        publisher.build_publish_plans(
            tmp_path, (spec,), since="HEAD", include_all=False
        )


def test_space_selector_accepts_manifest_ids_and_rejects_unknown_values() -> None:
    specs = publisher.load_space_manifest(ROOT)
    chosen = publisher.select_spaces(specs, ["hf_space_motion_repair"])

    assert [spec.benchmark_id for spec in chosen] == ["motion_repair_brokenamass"]
    with pytest.raises(publisher.PublishError, match="Unknown Space selector"):
        publisher.select_spaces(specs, ["not-a-space"])


def test_cli_is_dry_run_unless_apply_is_explicit() -> None:
    args = publisher.parse_args([])

    assert args.apply is False
    assert args.since == "HEAD"

    selected = publisher.parse_args(["--space", "hf_space_motion_repair"])
    assert selected.since == "HEAD"
    assert selected.space == ["hf_space_motion_repair"]
