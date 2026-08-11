"""Regression tests for the public Hugging Face checkpoint registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


def _load_specs_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "hf_checkpoint_specs.py"
    spec = importlib.util.spec_from_file_location("hf_checkpoint_specs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _import_object(path: str):
    module_name, _, name = path.rpartition(".")
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)


def test_checkpoint_specs_have_unique_repositories():
    specs = _load_specs_module().CHECKPOINT_SPECS
    sources = [spec.source_repo for spec in specs]
    targets = [spec.target_repo for spec in specs]

    assert len(sources) == len(set(sources))
    assert len(targets) == len(set(targets))


def test_every_public_method_pipeline_has_a_checkpoint_contract():
    specs = _load_specs_module().CHECKPOINT_SPECS
    registered = {spec.pipeline_class.split(".")[-2] for spec in specs}
    pipeline_root = Path(__file__).resolve().parents[1] / "motius" / "pipelines"
    packages = {
        path.name
        for path in pipeline_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }

    checkpoint_free = set(_load_specs_module().CHECKPOINT_EXEMPTIONS)
    assert packages - checkpoint_free == registered


def test_checkpoint_specs_expose_exact_task_methods():
    specs = _load_specs_module().CHECKPOINT_SPECS

    for spec in specs:
        pipeline_class = _import_object(spec.pipeline_class)
        _import_object(spec.bundle_class)
        assert callable(pipeline_class.from_pretrained)
        for task in spec.tasks:
            assert callable(getattr(pipeline_class, f"infer_{task}", None)), (
                f"{spec.target_repo} is missing infer_{task}"
            )


def test_every_released_method_card_uses_the_generic_loader():
    specs = _load_specs_module().CHECKPOINT_SPECS
    methods: dict[str, list[str]] = {}
    for spec in specs:
        method = spec.pipeline_class.split(".")[-2]
        methods.setdefault(method, []).append(spec.target_repo)

    cards = Path(__file__).resolve().parents[1] / "docs" / "model_zoo"
    for method, repositories in methods.items():
        card = cards / f"{method}.md"
        assert card.is_file(), method
        text = card.read_text(encoding="utf-8")
        assert "from motius import Pipeline" in text, method
        for repository in repositories:
            assert repository in text, (method, repository)


def test_model_card_examples_cover_registered_task_methods_and_hub_ids():
    specs = _load_specs_module().CHECKPOINT_SPECS
    methods: dict[str, dict[str, set[str]]] = {}
    for spec in specs:
        method = spec.pipeline_class.split(".")[-2]
        entry = methods.setdefault(method, {"repositories": set(), "tasks": set()})
        entry["repositories"].add(spec.target_repo)
        entry["tasks"].update(spec.tasks)

    cards = Path(__file__).resolve().parents[1] / "docs" / "model_zoo"
    for method, contract in methods.items():
        text = (cards / f"{method}.md").read_text(encoding="utf-8")
        literal_sources = set(
            re.findall(r'Pipeline\.from_pretrained\(\s*["\']([^"\']+)', text)
        )
        assert literal_sources, f"{method} has no Pipeline.from_pretrained example"
        assert literal_sources <= contract["repositories"], (
            method,
            literal_sources - contract["repositories"],
        )
        for task in contract["tasks"]:
            assert f"infer_{task}" in text, (method, task)
