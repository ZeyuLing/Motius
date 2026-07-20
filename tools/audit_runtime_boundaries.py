#!/usr/bin/env python3
"""Audit method packages for accidental dependencies on legacy repositories."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path


LEGACY_MARKERS = ("ref_repo", "/hf_trainer", "\\hf_trainer")


def _is_legacy(value: object) -> bool:
    text = str(value).replace("\\", "/").lower()
    return any(marker.replace("\\", "/") in text for marker in LEGACY_MARKERS)


def scan_imports(root: Path) -> list[dict]:
    findings = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if _is_legacy(name):
                    findings.append(
                        {"file": str(path), "line": node.lineno, "import": name}
                    )
    return findings


class _LegacyOriginBlocker:
    def find_spec(self, fullname, path=None, target=None):
        for entry in path or sys.path:
            if _is_legacy(entry):
                raise ImportError(
                    f"Blocked legacy import path while resolving {fullname}: {entry}"
                )
        return None


def import_pipeline_packages(root: Path) -> dict:
    packages = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )
    results = {}
    blocker = _LegacyOriginBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        for package in packages:
            try:
                importlib.import_module(f"motius.pipelines.{package}")
                legacy_origins = sorted(
                    {
                        str(module.__file__)
                        for module in sys.modules.values()
                        if getattr(module, "__file__", None)
                        and _is_legacy(module.__file__)
                    }
                )
                results[package] = {
                    "status": "ok" if not legacy_origins else "legacy-origin",
                    "legacy_origins": legacy_origins,
                }
            except Exception as exc:
                results[package] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
    finally:
        sys.meta_path.remove(blocker)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audits/runtime_boundary_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_roots = [Path("motius/models"), Path("motius/pipelines"), Path("motius/trainers")]
    static_findings = [
        finding
        for root in code_roots
        if root.exists()
        for finding in scan_imports(root)
    ]
    pipeline_results = import_pipeline_packages(Path("motius/pipelines"))
    passed = not static_findings and all(
        result["status"] == "ok" for result in pipeline_results.values()
    )
    payload = {
        "passed": passed,
        "policy": "No model, pipeline, or trainer may import code from ref_repo or hf_trainer.",
        "static_import_findings": static_findings,
        "pipeline_packages": pipeline_results,
        "summary": {
            "pipeline_packages": len(pipeline_results),
            "runtime_imports_ok": sum(
                result["status"] == "ok" for result in pipeline_results.values()
            ),
            "static_legacy_imports": len(static_findings),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
