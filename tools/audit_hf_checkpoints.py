#!/usr/bin/env python3
"""Audit public Motius checkpoints against the direct-loading Hub contract."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_checkpoint_specs import CHECKPOINT_SPECS


def _credential() -> str | None:
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def _import_object(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.model_info(repo_id)
    except Exception:
        return False
    return True


def _active_repo(api: HfApi, spec) -> str:
    return spec.target_repo if _repo_exists(api, spec.target_repo) else spec.source_repo


def _find_unbundled_components(value, path: str = "model_index") -> list[str]:
    errors = []
    if isinstance(value, dict):
        if value.get("stored_in_artifact") is False:
            errors.append(path)
        for key, item in value.items():
            errors.extend(_find_unbundled_components(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_find_unbundled_components(item, f"{path}[{index}]"))
    return errors


def audit_spec(api: HfApi, spec) -> dict:
    repo_id = _active_repo(api, spec)
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "status": "unreachable",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    files = {sibling.rfilename for sibling in info.siblings}
    errors = []
    if "model_index.json" not in files:
        errors.append("missing model_index.json")
        manifest = {}
    else:
        try:
            path = hf_hub_download(
                repo_id,
                "model_index.json",
                revision=info.sha,
                cache_dir=str(REPO_ROOT / "outputs" / "cache" / "hf_contract_audit"),
                token=api.token,
            )
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            manifest = {}
            errors.append(f"invalid model_index.json: {type(exc).__name__}: {exc}")

    serialized = json.dumps(manifest).lower()
    if "hftrainer" in repo_id.lower():
        errors.append("legacy package name remains in repo id")
    if "hftrainer" in serialized:
        errors.append("legacy package name remains in model_index.json")
    if manifest.get("_library_name") != "motius":
        errors.append("_library_name must be motius")
    if manifest.get("pipeline_class") != spec.pipeline_class:
        errors.append("pipeline_class mismatch")
    if manifest.get("bundle_class") != spec.bundle_class:
        errors.append("bundle_class mismatch")
    if tuple(manifest.get("tasks", ())) != spec.tasks:
        errors.append("tasks mismatch")
    unbundled = _find_unbundled_components(manifest)
    if unbundled:
        errors.append(f"components are not stored in artifact: {unbundled[:5]}")

    required_files = manifest.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        errors.append("required_files is missing or empty")
        required_files = []
    missing = sorted(set(required_files) - files)
    if missing:
        errors.append(f"required_files missing remotely: {missing[:5]}")

    try:
        pipeline_class = _import_object(spec.pipeline_class)
    except Exception as exc:
        pipeline_class = None
        errors.append(f"pipeline import failed: {type(exc).__name__}: {exc}")
    try:
        _import_object(spec.bundle_class)
    except Exception as exc:
        errors.append(f"bundle import failed: {type(exc).__name__}: {exc}")
    if pipeline_class is not None:
        if not callable(getattr(pipeline_class, "from_pretrained", None)):
            errors.append("pipeline has no from_pretrained")
        for task in spec.tasks:
            method = f"infer_{task}"
            if not callable(getattr(pipeline_class, method, None)):
                errors.append(f"missing canonical task API {method}")

    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "status": "pass" if not errors else "fail",
        "files": len(files),
        "size_bytes": sum((sibling.size or 0) for sibling in info.siblings),
        "required_files": len(required_files),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audits/hf_checkpoint_direct_load.json"),
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Limit to a source or target repo id; repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = HfApi(token=_credential())
    selected = [
        spec
        for spec in CHECKPOINT_SPECS
        if not args.repo
        or spec.source_repo in args.repo
        or spec.target_repo in args.repo
    ]
    results = [audit_spec(api, spec) for spec in selected]
    summary = {
        "checkpoints": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "failed": sum(result["status"] != "pass" for result in results),
    }
    payload = {"summary": summary, "checkpoints": results}
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for result in results:
        if result["status"] != "pass":
            print(f"{result['repo_id']}: {'; '.join(result['errors'])}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
