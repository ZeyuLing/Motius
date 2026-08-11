#!/usr/bin/env python3
"""Normalize Motius Hugging Face checkpoint metadata without re-uploading weights."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from hf_checkpoint_specs import CHECKPOINT_SPECS, CheckpointSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "hf_checkpoint_metadata"
IGNORED_REQUIRED_FILES = {".gitattributes", "README.md", "model_index.json"}
LOCAL_TEXT_ENCODERS = {
    "MLDPipeline": ("text_encoder_name", "text_encoder/sentence-t5-large"),
    "MotionLCMPipeline": ("text_encoder_name", "text_encoder/sentence-t5-large"),
    "MotionStreamerPipeline": (
        "text_model_name",
        "text_encoder/sentence-t5-xxl",
    ),
}


def _credential() -> str | None:
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def _replace_legacy(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_legacy(item, replacements)
            for key, item in value.items()
            if key not in {"_diffusers_version"}
        }
    if isinstance(value, list):
        return [_replace_legacy(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.replace("hftrainer", "motius")


def _mark_kimodo_text_encoder_local(value: Any) -> Any:
    if isinstance(value, list):
        return [_mark_kimodo_text_encoder_local(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _mark_kimodo_text_encoder_local(item)
        for key, item in value.items()
    }
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict) and (
        "kimodo_checkpoint" in artifacts or "text_encoders" in artifacts
    ):
        artifacts["text_encoders"] = "text_encoders"

    if "text_encoder_mode" in normalized or "text_encoders_repo" in normalized:
        normalized["text_encoder_mode"] = "local"
        normalized["text_encoders_dir"] = "text_encoders"
        normalized["text_encoders_subdir"] = "text_encoders"
        normalized.pop("text_encoders_repo", None)

    for key in ("components", "external_components"):
        components = normalized.get(key)
        if not isinstance(components, dict):
            continue
        text_encoder = components.get("text_encoder")
        if not isinstance(text_encoder, dict):
            continue
        text_encoder["stored_in_artifact"] = True
        text_encoder["path"] = "text_encoders"
        text_encoder.pop("repo", None)
    return normalized


def _mark_text_encoder_local(
    value: Any,
    *,
    config_key: str,
    component_path: str,
) -> Any:
    if isinstance(value, list):
        return [
            _mark_text_encoder_local(
                item,
                config_key=config_key,
                component_path=component_path,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _mark_text_encoder_local(
            item,
            config_key=config_key,
            component_path=component_path,
        )
        for key, item in value.items()
    }
    if config_key in normalized:
        normalized[config_key] = component_path
        normalized["text_encoder_stored_in_artifact"] = True

    for key in ("components", "external_components"):
        components = normalized.get(key)
        if not isinstance(components, dict):
            continue
        text_encoder = components.get("text_encoder")
        if not isinstance(text_encoder, dict):
            continue
        text_encoder["stored_in_artifact"] = True
        text_encoder["path"] = component_path
        text_encoder.pop("repo", None)
    return normalized


def _mark_ardy_text_encoder_local(value: Any) -> Any:
    if isinstance(value, list):
        return [_mark_ardy_text_encoder_local(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _mark_ardy_text_encoder_local(item)
        for key, item in value.items()
    }
    if "text_encoders_dir" in normalized:
        normalized["text_encoders_dir"] = "text_encoders"
        normalized["text_encoder_mode"] = "local"
    text_encoder = normalized.get("text_encoder")
    if isinstance(text_encoder, dict):
        text_encoder["stored_in_artifact"] = True
        text_encoder["path"] = "text_encoders"
        text_encoder["requires_hf_auth"] = False
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["text_encoders"] = "text_encoders"
    return normalized


def _normalize_artifact_json(
    spec: CheckpointSpec,
    value: Any,
    repo_files: list[str],
    replacements: dict[str, str],
) -> Any:
    normalized = _replace_legacy(value, replacements)
    if (
        spec.class_name == "KIMODOPipeline"
        and any(filename.startswith("text_encoders/") for filename in repo_files)
    ):
        normalized = _mark_kimodo_text_encoder_local(normalized)
    if (
        spec.class_name == "ARDYPipeline"
        and any(filename.startswith("text_encoders/") for filename in repo_files)
    ):
        normalized = _mark_ardy_text_encoder_local(normalized)
    text_encoder_layout = LOCAL_TEXT_ENCODERS.get(spec.class_name)
    if text_encoder_layout is not None:
        config_key, component_path = text_encoder_layout
        if any(filename.startswith(f"{component_path}/") for filename in repo_files):
            normalized = _mark_text_encoder_local(
                normalized,
                config_key=config_key,
                component_path=component_path,
            )
    if (
        spec.class_name == "ViMoGenPipeline"
        and "wan/models_t5_umt5-xxl-enc-bf16.pth" in repo_files
    ):
        if isinstance(normalized, dict):
            normalized["wan_dir"] = "wan"
            normalized["text_encoder_stored_in_artifact"] = True
            normalized.pop("wan_repo_id", None)
    return normalized


def _read_remote_json(
    repo_id: str,
    filename: str,
    revision: str,
    token: str | None,
) -> dict:
    try:
        path = hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            token=token,
            cache_dir=str(REPO_ROOT / "outputs" / "cache" / "hf_contract_audit"),
        )
    except Exception:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _normalized_index(
    spec: CheckpointSpec,
    existing: dict,
    repo_files: list[str],
    replacements: dict[str, str],
) -> dict:
    normalized = _normalize_artifact_json(
        spec,
        deepcopy(existing),
        repo_files,
        replacements,
    )
    normalized.update(
        {
            "_class_name": spec.class_name,
            "_library_name": "motius",
            "format_version": 1,
            "pipeline_class": spec.pipeline_class,
            "bundle_class": spec.bundle_class,
            "tasks": list(spec.tasks),
            "required_files": sorted(
                filename
                for filename in repo_files
                if filename not in IGNORED_REQUIRED_FILES
            ),
            "api": {
                "loader": "motius.Pipeline.from_pretrained",
                "task_methods": [f"infer_{task}" for task in spec.tasks],
            },
        }
    )
    text_encoder_layout = LOCAL_TEXT_ENCODERS.get(spec.class_name)
    if text_encoder_layout is not None:
        _, component_path = text_encoder_layout
        if any(filename.startswith(f"{component_path}/") for filename in repo_files):
            artifacts = normalized.setdefault("artifacts", {})
            artifacts["text_encoder"] = component_path
            components = normalized.setdefault("components", {})
            components["text_encoder"] = {
                "stored_in_artifact": True,
                "path": component_path,
            }
    if (
        spec.class_name == "ViMoGenPipeline"
        and "wan/models_t5_umt5-xxl-enc-bf16.pth" in repo_files
    ):
        artifacts = normalized.setdefault("artifacts", {})
        artifacts["wan_text_encoder"] = "wan"
        components = normalized.setdefault("components", {})
        components["text_encoder"] = {
            "stored_in_artifact": True,
            "path": "wan",
        }
    if "clip.safetensors" in repo_files:
        artifacts = normalized.setdefault("artifacts", {})
        artifacts["clip"] = "clip.safetensors"
        components = normalized.setdefault("components", {})
        components["clip"] = {
            "stored_in_artifact": True,
            "path": "clip.safetensors",
        }
    normalized.pop("pipeline", None)
    normalized.pop("bundle", None)
    if "artifact_format" not in normalized and "format" not in normalized:
        slug = spec.class_name.removesuffix("Pipeline").lower()
        normalized["artifact_format"] = f"motius-{slug}-v1"
    return normalized


def _normalize_readme(
    text: str,
    spec: CheckpointSpec,
    replacements: dict[str, str],
) -> str:
    text = _replace_legacy(text, replacements)
    if "from motius import Pipeline" not in text:
        usage = (
            "\n## Direct Loading\n\n"
            "```python\n"
            "from motius import Pipeline\n\n"
            f'pipeline = Pipeline.from_pretrained("{spec.target_repo}")\n'
            "```\n"
        )
        text = text.rstrip() + "\n" + usage
    return text


def _repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.model_info(repo_id)
    except Exception:
        return False
    return True


def _canonical_repo_id(api: HfApi, repo_id: str) -> str | None:
    try:
        return str(api.model_info(repo_id).id)
    except Exception:
        return None


def _active_repo(api: HfApi, spec: CheckpointSpec) -> str:
    if _repo_exists(api, spec.target_repo):
        return spec.target_repo
    return spec.source_repo


def normalize_spec(
    api: HfApi,
    spec: CheckpointSpec,
    *,
    move_repos: bool,
    upload: bool,
    replacements: dict[str, str],
) -> dict:
    if move_repos and spec.source_repo != spec.target_repo:
        source_identity = _canonical_repo_id(api, spec.source_repo)
        target_identity = _canonical_repo_id(api, spec.target_repo)
        source_exists = source_identity is not None
        target_exists = target_identity is not None
        if source_exists and target_exists:
            if source_identity.lower() != target_identity.lower():
                raise RuntimeError(
                    "Both source and target repos exist: "
                    f"{spec.source_repo}, {spec.target_repo}"
                )
            source_exists = False
        if source_exists:
            api.move_repo(spec.source_repo, spec.target_repo, repo_type="model")

    repo_id = _active_repo(api, spec)
    info = api.model_info(repo_id, files_metadata=True)
    files = sorted(sibling.rfilename for sibling in info.siblings)
    existing = _read_remote_json(repo_id, "model_index.json", info.sha, api.token)
    model_index = _normalized_index(spec, existing, files, replacements)

    staging = OUTPUT_ROOT / repo_id.replace("/", "--")
    staging.mkdir(parents=True, exist_ok=True)
    index_path = staging / "model_index.json"
    index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")
    operations = [
        CommitOperationAdd(path_in_repo="model_index.json", path_or_fileobj=index_path)
    ]
    json_changed = []
    for sibling in info.siblings:
        filename = sibling.rfilename
        if (
            filename == "model_index.json"
            or not filename.endswith(".json")
            or (sibling.size or 0) > 2_000_000
        ):
            continue
        try:
            remote_path = hf_hub_download(
                repo_id,
                filename,
                revision=info.sha,
                token=api.token,
                cache_dir=str(REPO_ROOT / "outputs" / "cache" / "hf_contract_audit"),
            )
            old_text = Path(remote_path).read_text(encoding="utf-8")
            old_value = json.loads(old_text)
        except (OSError, json.JSONDecodeError):
            continue
        new_value = _normalize_artifact_json(
            spec,
            old_value,
            files,
            replacements,
        )
        if new_value == old_value:
            continue
        staged_json = staging / filename
        staged_json.parent.mkdir(parents=True, exist_ok=True)
        staged_json.write_text(
            json.dumps(new_value, indent=2) + "\n",
            encoding="utf-8",
        )
        operations.append(
            CommitOperationAdd(
                path_in_repo=filename,
                path_or_fileobj=staged_json,
            )
        )
        json_changed.append(filename)

    readme_changed = False
    if "README.md" in files:
        readme_path = hf_hub_download(
            repo_id,
            "README.md",
            revision=info.sha,
            token=api.token,
            cache_dir=str(REPO_ROOT / "outputs" / "cache" / "hf_contract_audit"),
        )
        old_readme = Path(readme_path).read_text(encoding="utf-8")
        new_readme = _normalize_readme(old_readme, spec, replacements)
        readme_changed = new_readme != old_readme
        if readme_changed:
            staged_readme = staging / "README.md"
            staged_readme.write_text(new_readme, encoding="utf-8")
            operations.append(
                CommitOperationAdd(
                    path_in_repo="README.md",
                    path_or_fileobj=staged_readme,
                )
            )

    commit = None
    if upload:
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Normalize Motius Pipeline.from_pretrained metadata",
        )
    return {
        "source_repo": spec.source_repo,
        "target_repo": spec.target_repo,
        "active_repo": repo_id,
        "revision_before": info.sha,
        "files": len(files),
        "required_files": len(model_index["required_files"]),
        "readme_changed": readme_changed,
        "json_changed": json_changed,
        "uploaded": upload,
        "commit": None if commit is None else commit.oid,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move-repos", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Limit to a source or target repo id; repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audits/hf_checkpoint_normalization.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = _credential()
    if (args.move_repos or args.upload) and not token:
        print("error: authenticated Hugging Face token is required", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    selected = [
        spec
        for spec in CHECKPOINT_SPECS
        if not args.repo
        or spec.source_repo in args.repo
        or spec.target_repo in args.repo
    ]
    replacements = {
        spec.source_repo: spec.target_repo
        for spec in CHECKPOINT_SPECS
        if spec.source_repo != spec.target_repo
    }
    results = []
    for spec in selected:
        try:
            result = normalize_spec(
                api,
                spec,
                move_repos=args.move_repos,
                upload=args.upload,
                replacements=replacements,
            )
        except Exception as exc:
            result = {
                "source_repo": spec.source_repo,
                "target_repo": spec.target_repo,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(
            f"{spec.source_repo} -> {spec.target_repo}: "
            f"{'ERROR' if 'error' in result else 'ready'}"
        )

    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 1 if any("error" in result for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
