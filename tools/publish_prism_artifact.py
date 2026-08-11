#!/usr/bin/env python3
"""Publish and verify one self-contained PRISM artifact on Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = {
    "README.md",
    "motius_config.json",
    "motion_stats.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer_config.json",
    "scheduler/scheduler_config.json",
}


def _resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=huggingface.co\n\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    token = values.get("password")
    if not token:
        raise RuntimeError("No Hugging Face token in environment or git credentials")
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    missing_local = sorted(
        name for name in REQUIRED_FILES if not (artifact / name).is_file()
    )
    if missing_local:
        raise FileNotFoundError(f"Artifact is incomplete: {missing_local}")

    api = HfApi(token=_resolve_token())
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=artifact,
        repo_type="model",
        private=False,
        ignore_patterns=[".cache/**", "*.log"],
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )

    info = api.model_info(args.repo_id, files_metadata=True)
    remote_files = {item.rfilename: item.size for item in info.siblings}
    missing_remote = sorted(REQUIRED_FILES - set(remote_files))
    if missing_remote:
        raise RuntimeError(f"Remote repository is incomplete: {missing_remote}")
    report = {
        "repo_id": args.repo_id,
        "url": f"https://huggingface.co/{args.repo_id}",
        "private": bool(info.private),
        "sha": info.sha,
        "file_count": len(remote_files),
        "total_bytes": sum(size or 0 for size in remote_files.values()),
        "required_files_verified": sorted(REQUIRED_FILES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
