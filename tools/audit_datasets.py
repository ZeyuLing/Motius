#!/usr/bin/env python3
"""Audit the dataset catalog, benchmark coverage, and documentation links."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs" / "datasets" / "catalog.json"
DATASET_DOC_PATH = ROOT / "docs" / "datasets" / "README.md"
TAXONOMY_PATH = ROOT / "docs" / "tasks" / "taxonomy.json"
STATUS_VALUES = {
    "motius-hosted",
    "motius-hosted-license-bound",
    "upstream-license-bound",
    "upstream-reconstruction",
    "upstream-with-motius-subset",
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _online_status(url: str, timeout: float = 15.0) -> int:
    parts = url.split("/")
    if (
        len(parts) > 8
        and parts[2] == "huggingface.co"
        and parts[3] == "datasets"
        and parts[6] == "tree"
    ):
        repo_id = f"{parts[4]}/{parts[5]}"
        revision = parts[7]
        subtree = "/".join(parts[8:])
        url = (
            f"https://huggingface.co/api/datasets/{repo_id}/tree/"
            f"{revision}/{subtree}?limit=1"
        )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Motius-dataset-audit/1.0"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        if error.code not in {403, 405}:
            return error.code
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Motius-dataset-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def run(*, online: bool = False) -> Audit:
    audit = Audit()
    catalog = _load_json(CATALOG_PATH)
    taxonomy = _load_json(TAXONOMY_PATH)
    dataset_doc = DATASET_DOC_PATH.read_text()
    root_readme = (ROOT / "README.md").read_text()
    task_doc = (ROOT / "docs" / "tasks" / "README.md").read_text()
    benchmark_doc = (ROOT / "docs" / "leaderboards" / "README.md").read_text()

    datasets = catalog.get("datasets", [])
    ids = [item.get("id") for item in datasets]
    audit.require(len(ids) == len(set(ids)), "Dataset ids must be unique.")

    known_tasks = {item["id"] for item in taxonomy["tasks"]}
    known_benchmarks = {item["id"] for item in taxonomy["benchmarks"]}
    covered_benchmarks: list[str] = []
    urls: set[str] = set()

    for item in datasets:
        context = item.get("id", "<missing-id>")
        audit.require(bool(item.get("name")), f"{context}: missing name")
        audit.require(
            item.get("status") in STATUS_VALUES,
            f"{context}: unsupported status {item.get('status')!r}",
        )
        local_root = item.get("local_root", "")
        audit.require(
            local_root.startswith("data/"),
            f"{context}: local_root must live under data/",
        )
        unknown_tasks = set(item.get("tasks", [])) - known_tasks
        audit.require(
            not unknown_tasks,
            f"{context}: unknown tasks {sorted(unknown_tasks)}",
        )
        unknown_benchmarks = set(item.get("benchmark_ids", [])) - known_benchmarks
        audit.require(
            not unknown_benchmarks,
            f"{context}: unknown benchmarks {sorted(unknown_benchmarks)}",
        )
        covered_benchmarks.extend(item.get("benchmark_ids", []))

        access = item.get("access", [])
        audit.require(bool(access), f"{context}: missing access links")
        for entry in access:
            url = entry.get("url", "")
            audit.require(
                url.startswith("https://"),
                f"{context}: access URL must use HTTPS: {url}",
            )
            urls.add(url)
            audit.require(
                url in dataset_doc,
                f"{context}: access URL is missing from Dataset Hub: {url}",
            )

        if item["status"].startswith("motius-hosted"):
            download = item.get("download", {})
            audit.require(
                bool(download.get("repo_id")),
                f"{context}: hosted dataset is missing download.repo_id",
            )
            audit.require(
                download.get("repo_type") == "dataset",
                f"{context}: hosted download must declare repo_type=dataset",
            )

        audit.require(
            f'id="{context}"' in dataset_doc,
            f"{context}: Dataset Hub is missing its stable anchor",
        )

    audit.require(
        len(covered_benchmarks) == len(set(covered_benchmarks)),
        "A benchmark may be owned by only one dataset entry.",
    )
    exemptions = catalog.get("benchmark_exemptions", [])
    exemption_ids = [item.get("id") for item in exemptions]
    audit.require(
        len(exemption_ids) == len(set(exemption_ids)),
        "Benchmark exemption ids must be unique.",
    )
    unknown_exemptions = set(exemption_ids) - known_benchmarks
    audit.require(
        not unknown_exemptions,
        f"Unknown benchmark exemptions: {sorted(unknown_exemptions)}",
    )
    missing_benchmarks = (
        known_benchmarks - set(covered_benchmarks) - set(exemption_ids)
    )
    audit.require(
        not missing_benchmarks,
        f"Benchmarks missing dataset ownership or exemption: "
        f"{sorted(missing_benchmarks)}",
    )

    for artifact in catalog.get("artifact_datasets", []):
        url = artifact.get("url", "")
        audit.require(
            url.startswith("https://huggingface.co/datasets/"),
            f"{artifact.get('id')}: invalid artifact dataset URL",
        )
        audit.require(
            url in dataset_doc,
            f"{artifact.get('id')}: artifact URL missing from Dataset Hub",
        )
        urls.add(url)

    dataset_link = "docs/datasets/README.md"
    audit.require(dataset_link in root_readme, "Root README lacks Dataset Hub link.")
    audit.require(
        "../datasets/README.md" in task_doc,
        "Task Registry lacks Dataset Hub link.",
    )
    audit.require(
        "../datasets/README.md" in benchmark_doc,
        "Benchmark Hub lacks Dataset Hub link.",
    )
    audit.require(
        "https://huggingface.co/datasets/ZeyuLing/HumanML3D" not in dataset_doc,
        "Dataset Hub must not claim a nonexistent Motius HumanML3D repository.",
    )

    if online:
        for url in sorted(urls):
            try:
                status = _online_status(url)
                audit.require(status < 400, f"Offline URL ({status}): {url}")
                audit.notes.append(f"{status} {url}")
            except (OSError, urllib.error.URLError) as error:
                audit.errors.append(f"URL check failed for {url}: {error}")

    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Probe every external dataset URL.",
    )
    args = parser.parse_args()
    audit = run(online=args.online)
    for note in audit.notes:
        print(f"NOTE: {note}")
    for error in audit.errors:
        print(f"ERROR: {error}")
    if audit.errors:
        print(f"Dataset audit failed with {len(audit.errors)} error(s).")
        return 1
    print("Dataset catalog and documentation links are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
