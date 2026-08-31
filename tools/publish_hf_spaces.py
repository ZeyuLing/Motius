#!/usr/bin/env python3
"""Safely publish Motius' static leaderboard interfaces to HF Spaces.

The local-to-remote manifest is derived from ``catalog.json`` and
``taxonomy.json``.  The command is a local dry run unless ``--apply`` is
present.  It never creates delete operations and only accepts UTF-8 interface
files: ``.html``, ``.css``, ``.js``, and files named ``README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PurePosixPath("docs/leaderboards/catalog.json")
TAXONOMY_PATH = PurePosixPath("docs/tasks/taxonomy.json")
INTERFACE_SUFFIXES = frozenset({".html", ".css", ".js"})


class PublishError(RuntimeError):
    """Raised when a publish plan is ambiguous or unsafe."""


@dataclass(frozen=True)
class SpaceSpec:
    """One catalog benchmark and its corresponding Hugging Face Space."""

    benchmark_id: str
    source: PurePosixPath
    repo_id: str


@dataclass(frozen=True)
class PublishFile:
    """A validated interface file that can be uploaded."""

    local_path: Path
    path_in_repo: str
    change: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SkippedPath:
    """A changed path intentionally excluded from the upload."""

    repo_path: str
    reason: str


@dataclass(frozen=True)
class SpacePlan:
    """The files selected for one Space commit."""

    spec: SpaceSpec
    files: tuple[PublishFile, ...]
    skipped: tuple[SkippedPath, ...]


@dataclass(frozen=True)
class PublishResult:
    """The immutable identifiers returned by a successful Space commit."""

    repo_id: str
    before_sha: str
    after_sha: str
    commit_url: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublishError(f"Manifest must contain a JSON object: {path}")
    return value


def _space_repo_id(target: object, *, benchmark_id: str) -> str:
    if not isinstance(target, str):
        raise PublishError(f"Taxonomy benchmark {benchmark_id!r} has no string target")
    parsed = urlsplit(target)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "huggingface.co"
        or parsed.query
        or parsed.fragment
        or len(parts) != 3
        or parts[0] != "spaces"
    ):
        raise PublishError(
            f"Taxonomy benchmark {benchmark_id!r} does not target a canonical "
            f"Hugging Face Space URL: {target!r}"
        )
    owner, name = parts[1:]
    if not owner or not name or any(part in {".", ".."} for part in (owner, name)):
        raise PublishError(
            f"Taxonomy benchmark {benchmark_id!r} has an unsafe Space target"
        )
    return f"{owner}/{name}"


def _safe_source(root: Path, source: object, *, benchmark_id: str) -> PurePosixPath:
    if not isinstance(source, str):
        raise PublishError(f"Catalog benchmark {benchmark_id!r} has no source")
    relative = PurePosixPath(source)
    if relative.is_absolute() or ".." in relative.parts:
        raise PublishError(
            f"Catalog benchmark {benchmark_id!r} has an unsafe source: {source!r}"
        )
    expected_root = PurePosixPath("docs/leaderboards")
    if relative.parent != expected_root or not relative.name.startswith("hf_space_"):
        raise PublishError(
            f"Catalog benchmark {benchmark_id!r} source is not a Space root: {source!r}"
        )
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PublishError(
            f"Catalog benchmark {benchmark_id!r} source escapes the repository"
        ) from exc
    if not resolved.is_dir():
        raise PublishError(
            f"Catalog benchmark {benchmark_id!r} source does not exist: {source}"
        )
    return relative


def load_space_manifest(root: Path = REPO_ROOT) -> tuple[SpaceSpec, ...]:
    """Join catalog sources to taxonomy Space URLs by benchmark ID."""

    root = root.resolve()
    catalog = _read_json(root / Path(*CATALOG_PATH.parts))
    taxonomy = _read_json(root / Path(*TAXONOMY_PATH.parts))
    catalog_rows = catalog.get("benchmarks")
    taxonomy_rows = taxonomy.get("benchmarks")
    if not isinstance(catalog_rows, list) or not isinstance(taxonomy_rows, list):
        raise PublishError("Both manifests must contain a benchmarks list")

    targets: dict[str, object] = {}
    for row in taxonomy_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise PublishError("Taxonomy contains a benchmark without a string ID")
        benchmark_id = row["id"]
        if benchmark_id in targets:
            raise PublishError(f"Duplicate taxonomy benchmark ID: {benchmark_id}")
        targets[benchmark_id] = row.get("target")

    specs: list[SpaceSpec] = []
    seen_ids: set[str] = set()
    seen_sources: set[PurePosixPath] = set()
    seen_repos: set[str] = set()
    for row in catalog_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise PublishError("Catalog contains a benchmark without a string ID")
        benchmark_id = row["id"]
        if benchmark_id in seen_ids:
            raise PublishError(f"Duplicate catalog benchmark ID: {benchmark_id}")
        if benchmark_id not in targets:
            raise PublishError(
                f"Catalog benchmark {benchmark_id!r} is missing from taxonomy"
            )
        source = _safe_source(root, row.get("source"), benchmark_id=benchmark_id)
        repo_id = _space_repo_id(targets[benchmark_id], benchmark_id=benchmark_id)
        if source in seen_sources:
            raise PublishError(f"Duplicate Space source directory: {source}")
        if repo_id in seen_repos:
            raise PublishError(f"Duplicate Space repo ID: {repo_id}")
        seen_ids.add(benchmark_id)
        seen_sources.add(source)
        seen_repos.add(repo_id)
        specs.append(SpaceSpec(benchmark_id, source, repo_id))

    if not specs:
        raise PublishError("The leaderboard catalog does not define any Spaces")
    return tuple(specs)


def select_spaces(
    specs: Sequence[SpaceSpec], selectors: Sequence[str]
) -> tuple[SpaceSpec, ...]:
    """Resolve repeatable ``--space`` selectors without guessing."""

    if not selectors:
        return tuple(specs)
    selected: list[SpaceSpec] = []
    for selector in selectors:
        matches = [
            spec
            for spec in specs
            if selector
            in {
                spec.benchmark_id,
                spec.repo_id,
                spec.source.as_posix(),
                spec.source.name,
            }
        ]
        if not matches:
            raise PublishError(f"Unknown Space selector: {selector!r}")
        if len(matches) > 1:
            raise PublishError(f"Ambiguous Space selector: {selector!r}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return tuple(selected)


def _git(root: Path, args: Sequence[str]) -> bytes:
    command = ["git", "-C", str(root), *args]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise PublishError("git is required to construct a publish plan") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise PublishError(f"git command failed: {detail or 'unknown error'}") from exc
    return result.stdout


def _decode_nul_paths(payload: bytes) -> list[str]:
    return [
        part.decode("utf-8", errors="surrogateescape")
        for part in payload.split(b"\0")
        if part
    ]


def _resolve_ref(root: Path, ref: str) -> str:
    if not ref or ref.startswith("-"):
        raise PublishError("--since must be a non-option git ref")
    resolved = (
        _git(
            root,
            ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if len(resolved) != 40 or any(c not in "0123456789abcdef" for c in resolved):
        raise PublishError(f"git did not resolve {ref!r} to a full commit SHA")
    return resolved


def _parse_name_status(payload: bytes) -> list[tuple[str, str]]:
    tokens = _decode_nul_paths(payload)
    rows: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if not status:
            raise PublishError("git returned an empty change status")
        kind = status[0]
        path_count = 2 if kind in {"R", "C"} else 1
        if index + path_count > len(tokens):
            raise PublishError("git returned malformed name-status output")
        paths = tokens[index : index + path_count]
        index += path_count
        if kind == "D":
            rows.append(("D", paths[0]))
        elif kind in {"R", "C"}:
            # The old remote path is deliberately preserved; only the current
            # destination is eligible for an add/update operation.
            rows.append((kind, paths[-1]))
        else:
            rows.append((kind, paths[0]))
    return rows


def _is_interface_path(path: PurePosixPath) -> bool:
    return path.name == "README.md" or path.suffix.lower() in INTERFACE_SUFFIXES


def _spec_for_repo_path(
    specs: Sequence[SpaceSpec], repo_path: PurePosixPath
) -> SpaceSpec | None:
    for spec in specs:
        try:
            repo_path.relative_to(spec.source)
        except ValueError:
            continue
        return spec
    return None


def _validated_file(
    root: Path,
    spec: SpaceSpec,
    repo_path: PurePosixPath,
    *,
    change: str,
) -> PublishFile:
    path_in_repo = repo_path.relative_to(spec.source)
    if not path_in_repo.parts or ".." in path_in_repo.parts:
        raise PublishError(f"Unsafe path selected for {spec.repo_id}: {repo_path}")
    local_path = root.joinpath(*repo_path.parts)
    if local_path.is_symlink():
        raise PublishError(f"Refusing to upload symlink: {repo_path}")
    resolved = local_path.resolve()
    source_root = root.joinpath(*spec.source.parts).resolve()
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise PublishError(f"Path escapes Space source: {repo_path}") from exc
    if not resolved.is_file():
        raise PublishError(f"Selected interface file does not exist: {repo_path}")
    payload = resolved.read_bytes()
    if b"\0" in payload:
        raise PublishError(f"Interface file contains NUL bytes: {repo_path}")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PublishError(f"Interface file is not UTF-8 text: {repo_path}") from exc
    return PublishFile(
        local_path=resolved,
        path_in_repo=path_in_repo.as_posix(),
        change=change,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _source_args(specs: Sequence[SpaceSpec]) -> list[str]:
    return [spec.source.as_posix() for spec in specs]


def _all_candidates(root: Path, specs: Sequence[SpaceSpec]) -> list[tuple[str, str]]:
    tracked = _decode_nul_paths(
        _git(root, ["ls-files", "-z", "--cached", "--", *_source_args(specs)])
    )
    untracked = _decode_nul_paths(
        _git(
            root,
            [
                "ls-files",
                "-z",
                "--others",
                "--exclude-standard",
                "--",
                *_source_args(specs),
            ],
        )
    )
    tracked_rows = []
    for path in tracked:
        local_path = root.joinpath(*PurePosixPath(path).parts)
        change = "tracked" if local_path.exists() or local_path.is_symlink() else "D"
        tracked_rows.append((change, path))
    return tracked_rows + [("new", path) for path in untracked]


def _changed_candidates(
    root: Path, specs: Sequence[SpaceSpec], ref: str
) -> list[tuple[str, str]]:
    commit = _resolve_ref(root, ref)
    changed = _parse_name_status(
        _git(
            root,
            [
                "diff",
                "--name-status",
                "-z",
                "--find-renames",
                commit,
                "--",
                *_source_args(specs),
            ],
        )
    )
    untracked = _decode_nul_paths(
        _git(
            root,
            [
                "ls-files",
                "-z",
                "--others",
                "--exclude-standard",
                "--",
                *_source_args(specs),
            ],
        )
    )
    return changed + [("?", path) for path in untracked]


def build_publish_plans(
    root: Path,
    specs: Sequence[SpaceSpec],
    *,
    since: str | None = None,
    include_all: bool = False,
) -> tuple[SpacePlan, ...]:
    """Build validated per-Space plans from git without using the network."""

    root = root.resolve()
    if (since is None) == (not include_all):
        raise PublishError("Choose exactly one of since=<ref> or include_all=True")
    if not specs:
        raise PublishError("No Spaces were selected")
    candidates = (
        _all_candidates(root, specs)
        if include_all
        else _changed_candidates(root, specs, since or "")
    )
    files: dict[SpaceSpec, dict[str, PublishFile]] = {spec: {} for spec in specs}
    skipped: dict[SpaceSpec, list[SkippedPath]] = {spec: [] for spec in specs}
    for change, raw_path in candidates:
        repo_path = PurePosixPath(raw_path)
        spec = _spec_for_repo_path(specs, repo_path)
        if spec is None:
            raise PublishError(
                f"git returned a path outside selected Spaces: {raw_path}"
            )
        if change == "D":
            skipped[spec].append(SkippedPath(raw_path, "delete preserved remotely"))
            continue
        if not _is_interface_path(repo_path):
            skipped[spec].append(SkippedPath(raw_path, "not an interface file"))
            continue
        publish_file = _validated_file(
            root,
            spec,
            repo_path,
            change="new" if change in {"?", "A"} else change,
        )
        files[spec][publish_file.path_in_repo] = publish_file
    return tuple(
        SpacePlan(
            spec=spec,
            files=tuple(files[spec][name] for name in sorted(files[spec])),
            skipped=tuple(skipped[spec]),
        )
        for spec in specs
    )


def _read_planned_payload(file: PublishFile) -> bytes:
    payload = file.local_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != file.size or digest != file.sha256:
        raise PublishError(
            f"File changed after planning; rerun before applying: {file.local_path}"
        )
    if b"\0" in payload:
        raise PublishError(f"Interface file contains NUL bytes: {file.local_path}")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PublishError(
            f"Interface file is no longer UTF-8 text: {file.local_path}"
        ) from exc
    return payload


def publish_space(
    plan: SpacePlan,
    *,
    api: Any,
    operation_factory: Callable[..., Any],
    commit_message: str,
) -> PublishResult:
    """Create one additive Space commit pinned to its observed parent SHA."""

    if not plan.files:
        raise PublishError(f"No interface files selected for {plan.spec.repo_id}")
    info = api.repo_info(repo_id=plan.spec.repo_id, repo_type="space")
    before_sha = str(getattr(info, "sha", "") or "")
    if not before_sha:
        raise PublishError(f"Hugging Face returned no SHA for {plan.spec.repo_id}")
    operations = [
        operation_factory(
            path_in_repo=file.path_in_repo,
            path_or_fileobj=_read_planned_payload(file),
        )
        for file in plan.files
    ]
    commit = api.create_commit(
        repo_id=plan.spec.repo_id,
        repo_type="space",
        operations=operations,
        commit_message=commit_message,
        parent_commit=before_sha,
    )
    after_sha = str(
        getattr(commit, "oid", "") or getattr(commit, "commit_id", "") or ""
    )
    if not after_sha:
        raise PublishError(
            f"Hugging Face returned no commit SHA for {plan.spec.repo_id}"
        )
    commit_url = str(getattr(commit, "commit_url", "") or "")
    if not commit_url:
        commit_url = (
            f"https://huggingface.co/spaces/{plan.spec.repo_id}/commit/{after_sha}"
        )
    return PublishResult(plan.spec.repo_id, before_sha, after_sha, commit_url)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def print_plan(plans: Iterable[SpacePlan], *, apply: bool) -> None:
    mode = "APPLY" if apply else "DRY RUN (no network, no remote changes)"
    print(f"Mode: {mode}")
    for plan in plans:
        print(f"\n[{plan.spec.benchmark_id}] {plan.spec.source} -> {plan.spec.repo_id}")
        if not plan.files:
            print("  files: 0 (skip)")
        for file in plan.files:
            print(
                f"  {file.change:>7}  {file.path_in_repo}  "
                f"{_format_size(file.size)}  sha256:{file.sha256[:12]}"
            )
        if plan.skipped:
            print(f"  excluded: {len(plan.skipped)} non-publishable path(s)")
            for item in plan.skipped[:5]:
                print(f"    - {item.repo_path}: {item.reason}")
            if len(plan.skipped) > 5:
                print(f"    - ... and {len(plan.skipped) - 5} more")
        if not apply:
            print("  before: not queried")
            print("  after:  unchanged (dry run)")


def _load_huggingface() -> tuple[Any, Callable[..., Any], str]:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, get_token
    except ImportError as exc:
        raise PublishError(
            "huggingface_hub is required for --apply; install the project dependencies"
        ) from exc
    token = get_token()
    if not token:
        raise PublishError(
            "No Hugging Face credential was found in the standard environment or cache"
        )
    return HfApi(token=token), CommitOperationAdd, token


def _redact(message: str, secrets: Iterable[str]) -> str:
    redacted = message
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--since",
        metavar="GIT_REF",
        help="Publish eligible tracked changes since this commit plus untracked files.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Publish every tracked or untracked interface file in selected Spaces.",
    )
    parser.add_argument(
        "--space",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Limit by benchmark ID, repo ID, source path, or source basename; "
            "repeatable."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform one parent-pinned commit per Space. Default: local dry run.",
    )
    parser.add_argument(
        "--message",
        default="Refresh Motius leaderboard interface",
        help="Commit message used for each Space (only with --apply).",
    )
    args = parser.parse_args(argv)
    if args.since is None and not args.all:
        args.since = "HEAD"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    credential = ""
    try:
        args = parse_args(argv)
        all_specs = load_space_manifest(REPO_ROOT)
        specs = select_spaces(all_specs, args.space)
        plans = build_publish_plans(
            REPO_ROOT,
            specs,
            since=args.since,
            include_all=args.all,
        )
        print(f"Manifest: {len(specs)} of {len(all_specs)} Spaces")
        print_plan(plans, apply=args.apply)
        if not args.apply:
            return 0

        api, operation_factory, credential = _load_huggingface()
        uploaded = 0
        for plan in plans:
            if not plan.files:
                continue
            message = f"{args.message} ({plan.spec.benchmark_id})"
            result = publish_space(
                plan,
                api=api,
                operation_factory=operation_factory,
                commit_message=message,
            )
            uploaded += 1
            print(f"\nPublished {result.repo_id}")
            print(f"  before: {result.before_sha}")
            print(f"  after:  {result.after_sha}")
            print(f"  commit: {result.commit_url}")
        print(f"\nPublished {uploaded} Space(s).")
        return 0
    except PublishError as exc:
        secrets = [
            os.environ.get("HF_TOKEN", ""),
            os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
            credential,
        ]
        # The credential is intentionally never printed. Standard token values
        # are removed in the unlikely event an exception echoes one.
        print(f"error: {_redact(str(exc), secrets)}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        secrets = [
            os.environ.get("HF_TOKEN", ""),
            os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
            credential,
        ]
        detail = _redact(str(exc), secrets)
        print(f"error: {type(exc).__name__}: {detail}", file=sys.stderr)
        print("No further Spaces were attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
