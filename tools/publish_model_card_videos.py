#!/usr/bin/env python3
"""Publish Model Card renders as GitHub-native video attachments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import List, Optional

from PIL import Image

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise SystemExit("imageio-ffmpeg is required to transcode previews") from exc

try:
    from tools.audit_model_card_media import _audit_one
    from tools.normalize_model_cards import (
        REPO_ROOT,
        _catalog_cards,
        _catalog_task_contracts,
    )
    from tools.sync_model_card_content import (
        VIDEO_ATTACHMENTS,
        _matching_mp4,
        _preview_for,
    )
except ModuleNotFoundError:
    from audit_model_card_media import _audit_one
    from normalize_model_cards import (
        REPO_ROOT,
        _catalog_cards,
        _catalog_task_contracts,
    )
    from sync_model_card_content import (
        VIDEO_ATTACHMENTS,
        _matching_mp4,
        _preview_for,
    )


REPOSITORY = "ZeyuLing/Motius"
MEDIA_REGISTRY_TITLE = "Managed Model Card media registry"
ANIMATION_PATTERN = re.compile(
    r"!\[[^\]]*\]\("
    r"(\.\./\.\./assets/model_zoo/[^)]+\.(?:gif|webp))\)",
    flags=re.IGNORECASE,
)
HTML_ANIMATION_PATTERN = re.compile(
    r'<img\s+[^>]*src="'
    r'(\.\./\.\./assets/model_zoo/[^"]+\.(?:gif|webp))'
    r'"[^>]*>',
    flags=re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    if VIDEO_ATTACHMENTS.is_file():
        return json.loads(VIDEO_ATTACHMENTS.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "videos": {},
    }


def _write_manifest(payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = VIDEO_ATTACHMENTS.with_suffix(".json.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(VIDEO_ATTACHMENTS)


def _replace_published_urls(replacements: dict[str, str]) -> int:
    """Rewrite stale attachment URLs in Model Cards after a re-upload."""
    replacements = {
        old: new
        for old, new in replacements.items()
        if old and new and old != new
    }
    changed = 0
    for path in sorted(_catalog_cards().values()):
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated == text:
            continue
        path.write_text(updated, encoding="utf-8")
        changed += 1
    return changed


def _finish_pending_url_replacements(manifest: dict) -> int:
    entries = manifest.setdefault("videos", {})
    replacements = {
        entry["previous_url"]: entry["url"]
        for entry in entries.values()
        if isinstance(entry.get("previous_url"), str)
        and isinstance(entry.get("url"), str)
    }
    if not replacements:
        return 0
    changed = _replace_published_urls(replacements)
    for entry in entries.values():
        entry.pop("previous_url", None)
    _write_manifest(manifest)
    return changed


def _animation_sources() -> list[Path]:
    sources: set[Path] = set()
    manifest = _load_manifest()
    for target in manifest.get("videos", {}):
        source = REPO_ROOT / target
        if source.is_file():
            sources.add(source.resolve())
    cards = _catalog_cards()
    for card in cards.values():
        text = card.read_text(encoding="utf-8")
        targets = ANIMATION_PATTERN.findall(text)
        targets.extend(HTML_ANIMATION_PATTERN.findall(text))
        for target in targets:
            sources.add((card.parent / target).resolve())
    for package, tasks in _catalog_task_contracts().items():
        for task in tasks:
            try:
                preview = _preview_for(package, task)
            except FileNotFoundError:
                # Missing previews are reported by the media audit. Do not let
                # one incomplete card block publishing valid videos elsewhere.
                continue
            sources.add(preview.resolve())
    return sorted(sources)


def _fps_for(source: Path) -> int:
    match = re.search(r"_(20|30)fps", source.name)
    if match:
        return int(match.group(1))
    with Image.open(source) as animation:
        duration_ms = animation.info.get("duration")
    if duration_ms:
        return max(1, min(30, round(1000 / duration_ms)))
    return 30


def _transcode(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = _fps_for(source)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        (
            f"fps={fps},"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,"
            "format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def _source_video(animation: Path, output_root: Path) -> Path:
    relative = animation.relative_to(REPO_ROOT).with_suffix(".mp4")
    output = output_root / relative
    if (
        output.is_file()
        and output.stat().st_mtime >= animation.stat().st_mtime
    ):
        return output
    existing = _matching_mp4(animation)
    if existing is not None:
        return existing
    if (
        not output.is_file()
        or output.stat().st_mtime < animation.stat().st_mtime
    ):
        _transcode(animation, output)
    return output


def _render_metadata(video: Path) -> dict:
    metadata_path = video.with_suffix(".render.json")
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    allowed = {
        "render_backend",
        "render_profile",
        "representation",
        "frames",
        "fps",
        "width",
        "height",
        "floor",
        "audio",
    }
    return {
        key: payload[key]
        for key in allowed
        if key in payload
    }


def _github_context() -> tuple[str, str]:
    token = subprocess.check_output(
        ["gh", "auth", "token"],
        text=True,
    ).strip()
    repository_id = subprocess.check_output(
        ["gh", "api", f"repos/{REPOSITORY}", "--jq", ".id"],
        text=True,
    ).strip()
    return token, repository_id


def _repository_visibility() -> str:
    visibility = subprocess.check_output(
        [
            "gh",
            "api",
            f"repos/{REPOSITORY}",
            "--jq",
            ".visibility",
        ],
        text=True,
    ).strip()
    if visibility not in {"public", "private", "internal"}:
        raise RuntimeError(
            f"Unexpected repository visibility: {visibility!r}"
        )
    return visibility


def _upload(path: Path, token: str, repository_id: str) -> str:
    query = urlencode(
        {
            "name": path.name,
            "content_type": "video/mp4",
            "repository_id": repository_id,
        }
    )
    upload_url = (
        f"https://uploads.github.com/user-attachments/assets?{query}"
    )
    video_bytes = path.read_bytes()
    for attempt in range(8):
        request = Request(
            upload_url,
            data=video_bytes,
            method="POST",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            url = payload["url"]
            if not re.fullmatch(
                r"https://github\.com/user-attachments/assets/"
                r"[0-9a-fA-F-]{36}",
                url,
            ):
                raise RuntimeError(f"unexpected upload URL: {url}")
            return url
        except Exception as exc:
            if attempt == 7:
                raise
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 422 and "file extension" in detail:
                    raise RuntimeError(detail) from exc
            delay = min(60, 2 ** (attempt + 1))
            print(
                f"upload retry {attempt + 1}/7 for {path.name} in "
                f"{delay}s: {exc} {detail}".rstrip(),
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _registry_body(entries: dict[str, dict[str, str]]) -> str:
    lines = [
        (
            "This closed issue anchors GitHub-hosted video attachments used "
            "by the public Model Cards."
        ),
        "",
        (
            "It is generated by `tools/publish_model_card_videos.py`; "
            "manual edits will be overwritten."
        ),
        "",
    ]
    for source, entry in sorted(entries.items()):
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        lines.extend(
            [
                f"<!-- {source} -->",
                f'<video src="{url}" controls></video>',
                "",
            ]
        )
    return "\n".join(lines)


def _sync_media_registry(manifest: dict) -> str:
    issue_number = manifest.get("registry_issue_number")
    if issue_number is None:
        issues = json.loads(
            subprocess.check_output(
                [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    REPOSITORY,
                    "--state",
                    "all",
                    "--limit",
                    "100",
                    "--json",
                    "number,title",
                ],
                text=True,
            )
        )
        match = next(
            (
                issue
                for issue in issues
                if issue.get("title") == MEDIA_REGISTRY_TITLE
            ),
            None,
        )
        if match is None:
            issue_number = int(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        f"repos/{REPOSITORY}/issues",
                        "-f",
                        f"title={MEDIA_REGISTRY_TITLE}",
                        "-f",
                        "body=Initializing managed media registry.",
                        "--jq",
                        ".number",
                    ],
                    text=True,
                ).strip()
            )
        else:
            issue_number = int(match["number"])

    body = _registry_body(manifest["videos"])
    subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{REPOSITORY}/issues/{issue_number}",
            "-f",
            f"title={MEDIA_REGISTRY_TITLE}",
            "-f",
            f"body={body}",
            "-f",
            "state=closed",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    manifest["registry_issue_number"] = int(issue_number)
    registry_url = (
        f"https://github.com/{REPOSITORY}/issues/{int(issue_number)}"
    )
    manifest["registry_url"] = registry_url
    return registry_url


def _verify_public_video(url: str) -> None:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "video/mp4,*/*",
            "Range": "bytes=0-0",
            "User-Agent": "Motius-model-card-media-audit",
        },
    )
    with urlopen(request, timeout=60) as response:
        status = int(response.status)
        content_type = response.headers.get_content_type()
        response.read(1)
    if status not in {200, 206} or content_type != "video/mp4":
        raise RuntimeError(
            f"attachment is not publicly playable: {status} "
            f"{content_type} {url}"
        )


def _is_public_video(url: str) -> bool:
    try:
        _verify_public_video(url)
    except Exception:
        return False
    return True


def _verify_public_video_eventually(
    url: str,
    *,
    attempts: int = 6,
) -> None:
    """Allow newly uploaded GitHub attachments time to reach the public CDN."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            _verify_public_video(url)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(min(16, 2**attempt))


def _public_video_urls(entries: dict[str, dict[str, str]]) -> set[str]:
    urls = {
        entry["url"]
        for entry in entries.values()
        if isinstance(entry.get("url"), str)
    }
    public = set()
    with ThreadPoolExecutor(max_workers=8) as executor:
        pending = {
            executor.submit(_is_public_video, url): url
            for url in urls
        }
        for future in as_completed(pending):
            url = pending[future]
            if future.result():
                public.add(url)
    return public


def _verify_manifest_videos(entries: dict[str, dict[str, str]]) -> int:
    failures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        pending = {
            executor.submit(
                _verify_public_video_eventually,
                entry["url"],
            ): source
            for source, entry in entries.items()
        }
        for future in as_completed(pending):
            source = pending[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{source}: {exc}")
    if failures:
        raise RuntimeError(
            "GitHub attachment verification failed:\n"
            + "\n".join(sorted(failures))
        )
    return len(entries)


def _verify_private_registry(
    manifest: dict,
    entries: dict[str, dict[str, str]],
) -> int:
    issue_number = manifest.get("registry_issue_number")
    if issue_number is None:
        raise RuntimeError("Private media registry issue is missing")
    body = subprocess.check_output(
        [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--repo",
            REPOSITORY,
            "--json",
            "body",
            "--jq",
            ".body",
        ],
        text=True,
    )
    missing = [
        source
        for source, entry in entries.items()
        if not isinstance(entry.get("url"), str)
        or entry["url"] not in body
    ]
    if missing:
        raise RuntimeError(
            "Private media registry is missing attachment anchors:\n"
            + "\n".join(sorted(missing))
        )
    return len(entries)


def publish(
    output_root: Path,
    sources: Optional[List[Path]] = None,
) -> tuple[int, int, int]:
    manifest = _load_manifest()
    entries = manifest.setdefault("videos", {})
    _finish_pending_url_replacements(manifest)
    token, repository_id = _github_context()
    visibility = _repository_visibility()
    public_urls = (
        _public_video_urls(entries)
        if visibility == "public"
        else {
            entry["url"]
            for entry in entries.values()
            if isinstance(entry.get("url"), str)
        }
    )
    uploaded = 0
    skipped = 0
    animations = _animation_sources() if sources is None else sources
    for index, animation in enumerate(animations, start=1):
        source = animation.relative_to(REPO_ROOT).as_posix()
        video = _source_video(animation, output_root)
        _audit_one(source, output_root)
        source_sha256 = _sha256(animation)
        video_sha256 = _sha256(video)
        current = entries.get(source, {})
        render_metadata = _render_metadata(video)
        if (
            current.get("source_sha256") == source_sha256
            and current.get("video_sha256") == video_sha256
            and current.get("url")
            and current["url"] in public_urls
        ):
            updated = {**current, **render_metadata}
            if updated != current:
                entries[source] = updated
                _write_manifest(manifest)
            skipped += 1
            print(f"[{index:03d}] cached {source}", flush=True)
            continue
        if current.get("url"):
            reason = (
                "stale content"
                if current.get("source_sha256") != source_sha256
                or current.get("video_sha256") != video_sha256
                else "inaccessible attachment"
            )
            print(
                f"[{index:03d}] replacing {reason} for {source}",
                flush=True,
            )
        url = _upload(video, token, repository_id)
        replacement = {
            "source_sha256": source_sha256,
            "video_sha256": video_sha256,
            "url": url,
            **render_metadata,
        }
        if current.get("url") and current["url"] != url:
            replacement["previous_url"] = current["url"]
        entries[source] = replacement
        _write_manifest(manifest)
        _finish_pending_url_replacements(manifest)
        uploaded += 1
        print(f"[{index:03d}] published {source} -> {url}", flush=True)
    registry_url = _sync_media_registry(manifest)
    _write_manifest(manifest)
    if visibility == "public":
        verified = _verify_manifest_videos(entries)
        manifest["public_verification"] = {
            "count": verified,
            "method": "anonymous HTTP Range GET",
            "expected_content_type": "video/mp4",
        }
        manifest.pop("private_verification", None)
    else:
        verified = _verify_private_registry(manifest, entries)
        manifest["private_verification"] = {
            "count": verified,
            "method": (
                "authenticated registry anchor plus local source/video digest"
            ),
            "repository_visibility": visibility,
        }
        manifest.pop("public_verification", None)
    _write_manifest(manifest)
    print(f"Verified {verified} managed video(s) via {registry_url}")
    return uploaded, skipped, verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/model_card_video_uploads",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        help=(
            "Publish only this repository-relative GIF/WebP source. "
            "May be supplied more than once."
        ),
    )
    args = parser.parse_args()
    sources = None
    if args.source:
        sources = []
        for source in args.source:
            resolved = source if source.is_absolute() else REPO_ROOT / source
            resolved = resolved.resolve()
            if not resolved.is_file():
                parser.error(f"source does not exist: {source}")
            try:
                resolved.relative_to(REPO_ROOT)
            except ValueError:
                parser.error(f"source is outside the repository: {source}")
            sources.append(resolved)
    uploaded, skipped, verified = publish(args.output_dir, sources=sources)
    print(
        f"Published {uploaded} video(s); reused {skipped} cached video(s); "
        f"verified {verified} managed video(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
