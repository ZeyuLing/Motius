#!/usr/bin/env python3
"""Audit task-level demos and canonical metrics in public Model Cards."""

from __future__ import annotations

import json
from pathlib import Path
import re

from PIL import Image

try:
    from tools.normalize_model_cards import (
        REPO_ROOT,
        TASK_REGISTRY,
        _catalog_cards,
        _catalog_task_contracts,
    )
    from tools.sync_model_card_content import (
        DEMO_END,
        DEMO_START,
        FRAME_RATE_CONTRACTS,
        FRAME_RATE_END,
        FRAME_RATE_START,
        MODEL_ZOO_README,
        METRICS_END,
        METRICS_START,
        REPOSITORY_MEDIA_PREFIXES,
        T2M_RESULTS,
        RELEASE_MANIFEST,
        RUNTIME_ONLY_TASKS,
        VIDEO_ATTACHMENTS,
        ZOO_METRICS_END,
        ZOO_METRICS_START,
        _preview_for,
        _preview_metadata,
        sync_all,
    )
except ModuleNotFoundError:
    from normalize_model_cards import (
        REPO_ROOT,
        TASK_REGISTRY,
        _catalog_cards,
        _catalog_task_contracts,
    )
    from sync_model_card_content import (
        DEMO_END,
        DEMO_START,
        FRAME_RATE_CONTRACTS,
        FRAME_RATE_END,
        FRAME_RATE_START,
        MODEL_ZOO_README,
        METRICS_END,
        METRICS_START,
        REPOSITORY_MEDIA_PREFIXES,
        T2M_RESULTS,
        RELEASE_MANIFEST,
        RUNTIME_ONLY_TASKS,
        VIDEO_ATTACHMENTS,
        ZOO_METRICS_END,
        ZOO_METRICS_START,
        _preview_for,
        _preview_metadata,
        sync_all,
    )


def _block(text: str, start: str, end: str) -> str:
    match = re.search(
        rf"{re.escape(start)}(.*?){re.escape(end)}",
        text,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def _table_rows(block: str) -> list[list[str]]:
    rows = []
    for line in block.splitlines():
        if not line.startswith("| ") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] not in {"Task", "Evaluator", "Method"}:
            rows.append(cells)
    return rows


def _animation_duration_seconds(path: Path) -> float:
    duration_ms = 0
    with Image.open(path) as animation:
        for frame in range(animation.n_frames):
            animation.seek(frame)
            duration_ms += int(animation.info.get("duration", 0))
    return duration_ms / 1000.0


def audit_model_card_content() -> list[str]:
    errors: list[str] = []
    cards = _catalog_cards()
    tasks = _catalog_task_contracts()
    if set(FRAME_RATE_CONTRACTS) != set(cards):
        missing = sorted(set(cards) - set(FRAME_RATE_CONTRACTS))
        extra = sorted(set(FRAME_RATE_CONTRACTS) - set(cards))
        errors.append(
            "frame-rate contracts differ from the Model Zoo catalog: "
            f"missing={missing}, extra={extra}"
        )
    drift = set(sync_all(write=False))
    if RELEASE_MANIFEST in drift:
        errors.append("release_manifest.json: canonical metrics are stale")
    if MODEL_ZOO_README in drift:
        errors.append("README.md: canonical benchmark snapshot is stale")
    zoo_text = MODEL_ZOO_README.read_text(encoding="utf-8")
    zoo_metrics = _block(zoo_text, ZOO_METRICS_START, ZOO_METRICS_END)
    if not zoo_metrics:
        errors.append("README.md: missing canonical benchmark snapshot")
    elif "Motius FID (normalized)" not in zoo_metrics:
        errors.append("README.md: normalized Motius FID is not visible")
    t2m_payload = json.loads(T2M_RESULTS.read_text(encoding="utf-8"))
    task_labels = {
        task["id"]: task["label"] for task in TASK_REGISTRY["tasks"]
    }
    video_manifest = json.loads(
        VIDEO_ATTACHMENTS.read_text(encoding="utf-8")
    )
    video_entries = video_manifest.get("videos", {})
    published_video_urls = {
        entry["url"] for entry in video_entries.values()
    }
    public_verification = video_manifest.get("public_verification", {})
    private_verification = video_manifest.get("private_verification", {})
    public_complete = (
        public_verification.get("count") == len(video_entries)
    )
    private_complete = (
        private_verification.get("count") == len(video_entries)
        and private_verification.get("repository_visibility")
        in {"private", "internal"}
    )
    if not (public_complete or private_complete):
        errors.append(
            "video_attachments.json: visibility-appropriate verification "
            "count does not cover every video"
        )
    if not re.fullmatch(
        r"https://github\.com/ZeyuLing/Motius/issues/\d+",
        str(video_manifest.get("registry_url", "")),
    ):
        errors.append(
            "video_attachments.json: managed media registry URL is missing"
        )
    forbidden_raw_fids = {
        float(row["utmrFID"])
        for row in t2m_payload["semantic_rows"]
        if row.get("utmrFID") not in {None, 0}
        and (
            row.get("utmrFIDNorm") is None
            or abs(row["utmrFID"] - row["utmrFIDNorm"]) > 0.001
        )
    }

    for package, path in sorted(cards.items()):
        text = path.read_text(encoding="utf-8")
        if path in drift:
            errors.append(f"{path.name}: generated content is stale")
        for prefix in REPOSITORY_MEDIA_PREFIXES:
            if prefix in text:
                errors.append(
                    f"{path.name}: hardcodes default-branch media {prefix}"
                )
        if re.search(
            r"^\|\s*(?:HumanML3D Sample|AIST\+\+ (?:Sample|Case)|Sample)"
            r"\s*\|",
            text,
            flags=re.MULTILINE,
        ):
            errors.append(f"{path.name}: demo table exposes dataset case IDs")
        frame_rate = _block(text, FRAME_RATE_START, FRAME_RATE_END)
        contract = FRAME_RATE_CONTRACTS.get(package, {})
        if not frame_rate:
            errors.append(f"{path.name}: missing frame-rate contract")
        else:
            if frame_rate.count("| Training motion |") != 1:
                errors.append(
                    f"{path.name}: must declare one training motion clock"
                )
            if frame_rate.count("| Public preview |") != 1:
                errors.append(
                    f"{path.name}: must declare one public preview clock"
                )
            for field in ("training", "preview"):
                value = contract.get(field)
                if value and value not in frame_rate:
                    errors.append(
                        f"{path.name}: stale frame-rate {field} value"
                    )
        visual_parts = text.split("## Visual Results", 1)
        visual = (
            visual_parts[1].split("\n## ", 1)[0]
            if len(visual_parts) == 2
            else ""
        )
        if re.search(r"!\[[^\]]*\]\([^)]+\)|<img\s", visual):
            errors.append(
                f"{path.name}: Visual Results still contains an image "
                "instead of a video player"
            )
        video_tags = re.findall(r"<video\s+[^>]*>", visual)
        valid_video_urls = re.findall(
            r'<video\s+[^>]*src="'
            r"(https://github\.com/user-attachments/assets/"
            r'[0-9a-fA-F-]{36})"[^>]*controls[^>]*>',
            visual,
        )
        if len(video_tags) != len(valid_video_urls):
            errors.append(
                f"{path.name}: Visual Results contains a non-native or "
                "control-less video"
            )
        for url in valid_video_urls:
            if url not in published_video_urls:
                errors.append(
                    f"{path.name}: Visual Results video is absent from the "
                    f"attachment manifest: {url}"
                )

        demos = _block(text, DEMO_START, DEMO_END)
        if not demos:
            errors.append(f"{path.name}: missing task demo block")
        if "Motius Three.js viewer" in demos:
            errors.append(
                f"{path.name}: exposes a viewer label instead of inline media"
            )
        if "Open all-case demo" in demos:
            errors.append(
                f"{path.name}: uses a text-only demo as its primary preview"
            )
        for task in tasks[package]:
            method = f"`infer_{task}`"
            if method not in text:
                errors.append(f"{path.name}: missing public API {method}")
            if task in RUNTIME_ONLY_TASKS:
                continue
            preview = _preview_for(package, task)
            metadata = _preview_metadata(preview)
            if metadata.get("schema_version") == 1 and "task" in metadata:
                if metadata.get("task") != task:
                    errors.append(
                        f"{path.name}: {preview.name} metadata declares "
                        f"task {metadata.get('task')!r}, expected {task!r}"
                    )
                if not str(
                    metadata.get("input") or metadata.get("caption") or ""
                ).strip():
                    errors.append(
                        f"{path.name}: {preview.name} has no exact input"
                    )
                if task in {
                    "temporal_motion_completion",
                    "sequential_text_to_motion",
                    "kinematic_motion_control",
                    "part_level_motion_control",
                } and not str(
                    metadata.get("condition_visualization", "")
                ).strip():
                    errors.append(
                        f"{path.name}: {preview.name} does not describe how "
                        "the input condition is rendered"
                    )
                if int(metadata.get("capture_frame_step", 1)) > 1:
                    expected_duration = metadata.get(
                        "preview_duration_seconds"
                    )
                    if not isinstance(expected_duration, (int, float)):
                        errors.append(
                            f"{path.name}: {preview.name} skips capture "
                            "frames without declaring preview duration"
                        )
                    else:
                        actual_duration = _animation_duration_seconds(preview)
                        if abs(actual_duration - expected_duration) > 0.05:
                            errors.append(
                                f"{path.name}: {preview.name} duration "
                                f"{actual_duration:.3f}s differs from "
                                f"declared {expected_duration:.3f}s"
                            )
        demo_rows = _table_rows(demos)
        expected_rows = max(1, len(tasks[package]))
        if len(demo_rows) != expected_rows:
            errors.append(
                f"{path.name}: expected {expected_rows} task demo row(s), "
                f"found {len(demo_rows)}"
            )
        expected_labels = (
            [task_labels[task] for task in tasks[package]]
            if tasks[package]
            else ["No registered public task"]
        )
        actual_labels = [row[0] for row in demo_rows]
        if actual_labels != expected_labels:
            errors.append(
                f"{path.name}: task demo labels differ from the catalog: "
                f"{actual_labels!r}"
            )
        task_media: list[str] = []
        runtime_labels = {
            task_labels[task]
            for task in tasks[package]
            if task in RUNTIME_ONLY_TASKS
        }
        for row in demo_rows:
            if len(row) < 4:
                errors.append(f"{path.name}: malformed task demo row {row!r}")
                continue
            video = re.search(
                r'<video\s+[^>]*src="'
                r"(https://github\.com/user-attachments/assets/"
                r'[0-9a-fA-F-]{36})"[^>]*controls[^>]*>',
                row[2],
            )
            if row[0] in runtime_labels and video is None:
                if "pending a shared simulator protocol" not in row[2]:
                    errors.append(
                        f"{path.name}: {row[0]} lacks the runtime-only "
                        "preview disclosure"
                    )
                continue
            if row[0] != "No registered public task" and video is None:
                errors.append(
                    f"{path.name}: {row[0]} has no GitHub video player"
                )
                continue
            if video is None:
                continue
            target = video.group(1)
            task_media.append(target)
            if target not in published_video_urls:
                errors.append(
                    f"{path.name}: {row[0]} video is absent from the "
                    "attachment manifest"
                )
        duplicate_media = {
            target for target in task_media if task_media.count(target) > 1
        }
        if duplicate_media:
            errors.append(
                f"{path.name}: reuses one video across public tasks: "
                + ", ".join(sorted(duplicate_media))
            )
        for target in re.findall(r"\[MP4\]\(([^)]+)\)", demos):
            if target not in published_video_urls:
                errors.append(
                    f"{path.name}: MP4 link bypasses the audited attachment "
                    f"manifest: {target}"
                )
        media_targets = re.findall(
            r"!\[[^\]]*\]\(([^)]+)\)|"
            r"<img\s+[^>]*src=[\"']([^\"']+)[\"']",
            text,
        )
        for markdown_target, html_target in media_targets:
            target = markdown_target or html_target
            local: Path | None = None
            if target.startswith(
                "https://raw.githubusercontent.com/ZeyuLing/Motius/main/"
            ):
                local = REPO_ROOT / target.split("/main/", 1)[1]
            elif not target.startswith(("http://", "https://")):
                local = (path.parent / target.split("#", 1)[0]).resolve()
            if local is not None and not local.is_file():
                errors.append(
                    f"{path.name}: missing inline media target {target}"
                )
        for target in re.findall(
            r"\]\((\.\./\.\./assets/[^)]+)\)",
            text,
        ):
            local = (path.parent / target).resolve()
            if not local.is_file():
                errors.append(
                    f"{path.name}: missing linked media target {target}"
                )

        metrics = _block(text, METRICS_START, METRICS_END)
        if not metrics:
            errors.append(f"{path.name}: missing canonical metric disclosure")
        if "text_to_motion" in tasks[package]:
            table_values = [
                float(value)
                for value in re.findall(
                    r"\|\s*(-?\d+(?:\.\d+)?)\s*(?=\|)",
                    text,
                )
            ]
            for raw in forbidden_raw_fids:
                if any(abs(value - raw) < 0.001 for value in table_values):
                    errors.append(
                        f"{path.name}: displays raw uTMR FID {raw:.4f} "
                        "as a table value"
                    )
    return errors


def main() -> int:
    errors = audit_model_card_content()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"{len(errors)} Model Card content error(s)")
        return 1
    print(f"{len(_catalog_cards())} Model Cards satisfy the content contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
