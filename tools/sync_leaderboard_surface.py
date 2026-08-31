#!/usr/bin/env python3
"""Install the shared Motius visual surface in every static leaderboard Space.

The Space directories are deployed independently, so each one receives an exact
copy of the canonical stylesheet.  Every HTML document then links to that local
copy with a path relative to its own directory.  The generated hooks are kept
small and marker-delimited so the synchronizer is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import html
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_ROOT = ROOT / "docs" / "leaderboards"
CANONICAL_STYLE = ROOT / "tools" / "motius_surface.css"
SPACE_GLOB = "hf_space_*"
SPACE_STYLE_NAME = "motius-surface.css"

STYLE_START = "<!-- motius-surface:start -->"
STYLE_END = "<!-- motius-surface:end -->"
SKIP_START = "<!-- motius-skip-link:start -->"
SKIP_END = "<!-- motius-skip-link:end -->"
THEME_START = "<!-- motius-theme-color:start -->"
THEME_END = "<!-- motius-theme-color:end -->"
THEME_COLOR = "#111827"

CANVAS_LABELS = {
    "bar-chart": "Leaderboard metric comparison bar chart",
    "radar-chart": "Leaderboard normalized metric radar chart",
    "metric-chart": "Method metric comparison chart",
    "official-chart": "Official evaluator metric comparison chart",
    "utmr-chart": "Universal TMR metric comparison chart",
    "beat-chart": "Beat alignment metric comparison chart",
    "scene": "Interactive 3D motion comparison",
    "waveform": "Audio waveform",
    "canvas": "Interactive 3D motion preview",
}

TABLE_LABELS = {
    "semantic": "Semantic evaluation results",
    "physical": "Physical evaluation results",
    "paper": "Reported paper results",
    "calibration": "Evaluator calibration results",
    "control-text": "Text control evaluation results",
    "control-motion": "Motion control evaluation results",
    "temporal": "Temporal conditioning evaluation results",
}


def space_roots() -> list[Path]:
    """Return all independently deployed static leaderboard Space roots."""

    return sorted(path for path in LEADERBOARD_ROOT.glob(SPACE_GLOB) if path.is_dir())


def page_type(path: Path, space: Path) -> str:
    """Classify a page for scoped dashboard, viewer, or audit treatments."""

    relative = path.relative_to(space)
    if relative == Path("index.html"):
        return "dashboard"
    if relative.parts[0] == "audit":
        return "audit"
    return "viewer"


def _replace_marked_block(text: str, start: str, end: str, block: str) -> str:
    pattern = re.escape(start) + r".*?" + re.escape(end)
    return re.sub(pattern, block, text, count=1, flags=re.DOTALL)


def _surface_link(path: Path, space: Path) -> str:
    href = Path(os.path.relpath(space / SPACE_STYLE_NAME, path.parent)).as_posix()
    return "\n".join(
        [
            STYLE_START,
            f'<link rel="stylesheet" href="{href}" data-motius-surface>',
            STYLE_END,
        ]
    )


def _add_attribute(tag: str, name: str, value: str) -> str:
    """Add or replace one quoted attribute on an opening HTML tag."""

    pattern = rf"\s{name}\s*=\s*([\"']).*?\1"
    replacement = f' {name}="{value}"'
    if re.search(pattern, tag, flags=re.IGNORECASE):
        return re.sub(pattern, replacement, tag, count=1, flags=re.IGNORECASE)
    return tag[:-1] + replacement + ">"


def _attribute_value(tag: str, name: str) -> str | None:
    """Return one quoted attribute value without changing the tag."""

    match = re.search(
        rf"\s{name}\s*=\s*([\"'])(.*?)\1", tag, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(2) if match else None


def _has_accessible_name(tag: str) -> bool:
    return any(
        _attribute_value(tag, name) for name in ("aria-label", "aria-labelledby")
    )


def _ensure_theme_color(text: str, path: Path) -> str:
    """Install one marker-owned browser theme color in the document head."""

    block = "\n".join(
        [
            THEME_START,
            f'<meta name="theme-color" content="{THEME_COLOR}">',
            THEME_END,
        ]
    )
    if THEME_START in text and THEME_END in text:
        return _replace_marked_block(text, THEME_START, THEME_END, block)

    existing = re.search(
        r"<meta\b(?=[^>]*\bname\s*=\s*([\"'])theme-color\1)[^>]*>",
        text,
        flags=re.IGNORECASE,
    )
    if existing:
        return text[: existing.start()] + block + text[existing.end() :]

    text, count = re.subn(
        r"</head>",
        f"{block}\n</head>",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise ValueError(f"{path}: missing </head>")
    return text


def _ensure_main_landmark(text: str, path: Path) -> str:
    """Promote an existing page shell when a semantic main element is absent."""

    if re.search(r"<main\b", text, flags=re.IGNORECASE):
        return text
    for match in re.finditer(r"<(?:div|section)\b[^>]*>", text, re.IGNORECASE):
        tag = match.group(0)
        roles = (_attribute_value(tag, "role") or "").lower().split()
        if "main" in roles:
            return text
        classes = (_attribute_value(tag, "class") or "").lower().split()
        if "shell" in classes:
            updated = _add_attribute(tag, "role", "main")
            return text[: match.start()] + updated + text[match.end() :]
    raise ValueError(f"{path}: page has no <main> element or promotable .shell")


def _main_landmark(text: str) -> re.Match[str] | None:
    main = re.search(r"<main\b[^>]*>", text, flags=re.IGNORECASE)
    if main:
        return main
    for match in re.finditer(
        r"<(?:div|section|article|header)\b[^>]*>", text, re.IGNORECASE
    ):
        if "main" in (_attribute_value(match.group(0), "role") or "").lower().split():
            return match
    return None


def _content_target(text: str) -> tuple[str, str]:
    """Ensure the first main landmark is a focus target for the skip link."""

    target_match = _main_landmark(text)
    if target_match is None:
        raise ValueError("page has no main landmark")

    tag = target_match.group(0)
    id_match = re.search(r"\sid\s*=\s*([\"'])(.*?)\1", tag, flags=re.IGNORECASE)
    target = id_match.group(2) if id_match else "motius-main"
    if id_match is None and re.search(
        rf"\sid\s*=\s*([\"']){re.escape(target)}\1", text, re.IGNORECASE
    ):
        suffix = 2
        while re.search(
            rf"\sid\s*=\s*([\"']){re.escape(target)}-{suffix}\1",
            text,
            re.IGNORECASE,
        ):
            suffix += 1
        target = f"{target}-{suffix}"
    updated = tag
    if id_match is None:
        updated = _add_attribute(updated, "id", target)
    updated = _add_attribute(updated, "tabindex", "-1")
    return text[: target_match.start()] + updated + text[target_match.end() :], target


def _canvas_label(tag: str, kind: str) -> str:
    canvas_id = (_attribute_value(tag, "id") or "").lower()
    if canvas_id in CANVAS_LABELS:
        return CANVAS_LABELS[canvas_id]
    classes = (_attribute_value(tag, "class") or "").lower().split()
    if "render-canvas" in classes or kind == "viewer":
        return "Interactive 3D motion preview"
    return "Leaderboard visualization"


def _ensure_canvas_names(text: str, kind: str) -> str:
    """Give every rendered canvas an explicit image role and accessible name."""

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        if _attribute_value(tag, "role") is None:
            tag = _add_attribute(tag, "role", "img")
        if not _has_accessible_name(tag):
            tag = _add_attribute(tag, "aria-label", _canvas_label(tag, kind))
        return tag

    return re.sub(r"<canvas\b[^>]*>", replace, text, flags=re.IGNORECASE)


def _plain_heading(markup: str) -> str:
    plain = re.sub(r"<[^>]+>", " ", markup)
    return " ".join(html.unescape(plain).split())


def _table_label(tag: str, prefix: str) -> str:
    data_table = (_attribute_value(tag, "data-table") or "").lower()
    if data_table in TABLE_LABELS:
        return TABLE_LABELS[data_table]
    classes = (_attribute_value(tag, "class") or "").lower().split()
    if "parity" in classes:
        return "Reproducibility parity report"
    if "leaderboard" in classes:
        return "Leaderboard results"

    headings = list(
        re.finditer(r"<h([2-4])\b[^>]*>(.*?)</h\1>", prefix, re.IGNORECASE | re.DOTALL)
    )
    if headings:
        heading = _plain_heading(headings[-1].group(2))
        if heading:
            return f"{heading} table"
    return "Leaderboard results"


def _ensure_table_names(text: str) -> str:
    """Give every data table a stable name without changing its contents."""

    original = text

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        if _has_accessible_name(tag):
            return tag
        return _add_attribute(
            tag, "aria-label", _table_label(tag, original[: match.start()])
        )

    return re.sub(r"<table\b[^>]*>", replace, text, flags=re.IGNORECASE)


def transform_page(path: Path, space: Path, original: str) -> str:
    """Return one page with idempotent surface and accessibility hooks."""

    text = original
    text = _ensure_theme_color(text, path)
    link = _surface_link(path, space)
    if STYLE_START in text and STYLE_END in text:
        text = _replace_marked_block(text, STYLE_START, STYLE_END, link)
    else:
        text, count = re.subn(
            r"</head>",
            f"{link}\n</head>",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise ValueError(f"{path}: missing </head>")

    body_match = re.search(r"<body\b[^>]*>", text, flags=re.IGNORECASE)
    if body_match is None:
        raise ValueError(f"{path}: missing <body>")
    body_tag = _add_attribute(
        body_match.group(0), "data-motius-page", page_type(path, space)
    )
    text = text[: body_match.start()] + body_tag + text[body_match.end() :]

    text = _ensure_main_landmark(text, path)
    text, target = _content_target(text)
    kind = page_type(path, space)
    text = _ensure_canvas_names(text, kind)
    text = _ensure_table_names(text)
    skip = "\n".join(
        [
            SKIP_START,
            f'<a class="motius-skip-link" href="#{target}">Skip to main content</a>',
            SKIP_END,
        ]
    )
    if SKIP_START in text and SKIP_END in text:
        text = _replace_marked_block(text, SKIP_START, SKIP_END, skip)
    else:
        text, count = re.subn(
            r"(<body\b[^>]*>)",
            rf"\1\n{skip}",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise ValueError(f"{path}: could not insert skip link")
    return text


def pending_changes() -> dict[Path, str]:
    """Build a map of files whose desired content differs from disk."""

    canonical = CANONICAL_STYLE.read_text(encoding="utf-8")
    changes: dict[Path, str] = {}
    roots = space_roots()
    if not roots:
        raise RuntimeError(f"No Space roots found below {LEADERBOARD_ROOT}")

    for space in roots:
        local_style = space / SPACE_STYLE_NAME
        current_style = (
            local_style.read_text(encoding="utf-8") if local_style.exists() else None
        )
        if current_style != canonical:
            changes[local_style] = canonical

        for page in sorted(space.rglob("*.html")):
            original = page.read_text(encoding="utf-8")
            transformed = transform_page(page, space, original)
            if transformed != original:
                changes[page] = transformed
    return changes


def synchronize(*, check: bool = False) -> list[Path]:
    """Synchronize the surface; in check mode only report stale files."""

    changes = pending_changes()
    if not check:
        for path, content in changes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    return sorted(changes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any Space is missing the current shared surface.",
    )
    args = parser.parse_args()

    changed = synchronize(check=args.check)
    if changed:
        action = "out of date" if args.check else "updated"
        print(f"{len(changed)} leaderboard surface files {action}:")
        for path in changed:
            print(f"  {path.relative_to(ROOT)}")
        if args.check:
            raise SystemExit(1)
    else:
        print("Leaderboard visual surface is synchronized.")


if __name__ == "__main__":
    main()
