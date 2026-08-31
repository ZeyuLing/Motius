#!/usr/bin/env python3
"""Synchronize the navigation shell across every published leaderboard page."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs/leaderboards/catalog.json"
TAXONOMY_PATH = ROOT / "docs/tasks/taxonomy.json"
GITHUB_DOCS_BASE = "https://github.com/ZeyuLing/Motius/blob/main/docs/tasks/"

START = "<!-- motius-benchmark-nav:start -->"
END = "<!-- motius-benchmark-nav:end -->"
STYLE_START = "<!-- motius-benchmark-nav-style:start -->"
STYLE_END = "<!-- motius-benchmark-nav-style:end -->"
SETTINGS_START = "<!-- motius-benchmark-settings:start -->"
SETTINGS_END = "<!-- motius-benchmark-settings:end -->"

STATUS_LABELS = {
    "complete": "Complete",
    "metrics_complete": "Metrics ready",
    "protocol_only": "Protocol",
    "protocol_ready": "Protocol ready",
    "paused": "Paused",
}

STYLE = """
<style id="motius-benchmark-nav-style">
  .motius-benchmark-nav {
    position: relative;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    width: min(1420px, calc(100% - 36px));
    min-height: 50px;
    margin: 14px auto 0;
    padding: 8px 10px 8px 14px;
    border: 1px solid #d8dfdc;
    border-radius: 7px;
    background: #ffffff;
    color: #17211e;
    box-shadow: 0 5px 18px rgba(23, 33, 30, 0.06);
    font: 13px/1.4 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    letter-spacing: 0;
  }
  .motius-benchmark-nav *,
  .motius-benchmark-nav *::before,
  .motius-benchmark-nav *::after { box-sizing: border-box; }
  .motius-benchmark-brand {
    color: #0a746b;
    font-weight: 760;
    text-decoration: none;
    white-space: nowrap;
  }
  .motius-benchmark-context {
    min-width: 0;
    margin-left: auto;
    overflow: hidden;
    color: #53615c;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .motius-benchmark-menu { position: relative; flex: 0 0 auto; }
  .motius-benchmark-menu > summary {
    display: flex;
    align-items: center;
    min-height: 34px;
    padding: 6px 10px;
    border: 1px solid #cbd5d1;
    border-radius: 6px;
    background: #f7f9f8;
    color: #26332f;
    cursor: pointer;
    font-weight: 680;
    list-style: none;
    user-select: none;
  }
  .motius-benchmark-menu > summary::-webkit-details-marker { display: none; }
  .motius-benchmark-menu > summary::after {
    content: "▾";
    margin-left: 8px;
    color: #0a746b;
  }
  .motius-benchmark-menu[open] > summary::after { content: "▴"; }
  .motius-benchmark-list {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    display: grid;
    grid-template-columns: repeat(2, minmax(260px, 1fr));
    width: min(720px, calc(100vw - 36px));
    max-height: min(70vh, 620px);
    padding: 8px;
    overflow: auto;
    border: 1px solid #cbd5d1;
    border-radius: 7px;
    background: #ffffff;
    box-shadow: 0 16px 42px rgba(23, 33, 30, 0.16);
  }
  .motius-benchmark-link {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 12px;
    align-items: center;
    min-height: 42px;
    padding: 8px 10px;
    border-radius: 5px;
    color: #26332f;
    text-decoration: none;
  }
  .motius-benchmark-link:hover { background: #f0f5f3; }
  .motius-benchmark-link[aria-current="page"] {
    background: #e5f3ef;
    color: #075e57;
    font-weight: 720;
  }
  .motius-benchmark-status {
    color: #69756f;
    font: 10px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
    white-space: nowrap;
  }
  .motius-benchmark-status[data-status="complete"] { color: #087267; }
  .motius-benchmark-status[data-status="metrics_complete"] { color: #946018; }
  .motius-benchmark-status[data-status="protocol_ready"] { color: #315f9d; }
  .motius-benchmark-status[data-status="paused"] { color: #a1453a; }
  .motius-benchmark-status[data-status="settings"] { color: #315f9d; }
  .motius-setting-nav {
    display: flex;
    align-items: center;
    gap: 6px;
    width: min(1420px, calc(100% - 36px));
    margin: 10px auto 0;
    padding: 4px;
    overflow-x: auto;
    border: 1px solid #d8dfdc;
    border-radius: 7px;
    background: #eef2f0;
    font: 13px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    letter-spacing: 0;
  }
  .motius-setting-label {
    flex: 0 0 auto;
    padding: 0 10px;
    color: #69756f;
    font: 10px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
    text-transform: uppercase;
  }
  .motius-setting-link {
    display: flex;
    flex: 0 0 auto;
    align-items: baseline;
    gap: 8px;
    min-height: 36px;
    padding: 8px 12px;
    border: 1px solid transparent;
    border-radius: 5px;
    color: #53615c;
    text-decoration: none;
  }
  .motius-setting-link:hover {
    border-color: #cbd5d1;
    background: #ffffff;
    color: #17211e;
  }
  .motius-setting-link[aria-current="page"] {
    border-color: #adc9c2;
    background: #ffffff;
    color: #075e57;
    box-shadow: 0 3px 10px rgba(23, 33, 30, 0.06);
    font-weight: 720;
  }
  .motius-setting-detail {
    color: #7a8581;
    font: 10px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
  }
  @media (max-width: 720px) {
    .motius-benchmark-nav {
      width: calc(100% - 24px);
      margin-top: 10px;
      gap: 8px;
      padding-left: 10px;
    }
    .motius-benchmark-context { display: none; }
    .motius-benchmark-list {
      position: fixed;
      top: 64px;
      right: 12px;
      grid-template-columns: 1fr;
      width: calc(100vw - 24px);
      max-height: calc(100vh - 82px);
    }
    .motius-setting-nav { width: calc(100% - 24px); }
    .motius-setting-label { display: none; }
  }
</style>
""".strip()


def _load() -> tuple[list[dict], dict[str, dict]]:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_by_id = {item["id"]: item for item in catalog["benchmarks"]}
    benchmarks = []
    for benchmark in taxonomy["benchmarks"]:
        item = dict(benchmark)
        item.update(catalog_by_id[item["id"]])
        benchmarks.append(item)
    return benchmarks, catalog


def _public_target(target: str) -> str:
    if target.startswith(("https://", "http://")):
        return target
    return urljoin(GITHUB_DOCS_BASE, target)


def _navigation_target(target: str) -> tuple[str, str]:
    marker = "https://huggingface.co/spaces/"
    public = _public_target(target)
    if public.startswith(marker):
        repo_id = public[len(marker):].strip("/")
        static_host = repo_id.replace("/", "-").lower()
        return f"https://{static_host}.static.hf.space/", "_self"
    return public, "_blank"


def _leaderboard_id(benchmark: dict) -> str:
    return benchmark.get("leaderboard", {}).get("id", benchmark["id"])


def _navigation_entries(benchmarks: list[dict]) -> list[dict]:
    entries = []
    seen = set()
    for benchmark in benchmarks:
        leaderboard = benchmark.get("leaderboard")
        entry_id = _leaderboard_id(benchmark)
        if entry_id in seen:
            continue
        seen.add(entry_id)
        if leaderboard:
            settings = [
                item
                for item in benchmarks
                if _leaderboard_id(item) == entry_id
            ]
            entries.append(
                {
                    "id": entry_id,
                    "label": leaderboard["label"],
                    "target": leaderboard["target"],
                    "status": "settings",
                    "status_label": f"{len(settings)} settings",
                }
            )
        else:
            entries.append(
                {
                    "id": entry_id,
                    "label": benchmark["label"],
                    "target": benchmark["target"],
                    "status": benchmark["status"],
                    "status_label": STATUS_LABELS[benchmark["status"]],
                }
            )
    return entries


def _navigation(benchmarks: list[dict], current: dict, hub: str) -> str:
    links = []
    current_id = _leaderboard_id(current)
    for entry in _navigation_entries(benchmarks):
        current_attr = (
            ' aria-current="page"'
            if entry["id"] == current_id
            else ""
        )
        status = entry["status"]
        target, target_mode = _navigation_target(entry["target"])
        links.append(
            "      "
            f'<a class="motius-benchmark-link" '
            f'href="{html.escape(target)}" '
            f'target="{target_mode}" rel="noopener noreferrer"'
            f"{current_attr}>"
            f"<span>{html.escape(entry['label'])}</span>"
            f'<span class="motius-benchmark-status" data-status="{status}">'
            f"{html.escape(entry['status_label'])}</span></a>"
        )
    return "\n".join(
        [
            START,
            (
                f'<nav class="motius-benchmark-nav" '
                f'data-benchmark-id="{current["id"]}" '
                'aria-label="Motius benchmark navigation">'
            ),
            (
                f'  <a class="motius-benchmark-brand" href="{hub}" '
                'target="_blank" rel="noopener noreferrer">'
                "Motius Benchmark Hub</a>"
            ),
            (
                '  <span class="motius-benchmark-context">'
                f"{html.escape(current['label'])}</span>"
            ),
            '  <details class="motius-benchmark-menu">',
            "    <summary>All benchmarks</summary>",
            '    <div class="motius-benchmark-list">',
            *links,
            "    </div>",
            "  </details>",
            "</nav>",
            END,
        ]
    )


def _settings_navigation(benchmarks: list[dict], current: dict) -> str:
    leaderboard = current.get("leaderboard")
    if not leaderboard:
        return ""
    settings = [
        benchmark
        for benchmark in benchmarks
        if _leaderboard_id(benchmark) == leaderboard["id"]
    ]
    if len(settings) < 2:
        return ""
    links = []
    for benchmark in settings:
        setting = benchmark["setting"]
        target, target_mode = _navigation_target(benchmark["target"])
        current_attr = (
            ' aria-current="page"'
            if benchmark["id"] == current["id"]
            else ""
        )
        links.append(
            "  "
            f'<a class="motius-setting-link" '
            f'href="{html.escape(target)}" '
            f'target="{target_mode}" rel="noopener noreferrer"'
            f"{current_attr}>"
            f"<span>{html.escape(setting['label'])}</span>"
            f'<span class="motius-setting-detail">'
            f"{html.escape(setting['detail'])}</span></a>"
        )
    return "\n".join(
        [
            SETTINGS_START,
            (
                f'<nav class="motius-setting-nav" '
                f'data-leaderboard-id="{leaderboard["id"]}" '
                f'aria-label="{html.escape(leaderboard["label"])} settings">'
            ),
            '  <span class="motius-setting-label">Evaluation setting</span>',
            *links,
            "</nav>",
            SETTINGS_END,
        ]
    )


def _replace_block(text: str, start: str, end: str, block: str) -> str:
    if start in text and end in text:
        pattern = re.escape(start) + r".*?" + re.escape(end)
        return re.sub(pattern, block, text, count=1, flags=re.DOTALL)
    return text


def _sync_page(path: Path, benchmark: dict, benchmarks: list[dict], hub: str) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original

    leaderboard = benchmark.get("leaderboard")
    setting = benchmark.get("setting")
    title = (
        f"{leaderboard['label']} · {setting['label']} · Motius"
        if leaderboard and setting
        else f"{benchmark['label']} · Motius"
    )
    heading = (
        f"{leaderboard['label']} Leaderboard"
        if leaderboard
        else benchmark["label"]
    )
    text = re.sub(
        r"<title>.*?</title>",
        f"<title>{html.escape(title)}</title>",
        text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"(<h1(?:\s[^>]*)?>).*?(</h1>)",
        rf"\1{html.escape(heading)}\2",
        text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<body(?![^>]*\bdata-benchmark-id=)([^>]*)>",
        f'<body data-benchmark-id="{benchmark["id"]}"\\1>',
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    style_block = "\n".join([STYLE_START, STYLE, STYLE_END])
    if STYLE_START in text:
        text = _replace_block(text, STYLE_START, STYLE_END, style_block)
    else:
        text = re.sub(
            r"</head>",
            f"{style_block}\n</head>",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    nav = _navigation(benchmarks, benchmark, hub)
    if START in text:
        text = _replace_block(text, START, END, nav)
    else:
        text, replaced = re.subn(
            r"<nav\s+class=[\"']leaderboard-nav[\"'][^>]*>.*?</nav>",
            nav,
            text,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not replaced:
            text = re.sub(
                r"(<body[^>]*>)",
                rf"\1\n{nav}",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

    settings_nav = _settings_navigation(benchmarks, benchmark)
    if SETTINGS_START in text and SETTINGS_END in text:
        if settings_nav:
            text = _replace_block(
                text,
                SETTINGS_START,
                SETTINGS_END,
                settings_nav,
            )
        else:
            text = _replace_block(text, SETTINGS_START, SETTINGS_END, "")
    elif settings_nav:
        text = text.replace(END, f"{END}\n{settings_nav}", 1)

    if text != original:
        path.write_text(text, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when running the synchronizer would change a page.",
    )
    args = parser.parse_args()

    benchmarks, catalog = _load()
    changed = []
    for benchmark in benchmarks:
        source = benchmark["source"].split("#", 1)[0]
        source_path = ROOT / source
        index_path = source_path / "index.html" if source_path.is_dir() else None
        if index_path and index_path.is_file():
            before = index_path.read_text(encoding="utf-8")
            would_change = _sync_page(
                index_path,
                benchmark,
                benchmarks,
                catalog["navigation_target"],
            )
            if would_change:
                changed.append(index_path.relative_to(ROOT))
                if args.check:
                    index_path.write_text(
                        before,
                        encoding="utf-8",
                        newline="\n",
                    )

    if changed:
        action = "out of date" if args.check else "updated"
        print(f"{len(changed)} leaderboard pages {action}:")
        for path in changed:
            print(f"  {path}")
        if args.check:
            raise SystemExit(1)
    else:
        print("Leaderboard navigation is synchronized.")


if __name__ == "__main__":
    main()
