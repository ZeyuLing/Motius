#!/usr/bin/env python3
"""Propagate the shared gallery export runtime to published case viewers."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools/leaderboard_smpl_gallery.html"
FUNCTIONS = (
    "updateView",
    "fbxKey",
    "buildFbx",
    "preparedFbx",
    "scheduleFbxWarm",
    "exportFbx",
    "downloadView",
    "renderTiles",
)


def function_line(text: str, name: str) -> str:
    match = re.search(
        rf"^    (?:async )?function {re.escape(name)}\(.*$",
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"template function {name} is missing")
    return match.group(0)


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    preload = next(
        line
        for line in template.splitlines()
        if 'rel="modulepreload"' in line and "fbx-exporter-three" in line
    )
    runtime_const = next(
        line for line in template.splitlines() if line.startswith("    const assetCache=")
    )
    functions = {name: function_line(template, name) for name in FUNCTIONS}
    updated = 0
    for page in sorted((ROOT / "docs/leaderboards").glob("hf_space_*/cases/**/index.html")):
        text = page.read_text(encoding="utf-8")
        if "fbx-exporter-three" not in text:
            continue
        if preload not in text:
            marker = '  <link rel="preload" href="manifest.json" as="fetch" crossorigin="anonymous">'
            if marker not in text:
                raise RuntimeError(f"{page}: manifest preload is missing")
            text = text.replace(marker, f"{marker}\n{preload}", 1)
        text, count = re.subn(
            r"^    const assetCache=.*$",
            runtime_const,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"{page}: runtime cache declaration is missing")
        for name, replacement in functions.items():
            pattern = rf"^    (?:async )?function {re.escape(name)}\(.*$"
            text, count = re.subn(
                pattern,
                replacement,
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if count == 0 and name in {
                "fbxKey",
                "buildFbx",
                "preparedFbx",
                "scheduleFbxWarm",
            }:
                anchor = re.search(
                    r"^    async function exportFbx\(.*$",
                    text,
                    re.MULTILINE,
                )
                if not anchor:
                    raise RuntimeError(f"{page}: exportFbx anchor is missing")
                text = (
                    text[: anchor.start()]
                    + replacement
                    + "\n"
                    + text[anchor.start() :]
                )
            elif count != 1:
                raise RuntimeError(f"{page}: function {name} is missing")
        page.write_text(text, encoding="utf-8")
        updated += 1
    print(f"synchronized {updated} SMPL gallery viewers")


if __name__ == "__main__":
    main()
