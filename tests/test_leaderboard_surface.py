import importlib.util
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "sync_leaderboard_surface.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("sync_leaderboard_surface", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_leaderboard_pages_share_one_visual_surface() -> None:
    surface = _load_tool()
    roots = surface.space_roots()
    pages = [page for root in roots for page in root.rglob("*.html")]
    canonical = surface.CANONICAL_STYLE.read_text(encoding="utf-8")

    assert len(roots) == 16
    assert len(pages) == 37
    assert surface.synchronize(check=True) == []

    for root in roots:
        assert (root / surface.SPACE_STYLE_NAME).read_text(encoding="utf-8") == canonical

    for page in pages:
        root = next(root for root in roots if page.is_relative_to(root))
        text = page.read_text(encoding="utf-8")
        relative_style = Path(
            os.path.relpath(root / surface.SPACE_STYLE_NAME, page.parent)
        ).as_posix()
        expected_type = surface.page_type(page, root)

        assert text.count(surface.STYLE_START) == 1
        assert text.count(surface.STYLE_END) == 1
        assert f'href="{relative_style}" data-motius-surface' in text
        assert f'data-motius-page="{expected_type}"' in text
        assert text.count(surface.SKIP_START) == 1
        assert text.count(surface.SKIP_END) == 1
        assert text.count(surface.THEME_START) == 1
        assert text.count(surface.THEME_END) == 1
        assert text.count('name="theme-color"') == 1
        assert f'content="{surface.THEME_COLOR}"' in text

        skip_target = re.search(r'class="motius-skip-link" href="#([^"]+)"', text)
        assert skip_target is not None
        target = re.escape(skip_target.group(1))
        assert re.search(
            rf'<main\b[^>]*\bid="{target}"[^>]*\btabindex="-1"',
            text,
            flags=re.IGNORECASE,
        ) or re.search(
            rf'<(?:div|section|article|header)\b'
            rf'(?=[^>]*\brole="main")(?=[^>]*\bid="{target}")'
            rf'(?=[^>]*\btabindex="-1")[^>]*>',
            text,
            flags=re.IGNORECASE,
        )

        canvases = re.findall(r"<canvas\b[^>]*>", text, flags=re.IGNORECASE)
        for canvas in canvases:
            assert re.search(r'\brole="img"', canvas, flags=re.IGNORECASE)
            assert re.search(
                r'\baria-(?:label|labelledby)="[^"]+"',
                canvas,
                flags=re.IGNORECASE,
            )

        tables = re.findall(r"<table\b[^>]*>", text, flags=re.IGNORECASE)
        for table in tables:
            assert re.search(
                r'\baria-(?:label|labelledby)="[^"]+"',
                table,
                flags=re.IGNORECASE,
            )


def test_surface_transform_is_idempotent_and_preserves_authored_names() -> None:
    surface = _load_tool()
    space = surface.space_roots()[0]
    path = space / "fixture.html"
    source = """<!doctype html>
<html><head><title>Fixture</title></head><body>
<div class="shell"><h1>Fixture</h1>
<canvas id="custom" aria-label="Authored motion diagnostic"></canvas>
<table aria-labelledby="results-title"><tr><td>1</td></tr></table>
<h2 id="results-title">Results</h2></div>
</body></html>
"""

    transformed = surface.transform_page(path, space, source)

    assert surface.transform_page(path, space, transformed) == transformed
    assert 'role="main"' in transformed
    assert 'aria-label="Authored motion diagnostic"' in transformed
    assert 'aria-labelledby="results-title"' in transformed
    assert transformed.count('name="theme-color"') == 1


def test_surface_contains_accessible_and_adaptive_states() -> None:
    css = (ROOT / "tools" / "motius_surface.css").read_text(encoding="utf-8")

    assert ".motius-skip-link" in css
    assert ":focus-visible" in css
    assert "min-height: 44px" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "forced-colors: active" in css
    assert "@media print" in css
    assert 'body[data-motius-page="dashboard"]' in css
    assert 'body[data-motius-page="viewer"]' in css
    assert 'body[data-motius-page="audit"]' in css


def test_runtime_motion_previews_have_accessible_names() -> None:
    for space in ("hf_space_instruction_editing", "hf_space_motion_edit"):
        script = (
            ROOT / "docs" / "leaderboards" / space / "smpl_viewer.js"
        ).read_text(encoding="utf-8")

        assert 'renderer.domElement.setAttribute("role", "img")' in script
        assert '`${data.label} motion preview`' in script
