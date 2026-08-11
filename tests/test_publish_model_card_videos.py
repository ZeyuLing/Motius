from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import publish_model_card_videos as publisher


def test_verify_public_video_eventually_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = []
    sleeps = []

    def verify(url: str) -> None:
        attempts.append(url)
        if len(attempts) < 3:
            raise RuntimeError("attachment is still propagating")

    monkeypatch.setattr(publisher, "_verify_public_video", verify)
    monkeypatch.setattr(publisher.time, "sleep", sleeps.append)

    publisher._verify_public_video_eventually("https://example.test/demo.mp4")

    assert attempts == ["https://example.test/demo.mp4"] * 3
    assert sleeps == [1, 2]


def test_verify_public_video_eventually_rejects_invalid_attempt_count() -> None:
    with pytest.raises(ValueError, match="attempts must be positive"):
        publisher._verify_public_video_eventually(
            "https://example.test/demo.mp4",
            attempts=0,
        )


def test_replace_published_urls_updates_every_model_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    untouched = tmp_path / "untouched.md"
    first.write_text("old-a old-b\n", encoding="utf-8")
    second.write_text("old-a\n", encoding="utf-8")
    untouched.write_text("current\n", encoding="utf-8")
    monkeypatch.setattr(
        publisher,
        "_catalog_cards",
        lambda: {
            "first": first,
            "second": second,
            "untouched": untouched,
        },
    )

    changed = publisher._replace_published_urls(
        {"old-a": "new-a", "old-b": "new-b"}
    )

    assert changed == 2
    assert first.read_text(encoding="utf-8") == "new-a new-b\n"
    assert second.read_text(encoding="utf-8") == "new-a\n"
    assert untouched.read_text(encoding="utf-8") == "current\n"


def test_finish_pending_url_replacements_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "video_attachments.json"
    manifest = {
        "videos": {
            "assets/demo.gif": {
                "previous_url": "old-url",
                "url": "new-url",
            }
        }
    }
    seen = []
    monkeypatch.setattr(publisher, "VIDEO_ATTACHMENTS", manifest_path)
    monkeypatch.setattr(
        publisher,
        "_replace_published_urls",
        lambda replacements: seen.append(replacements) or 2,
    )

    changed = publisher._finish_pending_url_replacements(manifest)

    assert changed == 2
    assert seen == [{"old-url": "new-url"}]
    assert "previous_url" not in manifest["videos"]["assets/demo.gif"]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "previous_url" not in persisted["videos"]["assets/demo.gif"]
