import pytest
from PIL import Image

from tools.capture_leaderboard_method_gif import (
    _crop_frame,
    _duration_ms,
    _sample_durations_ms,
    _source_frame_indices,
    _validate_output,
)


def test_gif_duration_preserves_requested_30_fps() -> None:
    durations = _duration_ms(frames=90, fps=30)

    assert set(durations) == {30, 40}
    assert sum(durations) == 3000


def test_sampled_gif_preserves_source_timeline() -> None:
    source_frames = list(range(0, 300, 4))
    durations = _sample_durations_ms(
        source_frames=source_frames,
        source_fps=30,
        maximum=299,
    )

    assert set(durations) == {130, 140}
    assert sum(durations) == 10000


def test_sampled_gif_respects_partial_capture_duration() -> None:
    durations = _sample_durations_ms(
        source_frames=[19, 24, 29],
        source_fps=30,
        maximum=299,
    )

    assert sum(durations) == 500


def test_capture_is_cropped_to_the_declared_size() -> None:
    frame = Image.new("RGB", (512, 513))

    assert _crop_frame(frame, 512, 512).size == (512, 512)
    with pytest.raises(ValueError):
        _crop_frame(frame, 513, 513)


def test_source_frame_indices_stop_at_the_real_last_frame():
    assert _source_frame_indices(
        maximum=94,
        start_frame=0,
        frame_step=4,
        limit=75,
    ) == list(range(0, 95, 4))


def test_source_frame_indices_respect_start_and_limit():
    assert _source_frame_indices(
        maximum=299,
        start_frame=19,
        frame_step=5,
        limit=3,
    ) == [19, 24, 29]


@pytest.mark.parametrize(
    ("frame_step", "limit", "message"),
    [(0, 4, "frame_step"), (1, 0, "frames")],
)
def test_source_frame_indices_reject_invalid_sampling(
    frame_step,
    limit,
    message,
):
    with pytest.raises(ValueError, match=message):
        _source_frame_indices(
            maximum=10,
            start_frame=0,
            frame_step=frame_step,
            limit=limit,
        )


def test_sample_durations_reject_invalid_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        _sample_durations_ms([0, 1], source_fps=0, maximum=1)


def test_mp4_capture_requires_every_source_frame(tmp_path) -> None:
    _validate_output(tmp_path / "preview.mp4", frame_step=1, include_audio=True)

    with pytest.raises(ValueError, match="frame-step 1"):
        _validate_output(
            tmp_path / "preview.mp4",
            frame_step=4,
            include_audio=False,
        )


def test_audio_capture_requires_mp4(tmp_path) -> None:
    with pytest.raises(ValueError, match="MP4"):
        _validate_output(
            tmp_path / "preview.gif",
            frame_step=1,
            include_audio=True,
        )
