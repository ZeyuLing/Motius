from tools.audit_model_card_media import (
    _audio_duration_seconds,
    _audio_required,
)


def test_audio_is_required_for_music_conditioned_previews():
    assert _audio_required(
        "assets/model_zoo/bailando/"
        "bailando_aistpp_break_smpl_mesh_512_30fps.gif"
    )
    assert _audio_required(
        "assets/model_zoo/unimumo/"
        "unimumo_dance_to_music_input_smpl_512_30fps.gif"
    )


def test_audio_is_not_required_for_text_to_motion():
    assert not _audio_required(
        "assets/model_zoo/tm2d/"
        "tm2d_humanml3d_004822_smpl_mesh_512_30fps.gif"
    )


def test_missing_audio_has_zero_duration(tmp_path):
    path = tmp_path / "missing.mp4"
    path.write_bytes(b"not a video")

    assert _audio_duration_seconds(path) == 0.0
