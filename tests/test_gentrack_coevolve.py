from pathlib import Path

from tools.gentrack import coevolve, trainee_round


def test_coevolve_selects_repo_output_checkpoint_between_rounds(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "lagged"
    runtime_root = tmp_path / "outputs" / "training" / "protomotions"

    selected = coevolve.select_tracker_input_checkpoint(
        arm=arm,
        round_index=1,
        trainee_init_checkpoint=tmp_path / "t0.ckpt",
        restart_each_round=False,
        runtime_checkpoint_root=runtime_root,
    )

    assert selected == runtime_root / "lagged_co_r0" / "last.ckpt"


def test_coevolve_round_zero_keeps_explicit_initial_tracker(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "t0.ckpt"

    selected = coevolve.select_tracker_input_checkpoint(
        arm=tmp_path / "lagged",
        round_index=0,
        trainee_init_checkpoint=initial,
        restart_each_round=False,
        runtime_checkpoint_root=tmp_path / "runtime",
    )

    assert selected == initial


def test_trainee_runtime_writes_below_declared_output_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "training" / "protomotions"

    env = trainee_round._tracker_env(output_root)

    assert env["MOTIUS_PROTOMOTIONS_OUTPUT_ROOT"] == str(output_root)
