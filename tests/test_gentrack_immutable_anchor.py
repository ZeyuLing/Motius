from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from motius.models.gentrack.g1_bundle import PhysFlowG1Bundle
from motius.trainers.gentrack.physflow_g1_grpo_trainer import PhysFlowG1GRPOTrainer


class _TinyCore(nn.Module):
    output_dim = 4

    def __init__(self, fill: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((3, 4), float(fill)))

    def forward(self, x, **kwargs):
        return x + self.weight.mean()


def _save_motion_transformer_checkpoint(path: str, core: nn.Module) -> str:
    os.makedirs(path, exist_ok=True)
    torch.save(
        {"motion_transformer": core.state_dict()},
        os.path.join(path, "model.pt"),
    )
    return path


def _make_bundle(
    tmp_path,
    *,
    g0_fill: float = 1.0,
    live_fill: float = 99.0,
    require: bool = True,
) -> tuple[PhysFlowG1Bundle, _TinyCore, _TinyCore]:
    g0_core = _TinyCore(fill=g0_fill)
    live_core = _TinyCore(fill=live_fill)
    ckpt_dir = _save_motion_transformer_checkpoint(
        str(tmp_path / "g0_checkpoint"),
        g0_core,
    )

    bundle = object.__new__(PhysFlowG1Bundle)
    object.__setattr__(bundle, "motion_transformer", live_core)
    bundle.immutable_anchor_checkpoint = ckpt_dir
    bundle.require_immutable_anchor = require
    bundle._immutable_g0_transformer = None
    bundle._anchor_transformer = None
    return bundle, g0_core, live_core


def test_immutable_anchor_matches_g0_after_current_model_drift(tmp_path) -> None:
    bundle, g0_core, live_core = _make_bundle(tmp_path)
    assert not torch.allclose(live_core.weight, g0_core.weight)

    bundle.init_immutable_g0_anchor()
    anchor = bundle._reference_transformer()
    assert anchor is not None
    assert torch.allclose(anchor.weight, g0_core.weight)
    assert not torch.allclose(anchor.weight, live_core.weight)


def test_immutable_anchor_unchanged_after_resume_init(tmp_path) -> None:
    bundle, g0_core, live_core = _make_bundle(tmp_path)
    bundle.init_immutable_g0_anchor()
    fp1 = bundle.immutable_g0_anchor_fingerprint()

    live_core.load_state_dict(_TinyCore(fill=123.0).state_dict())
    bundle.init_immutable_g0_anchor()
    fp2 = bundle.immutable_g0_anchor_fingerprint()
    anchor = bundle._reference_transformer()

    assert fp1 == fp2
    assert anchor is not None
    assert torch.allclose(anchor.weight, g0_core.weight)
    assert bundle.immutable_g0_anchor_fingerprint() == fp1


def test_immutable_anchor_follows_live_policy_device(tmp_path) -> None:
    bundle, _, _ = _make_bundle(tmp_path)
    bundle.init_immutable_g0_anchor()
    object.__setattr__(bundle, "motion_transformer", _TinyCore().to("meta"))

    anchor = bundle._reference_transformer()

    assert anchor is not None
    assert next(anchor.parameters()).device.type == "meta"


def test_grpo_sampler_replays_complete_stochastic_path() -> None:
    core = _TinyCore(fill=0.25)
    bundle = object.__new__(PhysFlowG1Bundle)
    object.__setattr__(bundle, "motion_transformer", core)
    bundle._ctxt_input_dim = 4
    bundle.sample_steps = 3
    bundle.sample_guidance = 1.0
    text_vec = torch.zeros(2, 1, 4)
    text_ctxt = [torch.zeros(2, 4), torch.zeros(3, 4)]
    ctxt_len = torch.tensor([2, 3], dtype=torch.long)
    lengths = torch.tensor([5, 4], dtype=torch.long)

    first = bundle.sample_motion_grpo(
        text_vec,
        text_ctxt,
        ctxt_len,
        lengths,
        num_steps=3,
        eta=0.7,
        transformer=core,
    )
    second = bundle.sample_motion_grpo(
        text_vec,
        text_ctxt,
        ctxt_len,
        lengths,
        num_steps=3,
        eta=0.7,
        initial_noise=first["initial_noise"],
        transition_noises=first["transition_noises"],
        transformer=core,
        return_policy_artifacts=False,
    )

    assert torch.equal(first["sample"], second["sample"])


def test_missing_immutable_checkpoint_fail_closed(tmp_path) -> None:
    bundle = object.__new__(PhysFlowG1Bundle)
    object.__setattr__(bundle, "motion_transformer", _TinyCore())
    bundle.immutable_anchor_checkpoint = str(tmp_path / "missing")
    bundle.require_immutable_anchor = True
    bundle._immutable_g0_transformer = None
    bundle._anchor_transformer = None

    with pytest.raises(RuntimeError, match="not found"):
        bundle.init_immutable_g0_anchor()


def test_gt_anchor_step_forwards_g0_velocity_distill_weight() -> None:
    class _DistillBundle:
        def __init__(self) -> None:
            self.anchor_weight = None

        @staticmethod
        def _device() -> torch.device:
            return torch.device("cpu")

        @staticmethod
        def normalize_motion(motion: torch.Tensor) -> torch.Tensor:
            return motion

        def sft_loss_g1(self, *args, anchor_weight: float, **kwargs):
            self.anchor_weight = anchor_weight
            return {
                "loss": torch.tensor(2.0),
                "sft_mse": torch.tensor(1.5),
                "anchor_mse": torch.tensor(0.5),
            }

    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    trainer.bundle = _DistillBundle()
    trainer.gt_weight = 1.0
    trainer.gt_g0_distill_weight = 0.25
    result = trainer._gt_anchor_step(
        {
            "motion": torch.zeros(2, 3, 4),
            "text_vec_raw": torch.zeros(2, 1, 4),
            "text_ctxt_raw": [torch.zeros(1, 4), torch.zeros(1, 4)],
            "text_ctxt_raw_length": torch.ones(2, dtype=torch.long),
            "tgt_length": torch.full((2,), 3, dtype=torch.long),
        }
    )

    assert trainer.bundle.anchor_weight == pytest.approx(0.25)
    assert result["loss_gt_g0_distill"].item() == pytest.approx(0.5)


def test_require_effective_grpo_replay_rejects_replay_one() -> None:
    bundle = SimpleNamespace(
        immutable_anchor_checkpoint=None,
        require_immutable_anchor=False,
        init_immutable_g0_anchor=lambda checkpoint_path=None, force=False: None,
    )
    with pytest.raises(ValueError, match="require_effective_grpo_replay"):
        PhysFlowG1GRPOTrainer(
            bundle=bundle,
            grpo_replay_epochs=1,
            require_effective_grpo_replay=True,
            num_samples=2,
        )


def test_min_grpo_replay_epochs_rejects_replay_one() -> None:
    bundle = SimpleNamespace(
        immutable_anchor_checkpoint=None,
        require_immutable_anchor=False,
        init_immutable_g0_anchor=lambda checkpoint_path=None, force=False: None,
    )
    with pytest.raises(ValueError, match="min_grpo_replay_epochs"):
        PhysFlowG1GRPOTrainer(
            bundle=bundle,
            grpo_replay_epochs=1,
            min_grpo_replay_epochs=2,
            num_samples=2,
        )


def test_update_based_gt_anchor_interrupts_replay_at_requested_ratio() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    trainer.gt_anchor_update_interval = 1
    trainer.gt_anchor_interval = 0
    trainer.gt_weight = 1.0
    trainer._grpo_updates_since_anchor = 1
    trainer._grpo_rollouts_since_anchor = 0
    trainer._grpo_replay_cache = {"next_replay_epoch": 2}
    calls = []

    trainer._gt_anchor_step = lambda batch: calls.append("anchor") or {
        "loss": torch.tensor(0.0),
        "update_is_gt_anchor": torch.tensor(1.0),
    }
    trainer._replay_grpo_step = lambda: calls.append("replay") or {
        "loss": torch.tensor(0.0),
        "update_is_gt_anchor": torch.tensor(0.0),
    }

    first = trainer.train_step({})
    assert first["update_is_gt_anchor"].item() == pytest.approx(1.0)
    assert trainer._grpo_replay_cache is not None

    second = trainer.train_step({})
    assert second["update_is_gt_anchor"].item() == pytest.approx(0.0)

    third = trainer.train_step({})
    assert third["update_is_gt_anchor"].item() == pytest.approx(1.0)
    assert calls == ["anchor", "replay", "anchor"]


def test_paired_live_and_g0_samples_share_one_simulator_request() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    calls = []

    def score(qpos, num_frames, group_size, work_dir):
        calls.append((qpos.copy(), list(num_frames), group_size, work_dir))
        return [{"sample_id": index} for index in range(qpos.shape[0])]

    trainer._score_samples = score
    live = np.zeros((8, 5, 36), dtype=np.float32)
    g0 = np.ones((8, 5, 36), dtype=np.float32)

    live_metrics, g0_metrics = trainer._score_candidate_and_reference(
        live,
        g0,
        [5, 4],
        group_size=4,
        work_dir="/tmp/paired-score-test",
    )

    assert len(calls) == 1
    paired_qpos, paired_frames, group_size, work_dir = calls[0]
    assert paired_qpos.shape == (16, 5, 36)
    np.testing.assert_array_equal(paired_qpos[:8], live)
    np.testing.assert_array_equal(paired_qpos[8:], g0)
    assert paired_frames == [5, 4, 5, 4]
    assert group_size == 4
    assert work_dir == "/tmp/paired-score-test"
    assert [item["sample_id"] for item in live_metrics] == list(range(8))
    assert [item["sample_id"] for item in g0_metrics] == list(range(8, 16))


def test_paired_scoring_common_envs_uses_two_seed_matched_requests() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    trainer.paired_reward_common_envs = True
    calls = []

    def score(qpos, num_frames, group_size, work_dir):
        calls.append((qpos.copy(), list(num_frames), group_size, work_dir))
        return [{"sample_id": index} for index in range(qpos.shape[0])]

    trainer._score_samples = score
    live = np.zeros((8, 5, 36), dtype=np.float32)
    g0 = np.ones((8, 5, 36), dtype=np.float32)

    live_metrics, g0_metrics = trainer._score_candidate_and_reference(
        live,
        g0,
        [5, 4],
        group_size=4,
        work_dir="/tmp/paired-common-env-test",
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], live)
    np.testing.assert_array_equal(calls[1][0], g0)
    assert calls[0][3].endswith("/candidate")
    assert calls[1][3].endswith("/reference")
    assert [item["sample_id"] for item in live_metrics] == list(range(8))
    assert [item["sample_id"] for item in g0_metrics] == list(range(8))


def test_reward_telemetry_aggregates_public_sonic_components() -> None:
    metrics = [
        {
            "completion": 1.0,
            "fall_detected": False,
            "reward_components": {
                "global_anchor_pos": 0.8,
                "relative_body_pos": 0.6,
                "body_lin_vel": 0.4,
            },
        },
        {
            "completion": 0.5,
            "fall_detected": True,
            "reward_components": {
                "global_anchor_pos": 0.4,
                "relative_body_pos": 0.2,
                "body_lin_vel": 0.8,
            },
        },
    ]

    telemetry = PhysFlowG1GRPOTrainer._reward_telemetry(metrics)

    assert telemetry["judge_completion_mean"] == pytest.approx(0.75)
    assert telemetry["judge_fall_rate"] == pytest.approx(0.5)
    assert telemetry["judge_global_anchor_pos_reward_mean"] == pytest.approx(0.6)
    assert telemetry["judge_relative_body_pos_reward_mean"] == pytest.approx(0.4)
    assert telemetry["judge_body_lin_vel_reward_mean"] == pytest.approx(0.6)
    assert "judge_body_ang_vel_reward_mean" not in telemetry


def test_paired_motion_telemetry_masks_padding_and_detects_identity() -> None:
    candidate = torch.zeros(2, 4, 3)
    reference = candidate.clone()
    lengths = torch.tensor([2, 3])

    telemetry = PhysFlowG1GRPOTrainer._paired_motion_telemetry(
        candidate,
        reference,
        lengths,
    )
    assert telemetry["candidate_vs_g0_motion_mse"] == 0
    assert telemetry["candidate_vs_g0_motion_max_abs"] == 0

    candidate[0, 3] = 100
    candidate[1, 1, 2] = 2
    telemetry = PhysFlowG1GRPOTrainer._paired_motion_telemetry(
        candidate,
        reference,
        lengths,
    )
    assert telemetry["candidate_vs_g0_motion_mse"] == pytest.approx(4 / 15)
    assert telemetry["candidate_vs_g0_motion_max_abs"] == 2


def test_paired_reference_composite_advantage_penalizes_bad_candidates() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    trainer.grpo_advantage_eps = 1e-4
    trainer.grpo_physical_advantage_weight = 1.0
    trainer.grpo_semantic_advantage_weight = 0.0
    trainer.grpo_semantic_floor_weight = 0.0
    trainer.grpo_semantic_floor_margin = 0.0

    advantages, telemetry = trainer._compose_advantages(
        candidate_physical_rewards=torch.tensor([-3.0, -2.8, -2.6, -2.4]),
        reference_physical_rewards=torch.tensor([-1.8, -1.6, -1.4, -1.2]),
        group_size=4,
    )

    assert torch.all(advantages < 0)
    assert telemetry["candidate_vs_g0_physical_delta_mean"] < 0
    assert telemetry["candidate_vs_g0_physical_improved_fraction"] == 0


def test_semantic_floor_penalizes_only_regressions_from_paired_g0() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    trainer.grpo_advantage_eps = 1e-4
    trainer.grpo_physical_advantage_weight = 0.0
    trainer.grpo_semantic_advantage_weight = 0.0
    trainer.grpo_semantic_floor_weight = 1.0
    trainer.grpo_semantic_floor_margin = 0.0
    candidates = torch.tensor([2.0, 1.0, 3.0, 0.5])
    references = torch.tensor([1.0, 1.0, 2.0, 1.0])

    advantages, telemetry = trainer._compose_advantages(
        candidate_physical_rewards=torch.zeros(4),
        reference_physical_rewards=None,
        candidate_semantic_distances=candidates,
        reference_semantic_distances=references,
        group_size=4,
    )

    assert advantages[0] < 0
    assert advantages[1] == pytest.approx(0.0)
    assert advantages[2] < 0
    assert advantages[3] == pytest.approx(0.0)
    assert telemetry["tmr_floor_violation_mean"] > 0
