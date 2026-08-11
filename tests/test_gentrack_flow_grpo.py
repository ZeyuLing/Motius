from __future__ import annotations

import numpy as np
import torch

from motius.models.gentrack.g1_bundle import PhysFlowG1Bundle
from motius.models.gentrack.flow_grpo import (
    clipped_grpo_loss,
    flow_dpo_pair_loss,
    flow_grpo_transition,
    group_relative_advantages,
    paired_reference_advantages,
    reference_augmented_advantages,
    reverse_kl_from_log_probs,
    select_reliable_preference_pairs,
    sample_unique_timestep_indices,
)
from motius.trainers.gentrack.physflow_g1_grpo_trainer import (
    PhysFlowG1GRPOTrainer,
    PhysFlowG1RewardWeightedSFTTrainer,
)
from motius.trainers.gentrack.physflow_g1_dpo_trainer import (
    PhysFlowG1DPOTrainer,
)
from motius.trainers.gentrack.physflow_trainer import PhysFlowTrainer


def test_g1_bundle_can_restrict_posttraining_to_late_blocks() -> None:
    class TinyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.single_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(12)]
            )
            self.final_layer = torch.nn.Linear(2, 2)

    bundle = object.__new__(PhysFlowG1Bundle)
    torch.nn.Module.__init__(bundle)
    bundle.motion_transformer = TinyTransformer()
    bundle.null_vtxt_feat = torch.nn.Parameter(torch.zeros(1))
    bundle.null_ctxt_input = torch.nn.Parameter(torch.zeros(1))
    bundle.special_game_vtxt_feat = torch.nn.Parameter(torch.zeros(1))
    bundle.special_game_ctxt_feat = torch.nn.Parameter(torch.zeros(1))
    bundle.trainable_motion_parameter_prefixes = (
        "single_blocks.10.",
        "single_blocks.11.",
        "final_layer.",
    )
    bundle.freeze_condition_embeddings_when_restricted = True
    bundle._restricted_trainable_motion_parameters = ()

    bundle._restrict_trainable_motion_parameters()
    trainable = {
        name
        for name, parameter in bundle.motion_transformer.named_parameters()
        if parameter.requires_grad
    }

    assert trainable
    assert all(
        name.startswith(("single_blocks.10.", "single_blocks.11.", "final_layer."))
        for name in trainable
    )
    assert not bundle.motion_transformer.single_blocks[9].weight.requires_grad
    assert not bundle.null_vtxt_feat.requires_grad
    assert bundle.trainable_motion_parameter_scope()["parameter_numel"] > 0


def test_group_relative_advantages_are_prompt_local() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0])
    advantages = group_relative_advantages(rewards, group_size=4).reshape(2, 4)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(advantages[1], torch.zeros(4), atol=1e-6)


def test_execution_reward_modes_use_dense_or_fall_only_signal() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    metric = {"physical_score": 1.25, "fall_detected": False}

    trainer.execution_reward_mode = "dense"
    assert trainer._execution_reward(metric) == -1.25
    trainer.execution_reward_mode = "success"
    assert trainer._execution_reward(metric) == 1.0
    assert trainer._execution_reward({**metric, "fall_detected": True}) == 0.0
    assert trainer._execution_reward({**metric, "error": "failed"}) == 0.0
    trainer.execution_reward_mode = "none"
    assert trainer._execution_reward(metric) == 0.0


def test_reward_weighted_sft_normalizes_positive_advantages_per_prompt() -> None:
    class TinyBundle:
        def sft_loss_g1(self, *args, **kwargs):
            del args
            self.good_mask = kwargs["good_mask"]
            self.sample_weights = kwargs["sample_weights"]
            loss = torch.tensor(2.0, requires_grad=True)
            return {"loss": loss, "sft_mse": loss.detach()}

    trainer = object.__new__(PhysFlowG1RewardWeightedSFTTrainer)
    trainer.bundle = TinyBundle()
    trainer.num_samples = 4
    trainer.anchor_weight = 0.0
    cache = {
        "advantages": torch.tensor([2.0, 1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0]),
        "vtxt": torch.zeros(8, 1),
        "ctxt": [torch.zeros(1, 1) for _ in range(8)],
        "ctxt_len": torch.ones(8, dtype=torch.long),
        "target_latents": torch.zeros(8, 2, 3),
        "lengths": torch.full((8,), 2, dtype=torch.long),
    }

    output = trainer._grpo_loss_from_cache(cache)

    assert output["loss"].requires_grad
    assert trainer.bundle.good_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert torch.allclose(
        trainer.bundle.sample_weights[:4],
        torch.tensor([4.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]),
    )


def test_group_relative_telemetry_reports_per_prompt_reward_spread() -> None:
    trainer = object.__new__(PhysFlowG1GRPOTrainer)
    trainer.grpo_advantage_eps = 1e-4
    trainer.grpo_physical_advantage_weight = 1.0
    trainer.grpo_semantic_advantage_weight = 0.0
    trainer.grpo_semantic_floor_weight = 0.0

    _, telemetry = trainer._compose_advantages(
        torch.tensor([0.0, 2.0, 4.0, 6.0, 10.0, 10.0, 10.0, 10.0]),
        group_size=4,
    )

    assert telemetry["physical_reward_group_std_min"] == 0.0
    assert telemetry["physical_reward_group_std_max"] > 2.0
    assert telemetry["physical_reward_group_std_below_001_fraction"] == 0.5
    assert telemetry["physical_reward_group_mean"] == 6.5
    assert telemetry["physical_reward_group_best_mean"] == 8.0
    assert telemetry["physical_reward_group_best_minus_mean"] == 1.5


def test_reference_augmented_advantages_can_reject_an_entire_bad_group() -> None:
    candidates = torch.tensor([1.0, 1.5, 2.0, 2.5])
    references = torch.tensor([3.0, 3.5, 4.0, 4.5])
    advantages = reference_augmented_advantages(
        candidates,
        references,
        group_size=4,
    )
    assert torch.all(advantages < 0)


def test_paired_reference_advantages_preserve_counterfactual_sign() -> None:
    advantages = paired_reference_advantages(
        torch.tensor([9.0, 2.0, 4.0, 5.5]),
        torch.tensor([10.0, 1.0, 4.5, 5.0]),
        group_size=4,
    )

    assert advantages[0] < 0
    assert advantages[1] > 0
    assert advantages[2] < 0
    assert advantages[3] > 0


def test_paired_reference_advantages_keep_all_regressions_negative() -> None:
    advantages = paired_reference_advantages(
        torch.tensor([1.0, 8.0, 2.0, 7.0]),
        torch.tensor([2.0, 9.0, 3.0, 8.0]),
        group_size=4,
    )

    assert torch.all(advantages < 0)


def test_positive_advantage_increases_transition_probability() -> None:
    current = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])
    loss, _, _ = clipped_grpo_loss(current, old, advantages, clip_range=0.2)
    loss.backward()
    assert current.grad[0] < 0
    assert current.grad[1] > 0


def test_clipped_ratio_uses_fixed_behavior_policy_after_update() -> None:
    current = torch.log(torch.tensor([1.35, 0.70]))
    old = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])
    _, ratio_mean, clip_fraction = clipped_grpo_loss(
        current,
        old,
        advantages,
        clip_range=0.2,
    )
    assert not torch.allclose(ratio_mean, torch.tensor(1.0))
    assert torch.allclose(clip_fraction, torch.tensor(1.0))


def test_ratio_deviation_telemetry_detects_policy_change() -> None:
    current = torch.log(torch.tensor([1.001, 0.999]))
    old = torch.zeros(2)
    log_ratio = current - old
    ratio_abs_deviation = (log_ratio.exp() - 1.0).abs().mean()
    assert ratio_abs_deviation > 0


def test_flow_transition_replay_has_identical_log_prob() -> None:
    torch.manual_seed(7)
    shape = (3, 6, 4)
    model_output = torch.randn(shape)
    latents = torch.randn(shape)
    mask = torch.ones(3, 6, dtype=torch.bool)
    next_latents, sampled_log_prob = flow_grpo_transition(
        model_output,
        latents,
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.8),
        sigma_first_next=torch.tensor(0.8),
        eta=0.7,
        frame_mask=mask,
    )
    _, replay_log_prob = flow_grpo_transition(
        model_output,
        latents,
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.8),
        sigma_first_next=torch.tensor(0.8),
        eta=0.7,
        next_latents=next_latents,
        frame_mask=mask,
    )
    assert torch.isfinite(sampled_log_prob).all()
    assert torch.allclose(sampled_log_prob, replay_log_prob, atol=1e-6)


def test_flow_transition_reuses_explicit_noise() -> None:
    torch.manual_seed(11)
    shape = (2, 5, 3)
    model_output = torch.randn(shape)
    latents = torch.randn(shape)
    transition_noise = torch.randn(shape)
    kwargs = dict(
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.8),
        sigma_first_next=torch.tensor(0.8),
        eta=0.7,
        transition_noise=transition_noise,
    )
    first, first_log_prob = flow_grpo_transition(
        model_output,
        latents,
        **kwargs,
    )
    torch.manual_seed(999)
    second, second_log_prob = flow_grpo_transition(
        model_output,
        latents,
        **kwargs,
    )
    assert torch.equal(first, second)
    assert torch.equal(first_log_prob, second_log_prob)


def test_unique_timestep_sampling_covers_multiple_flow_times_per_trajectory() -> None:
    torch.manual_seed(17)
    indices = sample_unique_timestep_indices(
        batch_size=8,
        num_transitions=50,
        timesteps_per_update=4,
        device="cpu",
    )

    assert indices.shape == (8, 4)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 50
    assert all(len(torch.unique(row)) == 4 for row in indices)


def test_g1_grpo_loss_batches_multiple_flow_times_per_trajectory() -> None:
    class TinyBundle:
        grpo_loss_g1 = PhysFlowG1Bundle.grpo_loss_g1

        def __init__(self) -> None:
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.require_immutable_anchor = False
            self.seen_batches = []

        @staticmethod
        def _device() -> torch.device:
            return torch.device("cpu")

        @staticmethod
        def _pack_ctxt(text_ctxt, ctxt_len, device, dtype):
            del ctxt_len
            ctxt = torch.stack(text_ctxt).to(device=device, dtype=dtype)
            return ctxt, torch.ones(ctxt.shape[:2], device=device, dtype=torch.bool)

        def predict_flow(self, *, x_input, **kwargs):
            del kwargs
            self.seen_batches.append(int(x_input.shape[0]))
            return torch.zeros_like(x_input) + self.bias

        @staticmethod
        def _reference_transformer():
            return None

    bundle = TinyBundle()
    batch, transitions, frames, dims = 2, 4, 3, 2
    states = torch.randn(batch, transitions + 1, frames, dims)
    old_log_probs = torch.zeros(batch, transitions)
    sigmas = torch.linspace(1.0, 0.0, transitions + 1)
    indices = torch.arange(transitions).repeat(batch, 1)

    result = bundle.grpo_loss_g1(
        text_vec=torch.zeros(batch, 1),
        text_ctxt=[torch.zeros(1, 1) for _ in range(batch)],
        ctxt_len=torch.ones(batch, dtype=torch.long),
        lengths=torch.full((batch,), frames, dtype=torch.long),
        states=states,
        old_log_probs=old_log_probs,
        sigmas=sigmas,
        advantages=torch.tensor([1.0, -1.0]),
        eta=0.7,
        clip_range=0.2,
        reference_kl_weight=0.0,
        timestep_indices=indices,
    )
    result["loss"].backward()

    assert bundle.seen_batches == [batch * transitions]
    assert result["timesteps_per_update"] == transitions
    assert torch.isfinite(result["advantage_log_ratio_cov"])
    assert torch.isfinite(result["advantage_log_ratio_alignment"])
    assert torch.isfinite(result["positive_advantage_log_ratio_mean"])
    assert torch.isfinite(result["negative_advantage_log_ratio_mean"])
    assert torch.isfinite(bundle.bias.grad)


def test_g1_grpo_transition_microbatch_preserves_gradient() -> None:
    class TinyBundle:
        grpo_loss_g1 = PhysFlowG1Bundle.grpo_loss_g1

        def __init__(self) -> None:
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.require_immutable_anchor = False
            self.seen_batches = []

        @staticmethod
        def _device() -> torch.device:
            return torch.device("cpu")

        @staticmethod
        def _pack_ctxt(text_ctxt, ctxt_len, device, dtype):
            del ctxt_len
            ctxt = torch.stack(text_ctxt).to(device=device, dtype=dtype)
            return ctxt, torch.ones(ctxt.shape[:2], device=device, dtype=torch.bool)

        def predict_flow(self, *, x_input, **kwargs):
            del kwargs
            self.seen_batches.append(int(x_input.shape[0]))
            return torch.zeros_like(x_input) + self.bias

        @staticmethod
        def _reference_transformer():
            return None

    bundle = TinyBundle()
    batch, transitions, frames, dims = 4, 4, 3, 2
    result = bundle.grpo_loss_g1(
        text_vec=torch.zeros(batch, 1),
        text_ctxt=[torch.zeros(1, 1) for _ in range(batch)],
        ctxt_len=torch.ones(batch, dtype=torch.long),
        lengths=torch.full((batch,), frames, dtype=torch.long),
        states=torch.randn(batch, transitions + 1, frames, dims),
        old_log_probs=torch.zeros(batch, transitions),
        sigmas=torch.linspace(1.0, 0.0, transitions + 1),
        advantages=torch.tensor([1.0, 0.5, -0.5, -1.0]),
        eta=0.7,
        clip_range=0.2,
        reference_kl_weight=0.0,
        timestep_indices=torch.arange(transitions).repeat(batch, 1),
        transition_microbatch_size=2,
    )
    result["loss"].backward()

    assert max(bundle.seen_batches) == 2
    assert result["transition_microbatch_size"] == 2
    assert torch.isfinite(bundle.bias.grad)


def test_reference_kl_is_nonnegative_and_zero_at_match() -> None:
    current = torch.tensor([-1.0, -2.0])
    assert torch.allclose(reverse_kl_from_log_probs(current, current), torch.tensor(0.0))
    assert reverse_kl_from_log_probs(current, current + 0.5) > 0


def test_flow_dpo_pair_loss_has_preference_gradient_direction() -> None:
    winner = torch.tensor([1.0, 1.2], requires_grad=True)
    loser = torch.tensor([1.0, 1.2], requires_grad=True)
    reference_winner = torch.tensor([1.0, 1.2])
    reference_loser = torch.tensor([1.0, 1.2])

    output = flow_dpo_pair_loss(
        winner,
        loser,
        reference_winner,
        reference_loser,
        beta=100.0,
    )
    output["loss"].backward()

    assert torch.allclose(output["loss"], torch.log(torch.tensor(2.0)))
    assert torch.all(winner.grad > 0)
    assert torch.all(loser.grad < 0)
    assert output["implicit_accuracy"].item() == 0.0


def test_flow_dpo_pair_loss_rewards_better_than_reference_ordering() -> None:
    output = flow_dpo_pair_loss(
        torch.tensor([0.8]),
        torch.tensor([1.2]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=10.0,
    )

    assert output["loss"].item() < 0.2
    assert output["implicit_accuracy"].item() == 1.0


def test_reliable_preference_selection_rejects_below_noise_floor() -> None:
    selection = select_reliable_preference_pairs(
        torch.tensor([1.0, 4.0, 0.0, 2.0, 1.0, 1.1, 0.9, 1.05]),
        group_size=4,
        min_reward_gap=0.5,
    )

    assert selection["winner_indices"].tolist() == [1]
    assert selection["loser_indices"].tolist() == [2]
    assert torch.allclose(selection["reward_gaps"], torch.tensor([4.0]))
    assert selection["accepted_group_mask"].tolist() == [True, False]


def test_reliable_preference_selection_uses_tmr_as_a_constraint() -> None:
    selection = select_reliable_preference_pairs(
        torch.tensor([1.0, 5.0, 0.0, 2.0]),
        group_size=4,
        min_reward_gap=0.5,
        semantic_distances=torch.tensor([1.0, 2.0, 1.0, 1.0]),
        reference_semantic_distances=torch.ones(4),
        semantic_floor_margin=0.0,
    )

    # The physical winner at index 1 regresses against same-noise G0 and is
    # excluded. Physical reward still orders the remaining semantic-valid set.
    assert selection["winner_indices"].tolist() == [3]
    assert selection["loser_indices"].tolist() == [2]
    assert selection["eligible_mask"].tolist() == [True, False, True, True]


def test_reliable_preference_selection_respects_pairwise_cost_guard() -> None:
    selection = select_reliable_preference_pairs(
        torch.tensor([5.0, 4.0, 1.0, 0.0]),
        group_size=4,
        min_reward_gap=0.5,
        # The unconstrained [0, 3] pair trades a much worse root trajectory
        # for aggregate reward. The best root-safe pair is [1, 3].
        pairwise_guard_costs=torch.tensor([0.8, 0.2, 0.4, 0.5]),
        pairwise_guard_margins=torch.tensor([0.0]),
    )

    assert selection["winner_indices"].tolist() == [1]
    assert selection["loser_indices"].tolist() == [3]
    assert torch.allclose(selection["reward_gaps"], torch.tensor([4.0]))


def test_reliable_preference_selection_can_use_guard_margin() -> None:
    strict = select_reliable_preference_pairs(
        torch.tensor([5.0, 1.0]),
        group_size=2,
        min_reward_gap=0.5,
        pairwise_guard_costs=torch.tensor([0.31, 0.30]),
        pairwise_guard_margins=torch.tensor([0.0]),
    )
    tolerant = select_reliable_preference_pairs(
        torch.tensor([5.0, 1.0]),
        group_size=2,
        min_reward_gap=0.5,
        pairwise_guard_costs=torch.tensor([0.31, 0.30]),
        pairwise_guard_margins=torch.tensor([0.02]),
    )

    assert strict["winner_indices"].numel() == 0
    assert tolerant["winner_indices"].tolist() == [0]
    assert tolerant["loser_indices"].tolist() == [1]


def test_reliable_preference_selection_respects_multiple_pareto_guards() -> None:
    selection = select_reliable_preference_pairs(
        torch.tensor([5.0, 4.0, 1.0, 0.0]),
        group_size=4,
        min_reward_gap=0.5,
        pairwise_guard_costs=torch.tensor(
            [
                [0.2, 0.8],
                [0.3, 0.2],
                [0.4, 0.4],
                [0.5, 0.5],
            ]
        ),
        pairwise_guard_margins=torch.tensor([0.0, 0.0]),
    )

    # The aggregate winner at index 0 regresses on the second protected
    # metric. Index 1 is the largest-gap pair that is Pareto-safe on both.
    assert selection["winner_indices"].tolist() == [1]
    assert selection["loser_indices"].tolist() == [3]
    assert torch.allclose(selection["reward_gaps"], torch.tensor([4.0]))


def test_dpo_trainer_builds_lower_is_better_guard_costs() -> None:
    costs = PhysFlowG1DPOTrainer._pairwise_guard_costs(
        [
            {
                "root_trajectory_error_mean_m": 0.2,
                "er_mpjpe_mm": 40.0,
                "completion": 0.9,
                "fall_detected": False,
            },
            {
                "root_trajectory_error_mean_m": 0.3,
                "er_mpjpe_mm": 50.0,
                "completion": 0.7,
                "fall_detected": True,
            },
        ],
        (
            "root_trajectory_error_mean_m",
            "er_mpjpe_mm",
            "incompletion",
            "fall_detected",
        ),
    )

    assert costs is not None
    assert torch.allclose(
        costs,
        torch.tensor(
            [
                [0.2, 40.0, 0.1, 0.0],
                [0.3, 50.0, 0.3, 1.0],
            ]
        ),
    )


def test_g1_flow_dpo_averages_multiple_shared_pair_timesteps() -> None:
    class Reference(torch.nn.Module):
        def forward(self, *, x, **kwargs):
            del kwargs
            return torch.zeros_like(x)

    class TinyBundle:
        flow_dpo_loss_g1 = PhysFlowG1Bundle.flow_dpo_loss_g1
        pred_type = "velocity"

        def __init__(self) -> None:
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.reference = Reference()
            self.seen_batch = 0

        @staticmethod
        def _device() -> torch.device:
            return torch.device("cpu")

        @staticmethod
        def _pack_ctxt(text_ctxt, ctxt_len, device, dtype):
            del ctxt_len
            ctxt = torch.stack(text_ctxt).to(device=device, dtype=dtype)
            return ctxt, torch.ones(ctxt.shape[:2], dtype=torch.bool)

        def predict_flow(self, *, x_input, **kwargs):
            del kwargs
            self.seen_batch = int(x_input.shape[0])
            return torch.zeros_like(x_input) + self.bias

        def _reference_transformer(self):
            return self.reference

    bundle = TinyBundle()
    output = bundle.flow_dpo_loss_g1(
        text_vec=torch.zeros(2, 1),
        text_ctxt=[torch.zeros(1, 1), torch.zeros(1, 1)],
        ctxt_len=torch.ones(2, dtype=torch.long),
        winner_motion=torch.zeros(2, 3, 2),
        loser_motion=torch.ones(2, 3, 2),
        lengths=torch.full((2,), 3, dtype=torch.long),
        beta=100.0,
        timesteps_per_pair=4,
    )
    output["loss"].backward()

    assert bundle.seen_batch == 16
    assert output["timesteps_per_pair"].item() == 4
    assert torch.isfinite(bundle.bias.grad)


def test_centralized_sonic_scores_one_combined_ddp_request(monkeypatch) -> None:
    trainer = object.__new__(PhysFlowG1DPOTrainer)
    trainer.dpo_centralized_sonic = True
    seen = {}

    def fake_score(qpos, num_frames, group_size, work_dir):
        seen["shape"] = tuple(qpos.shape)
        seen["num_frames"] = list(num_frames)
        seen["group_size"] = group_size
        seen["work_dir"] = work_dir
        return [{"score": float(index)} for index in range(qpos.shape[0])]

    trainer._score_samples = fake_score
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def fake_gather(payload, object_gather_list, dst):
        assert dst == 0
        object_gather_list[0] = payload
        object_gather_list[1] = {
            "qpos": np.ones((2, 5, 36), dtype=np.float32),
            "num_frames": [5],
            "group_size": 2,
        }

    monkeypatch.setattr(torch.distributed, "gather_object", fake_gather)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda packet, src: None,
    )

    metrics = trainer._score_samples_centralized(
        np.zeros((2, 3, 36), dtype=np.float32),
        [3],
        2,
        "/tmp/central-sonic-test",
    )

    assert metrics == [{"score": 0.0}, {"score": 1.0}]
    assert seen == {
        "shape": (4, 5, 36),
        "num_frames": [3, 5],
        "group_size": 2,
        "work_dir": "/tmp/central-sonic-test",
    }


def test_dpo_sampling_unwraps_ddp_transformer() -> None:
    trainer = object.__new__(PhysFlowG1DPOTrainer)
    core = object()

    class Wrapper:
        def __init__(self, module):
            self.module = module

    trainer.bundle = type(
        "Bundle",
        (),
        {"motion_transformer": Wrapper(Wrapper(core))},
    )()

    assert trainer._live_inference_transformer() is core


def test_relative_frontier_uses_tracking_error_when_all_samples_complete() -> None:
    trainer = object.__new__(PhysFlowTrainer)
    trainer.frontier_selection = "relative_hard"
    trainer.frontier_topk_per_prompt = 1
    trainer.frontier_t_low = 0.1
    trainer.frontier_t_high = 0.95
    trainer.quality_judge = "quality"
    trainer.trainee_judge = "trainee"
    trainer.enable_reward = True
    trainer.accept_require_no_fall = True
    trainer.accept_min_completion = 0.5
    trainer.accept_require_root_metrics = False
    trainer.accept_max_joint_error_rad = None
    trainer.accept_max_root_trajectory_error_mean_m = None
    trainer.accept_max_root_displacement_error_m = None
    trainer.accept_min_joint_std = 0.0
    trainer.accept_max_root_disp_if_frozen = None
    trainer.accept_frozen_joint_std = 0.03
    trainer.quality_max_joint_vel = 0.0
    trainer.quality_max_root_vel = 0.0

    metrics = []
    for score in (0.2, 1.4, 0.7, 0.1):
        metrics.append({
            "score": 0.2,
            "completion": 1.0,
            "fall_detected": False,
            "per_judge": {
                "quality": {"score": 0.2, "completion": 1.0, "fall_detected": False},
                "trainee": {"score": score, "completion": 1.0, "fall_detected": False},
            },
        })

    assert trainer._select_frontier(metrics, batch_size=1, group_size=4) == [(0, 1)]


def test_on_policy_replay_is_not_capped_by_lagged_judge_success() -> None:
    trainer = object.__new__(PhysFlowTrainer)
    trainer.tracker_replay_selection = "on_policy_all"
    trainer.tracker_replay_samples_per_prompt = 0
    trainer.tracker_replay_max_joint_vel = 10.0
    trainer.tracker_replay_max_root_vel = 5.0

    metrics = [
        {
            "completion": 0.1,
            "fall_detected": True,
            "max_joint_vel": 2.0,
            "max_root_vel": 1.0,
        },
        {
            "completion": 1.0,
            "fall_detected": False,
            "max_joint_vel": 11.0,
            "max_root_vel": 1.0,
        },
        {"error": "rollout failed", "max_joint_vel": 1.0, "max_root_vel": 1.0},
        {
            "completion": 1.0,
            "fall_detected": False,
            "max_joint_vel": 3.0,
            "max_root_vel": 1.0,
        },
    ]

    assert trainer._select_tracker_replay(metrics, 1, 4) == [(0, 0), (0, 3)]


def test_on_policy_replay_rejects_nonfinite_reference_dynamics() -> None:
    trainer = object.__new__(PhysFlowTrainer)
    trainer.tracker_replay_selection = "on_policy_all"
    trainer.tracker_replay_samples_per_prompt = 0
    trainer.tracker_replay_max_joint_vel = 0.0
    trainer.tracker_replay_max_root_vel = 0.0

    metrics = [
        {"joint_std": 0.0, "root_disp": 0.0, "max_joint_vel": 0.0, "max_root_vel": 0.0},
        {"joint_std": float("nan"), "root_disp": 0.0},
    ]

    assert trainer._select_tracker_replay(metrics, 1, 2) == [(0, 0)]


def test_quality_valid_replay_keeps_legacy_judge_gate() -> None:
    trainer = object.__new__(PhysFlowTrainer)
    trainer.tracker_replay_selection = "quality_valid"
    trainer.tracker_replay_samples_per_prompt = 1
    trainer.quality_judge = "quality"
    trainer.trainee_judge = "trainee"
    trainer.enable_reward = True
    trainer.accept_require_no_fall = True
    trainer.accept_min_completion = 0.9
    trainer.accept_require_root_metrics = False
    trainer.accept_max_joint_error_rad = None
    trainer.accept_max_root_trajectory_error_mean_m = None
    trainer.accept_max_root_displacement_error_m = None
    trainer.accept_min_joint_std = 0.0
    trainer.accept_max_root_disp_if_frozen = None
    trainer.accept_frozen_joint_std = 0.03
    trainer.quality_max_joint_vel = 0.0
    trainer.quality_max_root_vel = 0.0

    metrics = [
        {"completion": 0.5, "fall_detected": True},
        {"completion": 1.0, "fall_detected": False},
    ]

    assert trainer._select_tracker_replay(metrics, 1, 2) == [(0, 1)]
