"""Tracker-rewarded Flow-GRPO post-training for the G1-native generator."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import numpy as np
import torch

from motius.models.gentrack.flow_grpo import (
    group_relative_advantages,
    paired_reference_advantages,
)
from motius.registry import TRAINERS
from motius.trainers.gentrack.physflow_g1_trainer import PhysFlowG1Trainer


@TRAINERS.register_module()
class PhysFlowG1GRPOTrainer(PhysFlowG1Trainer):
    """Same-prompt Flow-GRPO with a separately gated tracker replay export."""

    def __init__(
        self,
        bundle,
        grpo_eta: float = 0.7,
        grpo_clip_range: float = 0.2,
        grpo_reference_kl_weight: float = 0.01,
        grpo_advantage_eps: float = 1e-4,
        grpo_invalid_reference_penalty: float = 2.0,
        grpo_replay_epochs: int = 4,
        grpo_timesteps_per_update: int = 1,
        grpo_transition_microbatch_size: int = 0,
        gt_anchor_interval: int = 4,
        gt_anchor_update_interval: int = 0,
        gt_g0_distill_weight: float = 0.0,
        min_grpo_replay_epochs: int = 1,
        require_effective_grpo_replay: bool = False,
        grpo_reference_augmented_advantage: bool = False,
        grpo_physical_advantage_weight: float = 1.0,
        grpo_semantic_advantage_weight: float = 0.0,
        grpo_semantic_floor_weight: float = 0.0,
        grpo_semantic_floor_margin: float = 0.0,
        execution_reward_mode: str = "dense",
        paired_reward_common_envs: bool = False,
        tmr_reward_config: str | None = None,
        tmr_reward_checkpoint: str | None = "auto",
        tmr_reward_dataset_dir: str | None = None,
        tmr_reward_split: str = "train",
        tmr_reward_device: str | None = None,
        **kwargs,
    ) -> None:
        if int(grpo_replay_epochs) < max(1, int(min_grpo_replay_epochs)):
            raise ValueError(
                f"grpo_replay_epochs={int(grpo_replay_epochs)} is below "
                f"the configured min_grpo_replay_epochs="
                f"{int(min_grpo_replay_epochs)}"
            )
        if require_effective_grpo_replay and int(grpo_replay_epochs) < 2:
            raise ValueError(
                "require_effective_grpo_replay=True explicitly requires "
                "grpo_replay_epochs >= 2"
            )
        super().__init__(bundle=bundle, **kwargs)
        if self.num_samples < 2:
            raise ValueError("Flow-GRPO requires num_samples >= 2")
        self.grpo_eta = float(grpo_eta)
        self.grpo_clip_range = float(grpo_clip_range)
        self.grpo_reference_kl_weight = float(grpo_reference_kl_weight)
        self.grpo_advantage_eps = float(grpo_advantage_eps)
        self.grpo_invalid_reference_penalty = float(grpo_invalid_reference_penalty)
        self.grpo_replay_epochs = max(1, int(grpo_replay_epochs))
        self.grpo_timesteps_per_update = max(
            1,
            int(grpo_timesteps_per_update),
        )
        self.grpo_transition_microbatch_size = int(
            grpo_transition_microbatch_size
        )
        if self.grpo_transition_microbatch_size < 0:
            raise ValueError(
                "grpo_transition_microbatch_size must be non-negative"
            )
        if self.grpo_timesteps_per_update > int(self.diffusion_steps):
            raise ValueError(
                "grpo_timesteps_per_update cannot exceed diffusion_steps: "
                f"{self.grpo_timesteps_per_update} > {int(self.diffusion_steps)}"
            )
        self.gt_anchor_interval = max(0, int(gt_anchor_interval))
        self.gt_anchor_update_interval = max(0, int(gt_anchor_update_interval))
        self.gt_g0_distill_weight = float(gt_g0_distill_weight)
        self.min_grpo_replay_epochs = max(1, int(min_grpo_replay_epochs))
        self.require_effective_grpo_replay = bool(require_effective_grpo_replay)
        self.grpo_reference_augmented_advantage = bool(
            grpo_reference_augmented_advantage
        )
        self.grpo_physical_advantage_weight = float(
            grpo_physical_advantage_weight
        )
        self.grpo_semantic_advantage_weight = float(
            grpo_semantic_advantage_weight
        )
        self.grpo_semantic_floor_weight = float(grpo_semantic_floor_weight)
        self.grpo_semantic_floor_margin = float(grpo_semantic_floor_margin)
        self.execution_reward_mode = str(execution_reward_mode).lower()
        if self.execution_reward_mode not in {"dense", "success", "none"}:
            raise ValueError(
                "execution_reward_mode must be dense, success, or none, got "
                f"{execution_reward_mode!r}"
            )
        self.paired_reward_common_envs = bool(paired_reward_common_envs)
        if self.grpo_physical_advantage_weight < 0:
            raise ValueError("grpo_physical_advantage_weight must be non-negative")
        if self.grpo_semantic_advantage_weight < 0:
            raise ValueError("grpo_semantic_advantage_weight must be non-negative")
        if self.grpo_semantic_floor_weight < 0:
            raise ValueError("grpo_semantic_floor_weight must be non-negative")
        self.tmr_reward_split = str(tmr_reward_split)
        self._tmr_reward_evaluator = None
        needs_reference_samples = (
            self.grpo_reference_augmented_advantage
            or self.grpo_semantic_advantage_weight > 0
            or self.grpo_semantic_floor_weight > 0
        )
        self._needs_reference_samples = needs_reference_samples
        if needs_reference_samples and not getattr(
            bundle, "require_immutable_anchor", False
        ):
            raise ValueError(
                "reference-augmented/semantic FlowGRPO requires "
                "require_immutable_anchor=True"
            )
        if (
            self.grpo_semantic_advantage_weight > 0
            or self.grpo_semantic_floor_weight > 0
        ):
            from motius.evaluation.evaluators.tmr import TMRG1Evaluator

            evaluator_kwargs: Dict[str, Any] = {
                "checkpoint": tmr_reward_checkpoint,
                "dataset_dir": tmr_reward_dataset_dir,
                "device": tmr_reward_device or str(bundle._device()),
                "batch_size": max(8, int(self.num_samples)),
                "n_repeats": 1,
            }
            if tmr_reward_config:
                evaluator_kwargs["config"] = tmr_reward_config
            self._tmr_reward_evaluator = TMRG1Evaluator(**evaluator_kwargs)
        self._grpo_replay_cache: Dict[str, Any] | None = None
        self._grpo_rollouts_since_anchor = self.gt_anchor_interval
        self._grpo_updates_since_anchor = self.gt_anchor_update_interval
        if getattr(bundle, "immutable_anchor_checkpoint", None) or getattr(
            bundle, "require_immutable_anchor", False
        ):
            bundle.init_immutable_g0_anchor()

    @staticmethod
    def _cpu_detached(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, list):
            return [PhysFlowG1GRPOTrainer._cpu_detached(item) for item in value]
        return value

    def _execution_reward(self, metric: Dict[str, Any]) -> float:
        """Map one rollout to the named execution-reward ablation."""
        if self.execution_reward_mode == "none":
            return 0.0
        if self.execution_reward_mode == "success":
            return float(
                not metric.get("error")
                and not bool(metric.get("fall_detected", True))
            )
        return -float(metric.get("physical_score", metric.get("score", 1e6)))

    def _grpo_loss_from_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        return self.bundle.grpo_loss_g1(
            cache["vtxt"],
            cache["ctxt"],
            cache["ctxt_len"],
            cache["lengths"],
            states=cache["states"],
            old_log_probs=cache["old_log_probs"],
            sigmas=cache["sigmas"],
            advantages=cache["advantages"],
            eta=self.grpo_eta,
            clip_range=self.grpo_clip_range,
            reference_kl_weight=self.grpo_reference_kl_weight,
            timesteps_per_update=self.grpo_timesteps_per_update,
            transition_microbatch_size=self.grpo_transition_microbatch_size,
        )

    def _format_grpo_result(
        self,
        loss_out: Dict[str, Any],
        cache: Dict[str, Any],
        replay_epoch: int,
        new_rollout: bool,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "loss": loss_out["loss"],
            "loss_grpo": loss_out["policy_loss"],
            "loss_reference_kl": loss_out["reference_kl"],
            "grpo_ratio_mean": loss_out["ratio_mean"],
            "grpo_ratio_abs_deviation_mean": loss_out[
                "ratio_abs_deviation_mean"
            ],
            "grpo_log_ratio_abs_mean": loss_out["log_ratio_abs_mean"],
            "grpo_advantage_log_ratio_cov": loss_out[
                "advantage_log_ratio_cov"
            ],
            "grpo_advantage_log_ratio_alignment": loss_out[
                "advantage_log_ratio_alignment"
            ],
            "grpo_positive_advantage_log_ratio_mean": loss_out[
                "positive_advantage_log_ratio_mean"
            ],
            "grpo_negative_advantage_log_ratio_mean": loss_out[
                "negative_advantage_log_ratio_mean"
            ],
            "grpo_clip_fraction": loss_out["clip_fraction"],
            "grpo_behavior_log_prob": loss_out["behavior_log_prob"],
            "grpo_current_log_prob": loss_out["current_log_prob"],
            "grpo_timestep_mean": loss_out["timestep_mean"],
            "grpo_timesteps_per_update": loss_out["timesteps_per_update"],
            "grpo_transition_microbatch_size": loss_out[
                "transition_microbatch_size"
            ],
            "grpo_replay_epoch": torch.tensor(float(replay_epoch)),
            "grpo_new_rollout": torch.tensor(float(new_rollout)),
            "reward_mean": cache["reward_mean"],
            "reward_std": cache["reward_std"],
            "advantage_abs_mean": cache["advantage_abs_mean"],
            "n_valid": cache["n_valid"],
            "n_frontier": cache["n_frontier"],
            "n_reward_errors": cache["n_reward_errors"],
            "update_is_gt_anchor": torch.tensor(0.0),
        }
        if self.tracker_pool_dir:
            result["n_pooled"] = torch.tensor(
                float(cache["n_pooled"] if new_rollout else 0.0)
            )
        if self.tracker_qpos_pool_dir:
            result["n_qpos_pooled"] = torch.tensor(
                float(cache["n_qpos_pooled"] if new_rollout else 0.0)
            )
        for name, value in cache.get("reward_telemetry", {}).items():
            result[name] = torch.tensor(float(value))
        return result

    @staticmethod
    def _reward_telemetry(metrics: list[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate the public SONIC components for early failure detection."""

        telemetry: Dict[str, float] = {}

        def add_mean(output_name: str, values: list[float]) -> None:
            if values:
                telemetry[output_name] = sum(values) / len(values)

        add_mean(
            "judge_completion_mean",
            [float(metric["completion"]) for metric in metrics if "completion" in metric],
        )
        add_mean(
            "judge_fall_rate",
            [
                float(bool(metric["fall_detected"]))
                for metric in metrics
                if "fall_detected" in metric
            ],
        )
        component_names = (
            "global_anchor_pos",
            "global_anchor_ori",
            "relative_body_pos",
            "relative_body_ori",
            "body_lin_vel",
            "body_ang_vel",
        )
        for component_name in component_names:
            values = []
            for metric in metrics:
                components = metric.get("reward_components")
                if isinstance(components, dict) and component_name in components:
                    values.append(float(components[component_name]))
            add_mean(f"judge_{component_name}_reward_mean", values)
        return telemetry

    @torch.no_grad()
    def _denormalized_motion_arrays(
        self,
        latents: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list:
        raw = self.bundle.denormalize_motion(
            latents.to(self.bundle._device()).float()
        ).detach().cpu().numpy()
        return [
            raw[index, : int(length)].copy()
            for index, length in enumerate(lengths.tolist())
        ]

    def _semantic_pair_distances(
        self,
        captions: list[str],
        latents: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self._tmr_reward_evaluator is None:
            raise RuntimeError("TMR-G1 semantic reward is not initialized")
        motions = self._denormalized_motion_arrays(latents, lengths)
        distances = self._tmr_reward_evaluator.score_text_motion_pairs(
            captions,
            motions,
            split=self.tmr_reward_split,
        )
        return torch.as_tensor(distances, dtype=torch.float32)

    @staticmethod
    def _paired_motion_telemetry(
        candidate_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, float]:
        """Separate generator drift from stochastic tracker-rollout noise."""
        if candidate_latents.shape != reference_latents.shape:
            raise ValueError(
                "paired candidate/reference latents must have identical shape, "
                f"got {tuple(candidate_latents.shape)} and "
                f"{tuple(reference_latents.shape)}"
            )
        if lengths.shape != (candidate_latents.shape[0],):
            raise ValueError(
                f"lengths must be ({candidate_latents.shape[0]},), "
                f"got {tuple(lengths.shape)}"
            )
        frame_count = candidate_latents.shape[1]
        frame_mask = (
            torch.arange(frame_count, device=candidate_latents.device)[None]
            < lengths.to(candidate_latents.device)[:, None]
        ).unsqueeze(-1)
        squared_error = (
            candidate_latents.detach() - reference_latents.detach()
        ).float().square()
        valid = frame_mask.expand_as(squared_error)
        absolute_error = squared_error.sqrt()
        return {
            "candidate_vs_g0_motion_mse": float(
                squared_error.masked_select(valid).mean().cpu()
            ),
            "candidate_vs_g0_motion_max_abs": float(
                absolute_error.masked_select(valid).max().cpu()
            ),
        }

    def _compose_advantages(
        self,
        candidate_physical_rewards: torch.Tensor,
        group_size: int,
        reference_physical_rewards: torch.Tensor | None = None,
        candidate_semantic_distances: torch.Tensor | None = None,
        reference_semantic_distances: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        telemetry: Dict[str, float] = {}
        if reference_physical_rewards is None:
            grouped_physical_rewards = candidate_physical_rewards.reshape(
                -1, group_size
            )
            physical_group_std = grouped_physical_rewards.std(
                dim=1,
                unbiased=False,
            )
            physical_advantages = group_relative_advantages(
                candidate_physical_rewards,
                group_size=group_size,
                eps=self.grpo_advantage_eps,
            )
            telemetry.update(
                {
                    "physical_reward_group_mean": float(
                        grouped_physical_rewards.mean()
                    ),
                    "physical_reward_group_best_mean": float(
                        grouped_physical_rewards.max(dim=1).values.mean()
                    ),
                    "physical_reward_group_best_minus_mean": float(
                        (
                            grouped_physical_rewards.max(dim=1).values
                            - grouped_physical_rewards.mean(dim=1)
                        ).mean()
                    ),
                    "physical_reward_group_std_mean": float(
                        physical_group_std.mean()
                    ),
                    "physical_reward_group_std_min": float(
                        physical_group_std.min()
                    ),
                    "physical_reward_group_std_max": float(
                        physical_group_std.max()
                    ),
                    "physical_reward_group_std_below_001_fraction": float(
                        (physical_group_std < 0.01).float().mean()
                    ),
                    "physical_reward_group_std_below_005_fraction": float(
                        (physical_group_std < 0.05).float().mean()
                    ),
                }
            )
        else:
            physical_advantages = paired_reference_advantages(
                candidate_physical_rewards,
                reference_physical_rewards,
                group_size=group_size,
                eps=self.grpo_advantage_eps,
            )
            physical_delta = (
                candidate_physical_rewards - reference_physical_rewards
            )
            telemetry.update(
                {
                    "g0_physical_reward_mean": float(
                        reference_physical_rewards.mean()
                    ),
                    "candidate_vs_g0_physical_delta_mean": float(
                        physical_delta.mean()
                    ),
                    "candidate_vs_g0_physical_improved_fraction": float(
                        (physical_delta > 0).float().mean()
                    ),
                }
            )

        advantages = (
            self.grpo_physical_advantage_weight * physical_advantages
        )
        if candidate_semantic_distances is not None:
            if reference_semantic_distances is None:
                raise ValueError(
                    "candidate semantic distances require paired G0 distances"
                )
            semantic_advantages = paired_reference_advantages(
                -candidate_semantic_distances,
                -reference_semantic_distances,
                group_size=group_size,
                eps=self.grpo_advantage_eps,
            )
            advantages = (
                advantages
                + self.grpo_semantic_advantage_weight * semantic_advantages
            )

            candidate_grouped = candidate_semantic_distances.reshape(
                -1, group_size
            )
            reference_grouped = reference_semantic_distances.reshape(
                -1, group_size
            )
            semantic_scale = torch.cat(
                [candidate_grouped, reference_grouped], dim=1
            ).std(dim=1, unbiased=False, keepdim=True).clamp_min(
                self.grpo_advantage_eps
            )
            semantic_regression = (
                candidate_grouped
                - reference_grouped
                - self.grpo_semantic_floor_margin
            ).clamp_min(0.0)
            normalized_regression = (
                semantic_regression / semantic_scale
            ).reshape(-1)
            advantages = (
                advantages
                - self.grpo_semantic_floor_weight * normalized_regression
            )

            semantic_delta = (
                reference_semantic_distances
                - candidate_semantic_distances
            )
            telemetry.update(
                {
                    "tmr_candidate_distance_mean": float(
                        candidate_semantic_distances.mean()
                    ),
                    "tmr_g0_distance_mean": float(
                        reference_semantic_distances.mean()
                    ),
                    "candidate_vs_g0_tmr_delta_mean": float(
                        semantic_delta.mean()
                    ),
                    "candidate_vs_g0_tmr_improved_fraction": float(
                        (semantic_delta > 0).float().mean()
                    ),
                    "tmr_floor_violation_mean": float(
                        normalized_regression.mean()
                    ),
                }
            )
        telemetry.update(
            {
                "composite_advantage_mean": float(advantages.mean()),
                "composite_advantage_positive_fraction": float(
                    (advantages > 0).float().mean()
                ),
            }
        )
        return advantages, telemetry

    def _replay_grpo_step(self) -> Dict[str, Any]:
        cache = self._grpo_replay_cache
        if cache is None:
            raise RuntimeError("Flow-GRPO replay requested without a cached trajectory")
        replay_epoch = int(cache["next_replay_epoch"])
        loss_out = self._grpo_loss_from_cache(cache)
        result = self._format_grpo_result(
            loss_out,
            cache,
            replay_epoch=replay_epoch,
            new_rollout=False,
        )
        if replay_epoch >= self.grpo_replay_epochs:
            self._grpo_replay_cache = None
        else:
            cache["next_replay_epoch"] = replay_epoch + 1
        return result

    def _gt_anchor_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "motion" not in batch:
            raise ValueError("GT anchor step requires batch['motion']")
        target = self.bundle.normalize_motion(
            batch["motion"].to(self.bundle._device()).float()
        ).detach()
        batch_size = target.shape[0]
        out = self.bundle.sft_loss_g1(
            batch["text_vec_raw"],
            list(batch["text_ctxt_raw"]),
            batch["text_ctxt_raw_length"],
            target,
            batch["tgt_length"],
            good_mask=torch.ones(batch_size),
            anchor_weight=self.gt_g0_distill_weight,
        )
        weight = self.gt_weight if self.gt_weight > 0 else 1.0
        result = {
            "loss": out["loss"] * float(weight),
            "loss_gt_anchor": out["sft_mse"],
            "update_is_gt_anchor": torch.tensor(1.0),
        }
        if "anchor_mse" in out:
            result["loss_gt_g0_distill"] = out["anchor_mse"]
        return result

    def _score_candidate_and_reference(
        self,
        qpos,
        reference_qpos,
        num_frames,
        group_size: int,
        work_dir: str,
    ):
        """Score paired live/G0 samples in one persistent simulator request."""
        if reference_qpos is None:
            return (
                self._score_samples(qpos, num_frames, group_size, work_dir),
                None,
            )
        if tuple(qpos.shape) != tuple(reference_qpos.shape):
            raise ValueError(
                "paired live/G0 qpos shapes must match, got "
                f"{tuple(qpos.shape)} and {tuple(reference_qpos.shape)}"
            )
        if getattr(self, "paired_reward_common_envs", False):
            # Each warm-service request is reseeded. Separate requests map the
            # paired motions to identical environment IDs and reset noise;
            # concatenation maps G-theta and G0 to disjoint environment IDs.
            candidate_metrics = self._score_samples(
                qpos,
                num_frames,
                group_size,
                os.path.join(work_dir, "candidate"),
            )
            reference_metrics = self._score_samples(
                reference_qpos,
                num_frames,
                group_size,
                os.path.join(work_dir, "reference"),
            )
            return candidate_metrics, reference_metrics
        if torch.is_tensor(qpos):
            if not torch.is_tensor(reference_qpos):
                raise TypeError("paired live/G0 qpos must use the same array type")
            paired_qpos = torch.cat((qpos, reference_qpos), dim=0)
        else:
            paired_qpos = np.concatenate(
                (np.asarray(qpos), np.asarray(reference_qpos)),
                axis=0,
            )
        paired_frames = list(num_frames) + list(num_frames)
        paired_metrics = self._score_samples(
            paired_qpos,
            paired_frames,
            group_size,
            work_dir,
        )
        split = int(qpos.shape[0])
        if len(paired_metrics) != 2 * split:
            raise RuntimeError(
                "paired simulator request returned "
                f"{len(paired_metrics)} metrics for {2 * split} motions"
            )
        return paired_metrics[:split], paired_metrics[split:]

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # A rollout can be replayed for several optimizer updates.  Counting
        # only new rollouts makes GT anchoring unexpectedly sparse (for
        # example, interval=2 with eight replay epochs anchors about 6% of
        # optimizer steps).  Formal runs therefore use the update-based
        # interval, which is checked before replay so anchors are genuinely
        # interleaved with policy updates.  The rollout-based option remains
        # available for legacy ablations when update_interval is zero.
        if (
            self.gt_anchor_update_interval > 0
            and self.gt_weight > 0
            and self._grpo_updates_since_anchor
            >= self.gt_anchor_update_interval
        ):
            self._grpo_updates_since_anchor = 0
            return self._gt_anchor_step(batch)

        if self._grpo_replay_cache is not None:
            result = self._replay_grpo_step()
            self._grpo_updates_since_anchor += 1
            return result

        if (
            self.gt_anchor_update_interval == 0
            and self.gt_anchor_interval > 0
            and self.gt_weight > 0
            and self._grpo_rollouts_since_anchor >= self.gt_anchor_interval
        ):
            self._grpo_rollouts_since_anchor = 0
            return self._gt_anchor_step(batch)

        step = self._global_step()

        vtxt = batch["text_vec_raw"]
        ctxt = list(batch["text_ctxt_raw"])
        ctxt_len = batch["text_ctxt_raw_length"]
        num_frames = [int(value) for value in batch["tgt_length"].tolist()]
        batch_size = vtxt.shape[0]
        group_size = self.num_samples

        vtxt_rep = vtxt.repeat_interleave(group_size, dim=0)
        ctxt_rep = [value for value in ctxt for _ in range(group_size)]
        ctxt_len_rep = ctxt_len.repeat_interleave(group_size, dim=0)
        lengths_rep = torch.tensor(
            [length for length in num_frames for _ in range(group_size)],
            dtype=torch.long,
        )
        behavior = self.bundle.sample_motion_grpo(
            vtxt_rep,
            ctxt_rep,
            ctxt_len_rep,
            lengths_rep,
            num_steps=self.diffusion_steps,
            guidance=1.0,
            eta=self.grpo_eta,
        )
        latents = behavior["sample"]
        qpos = self.bundle.latents_to_qpos(latents)
        reference_latents = None
        reference_qpos = None
        if self._needs_reference_samples:
            reference_transformer = self.bundle._reference_transformer()
            if reference_transformer is None:
                raise RuntimeError(
                    "paired G0 reward requested but immutable reference is unavailable"
                )
            reference_behavior = self.bundle.sample_motion_grpo(
                vtxt_rep,
                ctxt_rep,
                ctxt_len_rep,
                lengths_rep,
                num_steps=self.diffusion_steps,
                guidance=1.0,
                eta=self.grpo_eta,
                initial_noise=behavior["initial_noise"],
                transition_noises=behavior["transition_noises"],
                transformer=reference_transformer,
                return_policy_artifacts=False,
            )
            reference_latents = reference_behavior["sample"]
            reference_qpos = self.bundle.latents_to_qpos(reference_latents)

        rollout = tempfile.TemporaryDirectory(
            prefix=f"physflow_g1_grpo_step{step}_",
            dir=self.rollout_dir,
        )
        try:
            # Standard FlowGRPO uses same-prompt group-relative physical
            # rewards.  The immutable G0 motion is still generated for the
            # deterministic semantic floor, but it should enter the simulator
            # only in the explicit paired-physical ablation.  This avoids
            # turning warm-simulator reset variance into a policy advantage.
            physical_reference_qpos = (
                reference_qpos
                if self.grpo_reference_augmented_advantage
                else None
            )
            metrics, reference_metrics = self._score_candidate_and_reference(
                qpos,
                physical_reference_qpos,
                num_frames,
                group_size,
                rollout.name,
            )
            reward_errors = [
                str(metric.get("error"))
                for metric in metrics
                if metric.get("error")
            ]
            if self.enable_reward and reward_errors and len(reward_errors) == len(metrics):
                raise RuntimeError(
                    "All tracker-reward rollouts failed; refusing a zero-signal "
                    f"Flow-GRPO update. First error: {reward_errors[0]}"
                )
            for prompt_index in range(batch_size):
                for sample_index in range(group_size):
                    flat = prompt_index * group_size + sample_index
                    metrics[flat].update(
                        self._motion_dynamics(qpos[flat], num_frames[prompt_index])
                    )
            self._add_style_costs(
                metrics,
                qpos,
                num_frames,
                group_size,
                captions=list(batch.get("caption", [])),
            )

            if reference_metrics is not None:
                for prompt_index in range(batch_size):
                    for sample_index in range(group_size):
                        flat = prompt_index * group_size + sample_index
                        reference_metrics[flat].update(
                            self._motion_dynamics(
                                reference_qpos[flat],
                                num_frames[prompt_index],
                            )
                        )

            rewards = []
            reference_rewards = []
            valid = []
            frontier = []
            for prompt_index in range(batch_size):
                for sample_index in range(group_size):
                    flat = prompt_index * group_size + sample_index
                    metric = metrics[flat]
                    reward = self._execution_reward(metric)
                    if not self._is_kinematically_valid(metric):
                        reward -= self.grpo_invalid_reference_penalty
                    rewards.append(reward)
                    if reference_metrics is not None:
                        reference_metric = reference_metrics[flat]
                        reference_reward = self._execution_reward(reference_metric)
                        if not self._is_kinematically_valid(reference_metric):
                            reference_reward -= self.grpo_invalid_reference_penalty
                        reference_rewards.append(reference_reward)
                    if self._is_valid(metric):
                        valid.append((prompt_index, sample_index))
            if self.frontier_mode:
                frontier = self._select_frontier(metrics, batch_size, group_size)

            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            reference_rewards_t = (
                torch.tensor(reference_rewards, dtype=torch.float32)
                if reference_rewards
                else None
            )
            captions = [str(value) for value in batch.get("caption", [])]
            candidate_semantic_distances = None
            reference_semantic_distances = None
            if self._tmr_reward_evaluator is not None:
                if len(captions) != batch_size:
                    raise ValueError(
                        "TMR-G1 semantic reward requires one caption per prompt"
                    )
                captions_rep = [
                    caption
                    for caption in captions
                    for _ in range(group_size)
                ]
                candidate_semantic_distances = self._semantic_pair_distances(
                    captions_rep,
                    latents,
                    lengths_rep,
                )
                if reference_latents is None:
                    raise RuntimeError(
                        "TMR-G1 semantic reward requires paired G0 latents"
                    )
                reference_semantic_distances = self._semantic_pair_distances(
                    captions_rep,
                    reference_latents,
                    lengths_rep,
                )
            advantages, advantage_telemetry = self._compose_advantages(
                rewards_t,
                group_size=group_size,
                reference_physical_rewards=(
                    reference_rewards_t
                    if self.grpo_reference_augmented_advantage
                    else None
                ),
                candidate_semantic_distances=candidate_semantic_distances,
                reference_semantic_distances=reference_semantic_distances,
            )
            reward_telemetry = self._reward_telemetry(metrics)
            if reference_latents is not None:
                reward_telemetry.update(
                    self._paired_motion_telemetry(
                        latents,
                        reference_latents,
                        lengths_rep,
                    )
                )
            reward_telemetry.update(advantage_telemetry)
            if reference_metrics is not None:
                for name, value in self._reward_telemetry(
                    reference_metrics
                ).items():
                    reward_telemetry[f"g0_{name}"] = value

            selected = self._select_tracker_replay(metrics, batch_size, group_size)
            prompt_ids = list(batch.get("prompt_id", [])) or list(batch.get("caption", []))
            n_pooled = 0
            n_qpos_pooled = 0
            if self.tracker_pool_dir and selected:
                n_pooled = self._export_to_pool(
                    os.path.join(rollout.name, "proto"),
                    selected,
                    prompt_ids,
                )
            if self.tracker_qpos_pool_dir and selected:
                n_qpos_pooled = self._export_qpos_to_pool(
                    qpos,
                    num_frames,
                    selected,
                    prompt_ids,
                )
        finally:
            if self.keep_rollouts:
                try:
                    rollout._finalizer.detach()
                except Exception:
                    pass
            else:
                rollout.cleanup()

        cache: Dict[str, Any] = {
            "vtxt": self._cpu_detached(vtxt_rep),
            "ctxt": self._cpu_detached(ctxt_rep),
            "ctxt_len": self._cpu_detached(ctxt_len_rep),
            "lengths": self._cpu_detached(lengths_rep),
            "states": self._cpu_detached(behavior["states"]),
            "old_log_probs": self._cpu_detached(behavior["old_log_probs"]),
            "sigmas": self._cpu_detached(behavior["sigmas"]),
            "advantages": self._cpu_detached(advantages),
            "target_latents": self._cpu_detached(latents),
            "reward_mean": rewards_t.mean(),
            "reward_std": rewards_t.std(unbiased=False),
            "advantage_abs_mean": advantages.abs().mean(),
            "n_valid": torch.tensor(float(len(valid))),
            "n_frontier": torch.tensor(float(len(frontier))),
            "n_reward_errors": torch.tensor(float(len(reward_errors))),
            "n_pooled": float(n_pooled),
            "n_qpos_pooled": float(n_qpos_pooled),
            "reward_telemetry": reward_telemetry,
            "next_replay_epoch": 2,
        }
        loss_out = self._grpo_loss_from_cache(cache)
        if self.grpo_replay_epochs > 1:
            self._grpo_replay_cache = cache
        self._grpo_rollouts_since_anchor += 1
        self._grpo_updates_since_anchor += 1
        return self._format_grpo_result(
            loss_out,
            cache,
            replay_epoch=1,
            new_rollout=True,
        )


@TRAINERS.register_module()
class PhysFlowG1RewardWeightedSFTTrainer(PhysFlowG1GRPOTrainer):
    """Same-prompt reward-weighted SFT under the FlowGRPO rollout budget."""

    def _grpo_loss_from_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        advantages = cache["advantages"].float()
        grouped = advantages.reshape(-1, self.num_samples)
        positive = grouped.clamp_min(0.0)
        positive_count = (positive > 0).sum(dim=1, keepdim=True)
        positive_sum = positive.sum(dim=1, keepdim=True)
        weights = torch.where(
            positive_sum > 0,
            positive * positive_count.clamp_min(1) / positive_sum.clamp_min(1e-8),
            torch.zeros_like(positive),
        ).reshape(-1)
        good_mask = (weights > 0).float()
        out = self.bundle.sft_loss_g1(
            cache["vtxt"],
            cache["ctxt"],
            cache["ctxt_len"],
            cache["target_latents"],
            cache["lengths"],
            good_mask=good_mask,
            sample_weights=weights,
            anchor_weight=float(getattr(self, "anchor_weight", 0.0)),
        )
        zero = out["loss"].detach().new_zeros(())
        one = out["loss"].detach().new_ones(())
        return {
            "loss": out["loss"],
            "policy_loss": out["sft_mse"],
            "reference_kl": out.get("anchor_mse", zero),
            "ratio_mean": one,
            "ratio_abs_deviation_mean": zero,
            "log_ratio_abs_mean": zero,
            "advantage_log_ratio_cov": zero,
            "advantage_log_ratio_alignment": zero,
            "positive_advantage_log_ratio_mean": zero,
            "negative_advantage_log_ratio_mean": zero,
            "clip_fraction": zero,
            "current_log_prob": zero,
            "behavior_log_prob": zero,
            "timestep_mean": zero,
            "timesteps_per_update": zero,
            "transition_microbatch_size": zero,
        }
