"""Round-synchronous Flow-DPO post-training for the G1 generator."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from motius.models.gentrack.flow_grpo import (
    select_reliable_preference_pairs,
)
from motius.registry import TRAINERS
from motius.trainers.gentrack.physflow_g1_grpo_trainer import (
    PhysFlowG1GRPOTrainer,
)


@TRAINERS.register_module()
class PhysFlowG1DPOTrainer(PhysFlowG1GRPOTrainer):
    """Use reliable same-prompt SONIC preferences for Flow-DPO updates.

    ``Accelerator.gradient_accumulation_steps`` defines one frozen-policy
    collection round. No model update occurs until every prompt group in the
    round has contributed its accepted pair or an explicit zero-gradient skip.
    """

    _PAIRWISE_GUARD_METRICS = frozenset(
        {
            "root_trajectory_error_mean_m",
            "er_mpjpe_mm",
            "evel_mps",
            "eacc_mps2",
            "root_vel_err_mps",
            "mpjpe_mm",
            "mpjve_mps",
            "fall_detected",
            "incompletion",
        }
    )

    def __init__(
        self,
        bundle,
        dpo_beta: float = 100.0,
        dpo_min_reward_gap: float = 0.2,
        dpo_timesteps_per_pair: int = 8,
        dpo_replay_epochs: int = 4,
        dpo_semantic_floor_margin: float = 0.0,
        dpo_require_semantic_guard: bool = True,
        dpo_pairwise_root_margin_m: float | None = None,
        dpo_pairwise_guard_margins: Dict[str, float] | None = None,
        dpo_pair_audit_dir: str | None = None,
        dpo_round_id: str = "round0",
        dpo_centralized_sonic: bool = True,
        dpo_sonic_gpu_base: int | None = None,
        dpo_sonic_gpu_rank_stride: int = 0,
        **kwargs,
    ) -> None:
        if float(dpo_beta) <= 0:
            raise ValueError("dpo_beta must be positive")
        if float(dpo_min_reward_gap) < 0:
            raise ValueError("dpo_min_reward_gap must be non-negative")
        if int(dpo_timesteps_per_pair) < 1:
            raise ValueError("dpo_timesteps_per_pair must be positive")
        if int(dpo_replay_epochs) < 1:
            raise ValueError("dpo_replay_epochs must be positive")
        if float(dpo_semantic_floor_margin) < 0:
            raise ValueError("dpo_semantic_floor_margin must be non-negative")
        if (
            dpo_pairwise_root_margin_m is not None
            and float(dpo_pairwise_root_margin_m) < 0
        ):
            raise ValueError("dpo_pairwise_root_margin_m must be non-negative")
        pairwise_guard_margins = {
            str(name): float(margin)
            for name, margin in dict(
                dpo_pairwise_guard_margins or {}
            ).items()
        }
        unknown_guards = (
            set(pairwise_guard_margins) - self._PAIRWISE_GUARD_METRICS
        )
        if unknown_guards:
            raise ValueError(
                "unsupported DPO pairwise guard metrics: "
                f"{sorted(unknown_guards)}"
            )
        if any(margin < 0 for margin in pairwise_guard_margins.values()):
            raise ValueError("DPO pairwise guard margins must be non-negative")
        root_guard_name = "root_trajectory_error_mean_m"
        if (
            dpo_pairwise_root_margin_m is not None
            and root_guard_name in pairwise_guard_margins
        ):
            raise ValueError(
                "configure the root pairwise guard through either "
                "dpo_pairwise_root_margin_m or dpo_pairwise_guard_margins, "
                "not both"
            )
        if dpo_pairwise_root_margin_m is not None:
            pairwise_guard_margins[root_guard_name] = float(
                dpo_pairwise_root_margin_m
            )

        # The GRPO parent owns the released-SONIC and TMR-G1 infrastructure.
        # DPO does not replay PPO trajectories or mix semantic/physical scalars.
        kwargs["grpo_replay_epochs"] = 1
        kwargs["min_grpo_replay_epochs"] = 1
        kwargs["require_effective_grpo_replay"] = False
        kwargs["grpo_reference_augmented_advantage"] = False
        kwargs["grpo_semantic_advantage_weight"] = 0.0
        kwargs["grpo_semantic_floor_weight"] = (
            1.0 if dpo_require_semantic_guard else 0.0
        )
        super().__init__(bundle=bundle, **kwargs)

        if not self.enable_reward:
            raise ValueError("Flow-DPO requires an execution reward")
        if self.style_reward_weight > 0:
            raise ValueError(
                "Flow-DPO physical ordering must not include an ad-hoc style scalar"
            )
        if dpo_require_semantic_guard and self._tmr_reward_evaluator is None:
            raise ValueError("semantic-guarded Flow-DPO requires TMR-G1")

        self.dpo_beta = float(dpo_beta)
        self.dpo_min_reward_gap = float(dpo_min_reward_gap)
        self.dpo_timesteps_per_pair = int(dpo_timesteps_per_pair)
        self.dpo_replay_epochs = int(dpo_replay_epochs)
        self.dpo_semantic_floor_margin = float(dpo_semantic_floor_margin)
        self.dpo_require_semantic_guard = bool(dpo_require_semantic_guard)
        self.dpo_pairwise_root_margin_m = (
            float(dpo_pairwise_root_margin_m)
            if dpo_pairwise_root_margin_m is not None
            else None
        )
        self.dpo_pairwise_guard_names = tuple(pairwise_guard_margins)
        self.dpo_pairwise_guard_margins = tuple(
            pairwise_guard_margins[name]
            for name in self.dpo_pairwise_guard_names
        )
        self.dpo_pair_audit_dir = (
            Path(dpo_pair_audit_dir)
            if dpo_pair_audit_dir not in (None, "")
            else None
        )
        self.dpo_round_id = str(dpo_round_id)
        self.dpo_centralized_sonic = bool(dpo_centralized_sonic)
        self.dpo_sonic_gpu_base = (
            int(dpo_sonic_gpu_base)
            if dpo_sonic_gpu_base is not None
            else None
        )
        self.dpo_sonic_gpu_rank_stride = int(dpo_sonic_gpu_rank_stride)
        self._dpo_rank_gpu_configured = False
        self._dpo_replay_cache: Dict[str, Any] | None = None
        if self.dpo_pair_audit_dir is not None:
            self.dpo_pair_audit_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _pairwise_guard_costs(
        cls,
        metrics: list[Dict[str, Any]],
        guard_names: tuple[str, ...],
    ) -> torch.Tensor | None:
        """Build lower-is-better costs for Pareto-safe pair selection."""

        if not guard_names:
            return None
        rows = []
        for metric in metrics:
            values = []
            for name in guard_names:
                if name == "incompletion":
                    value = 1.0 - float(metric.get("completion", float("-inf")))
                elif name == "fall_detected":
                    value = float(bool(metric.get("fall_detected", True)))
                else:
                    value = float(metric.get(name, float("inf")))
                values.append(value)
            rows.append(values)
        return torch.tensor(rows, dtype=torch.float32)

    def _configure_rank_local_sonic_gpu(self) -> None:
        if self._dpo_rank_gpu_configured:
            return
        if self._reward is not None:
            raise RuntimeError("rank-local SONIC GPU must be set before service start")
        if self.dpo_sonic_gpu_base is not None:
            local_rank = int(
                getattr(self.accelerator, "local_process_index", 0)
                if self.accelerator is not None
                else 0
            )
            if self.dpo_centralized_sonic:
                self.sonic_gpu_id = self.dpo_sonic_gpu_base
            else:
                self.sonic_gpu_id = (
                    self.dpo_sonic_gpu_base
                    + local_rank * self.dpo_sonic_gpu_rank_stride
                )
        self._dpo_rank_gpu_configured = True

    def _live_inference_transformer(self):
        """Use live weights for sampling without advancing the DDP reducer."""

        transformer = self.bundle.motion_transformer
        while hasattr(transformer, "module"):
            transformer = transformer.module
        return transformer

    def _score_samples_centralized(
        self,
        qpos: np.ndarray,
        num_frames: list[int],
        group_size: int,
        work_dir: str,
    ) -> list[Dict[str, Any]]:
        """Score all DDP candidates with one persistent Isaac Sim process."""
        distributed = (
            self.dpo_centralized_sonic
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if not distributed:
            return self._score_samples(qpos, num_frames, group_size, work_dir)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_qpos = np.asarray(qpos, dtype=np.float32)
        local_frames = [int(value) for value in num_frames]
        if local_qpos.shape[0] != len(local_frames) * int(group_size):
            raise ValueError(
                "centralized SONIC expects one frame length per prompt: "
                f"qpos={local_qpos.shape[0]}, prompts={len(local_frames)}, "
                f"group_size={group_size}"
            )

        gathered = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(
            {
                "qpos": local_qpos,
                "num_frames": local_frames,
                "group_size": int(group_size),
            },
            object_gather_list=gathered,
            dst=0,
        )

        packet = [None]
        if rank == 0:
            try:
                assert gathered is not None
                if any(
                    int(payload["group_size"]) != int(group_size)
                    for payload in gathered
                ):
                    raise RuntimeError(
                        "all DDP ranks must use the same DPO group size"
                    )
                qpos_batches = [
                    np.asarray(payload["qpos"], dtype=np.float32)
                    for payload in gathered
                ]
                if any(batch.ndim != 3 for batch in qpos_batches):
                    raise RuntimeError(
                        "central SONIC expects qpos batches shaped [B, T, D]"
                    )
                feature_dims = {int(batch.shape[2]) for batch in qpos_batches}
                if len(feature_dims) != 1:
                    raise RuntimeError(
                        "all DDP ranks must use the same qpos feature dimension"
                    )
                if any(batch.shape[1] < 1 for batch in qpos_batches):
                    raise RuntimeError(
                        "central SONIC received an empty qpos sequence"
                    )
                max_frames = max(int(batch.shape[1]) for batch in qpos_batches)
                padded_qpos = []
                for batch in qpos_batches:
                    if batch.shape[1] < max_frames:
                        pad_frames = max_frames - int(batch.shape[1])
                        batch = np.pad(
                            batch,
                            ((0, 0), (0, pad_frames), (0, 0)),
                            mode="edge",
                        )
                    padded_qpos.append(batch)
                counts = [int(batch.shape[0]) for batch in qpos_batches]
                combined_qpos = np.concatenate(
                    padded_qpos,
                    axis=0,
                )
                combined_frames = [
                    int(value)
                    for payload in gathered
                    for value in payload["num_frames"]
                ]
                combined_metrics = self._score_samples(
                    combined_qpos,
                    combined_frames,
                    group_size,
                    work_dir,
                )
                if len(combined_metrics) != sum(counts):
                    raise RuntimeError(
                        "central SONIC returned "
                        f"{len(combined_metrics)} metrics for {sum(counts)} "
                        "DDP candidates"
                    )
                metrics_by_rank = []
                offset = 0
                for count in counts:
                    metrics_by_rank.append(
                        combined_metrics[offset : offset + count]
                    )
                    offset += count
                packet[0] = {
                    "error": None,
                    "metrics_by_rank": metrics_by_rank,
                }
            except Exception:
                packet[0] = {
                    "error": traceback.format_exc(),
                    "metrics_by_rank": None,
                }

        torch.distributed.broadcast_object_list(packet, src=0)
        response = packet[0]
        if response is None:
            raise RuntimeError("central SONIC broadcast returned no response")
        if response["error"] is not None:
            raise RuntimeError(
                "centralized SONIC reward failed on rank 0:\n"
                f"{response['error']}"
            )
        return response["metrics_by_rank"][rank]

    def _zero_trainable_loss(self) -> torch.Tensor:
        parameters = self.bundle.trainable_parameters()
        if not parameters:
            raise RuntimeError("Flow-DPO has no trainable generator parameters")
        # Every DDP rank must traverse the same parameter hooks even when its
        # prompt yields no reliable pair. Connecting one scalar from every
        # trainable tensor keeps a zero-signal rank collective-safe with
        # find_unused_parameters=False.
        loss = parameters[0].reshape(-1)[0] * 0.0
        for parameter in parameters[1:]:
            loss = loss + parameter.reshape(-1)[0] * 0.0
        return loss

    def _dpo_loss_from_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        return self.bundle.flow_dpo_loss_g1(
            cache["vtxt"],
            cache["ctxt"],
            cache["ctxt_len"],
            cache["winner_latents"],
            cache["loser_latents"],
            cache["lengths"],
            beta=self.dpo_beta,
            timesteps_per_pair=self.dpo_timesteps_per_pair,
        )

    @staticmethod
    def _detach_result(result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: value.detach() if torch.is_tensor(value) else value
            for name, value in result.items()
        }

    def _format_dpo_result(
        self,
        loss_out: Dict[str, Any],
        static_result: Dict[str, Any],
        *,
        replay_epoch: int,
        new_rollout: bool,
    ) -> Dict[str, Any]:
        result = dict(static_result)
        result.update(
            {
                "loss": loss_out["loss"],
                "loss_dpo": loss_out["loss"].detach(),
                "dpo_implicit_accuracy": loss_out["implicit_accuracy"],
                "dpo_model_delta": loss_out["model_delta"],
                "dpo_reference_delta": loss_out["reference_delta"],
                "dpo_preference_delta": loss_out["preference_delta"],
                "dpo_timestep_mean": loss_out["timestep_mean"],
                "dpo_timestep_std": loss_out["timestep_std"],
                "dpo_timesteps_per_pair": loss_out["timesteps_per_pair"],
                "dpo_replay_epoch": loss_out["loss"].detach().new_tensor(
                    float(replay_epoch)
                ),
                "dpo_new_rollout": loss_out["loss"].detach().new_tensor(
                    float(new_rollout)
                ),
            }
        )
        if not new_rollout:
            if "n_pooled" in result:
                result["n_pooled"] = loss_out["loss"].detach().new_tensor(0.0)
            if "n_qpos_pooled" in result:
                result["n_qpos_pooled"] = loss_out["loss"].detach().new_tensor(
                    0.0
                )
        return result

    def _replay_dpo_step(self) -> Dict[str, Any]:
        cache = self._dpo_replay_cache
        if cache is None:
            raise RuntimeError("Flow-DPO replay requested without a cached pair")
        replay_epoch = int(cache["next_replay_epoch"])
        loss_out = self._dpo_loss_from_cache(cache)
        result = self._format_dpo_result(
            loss_out,
            cache["static_result"],
            replay_epoch=replay_epoch,
            new_rollout=False,
        )
        if replay_epoch >= self.dpo_replay_epochs:
            self._dpo_replay_cache = None
        else:
            cache["next_replay_epoch"] = replay_epoch + 1
        return result

    def _write_pair_audit(
        self,
        *,
        batch: Dict[str, Any],
        latents: torch.Tensor,
        lengths: torch.Tensor,
        rewards: torch.Tensor,
        candidate_semantic: torch.Tensor,
        reference_semantic: torch.Tensor,
        root_trajectory_errors: torch.Tensor,
        pairwise_guard_costs: torch.Tensor | None,
        pairwise_guard_names: tuple[str, ...],
        selection: Dict[str, torch.Tensor],
        group_size: int,
    ) -> None:
        if self.dpo_pair_audit_dir is None:
            return
        rank = int(
            getattr(self.accelerator, "process_index", 0)
            if self.accelerator is not None
            else 0
        )
        step = self._global_step()
        winners = selection["winner_indices"].tolist()
        losers = selection["loser_indices"].tolist()
        captions = [str(value) for value in batch.get("caption", [])]
        prompt_ids = [
            str(value) for value in batch.get("prompt_id", captions)
        ]
        jsonl_path = self.dpo_pair_audit_dir / f"rank{rank:02d}.jsonl"

        records = []
        for pair_index, (winner, loser) in enumerate(zip(winners, losers)):
            prompt_index = int(winner) // group_size
            stem = (
                f"{self.dpo_round_id}_rank{rank:02d}_step{step:06d}_"
                f"pair{pair_index:02d}"
            )
            pair_path = self.dpo_pair_audit_dir / f"{stem}.npz"
            text_ctxt = batch["text_ctxt_raw"][prompt_index]
            text_vec = batch["text_vec_raw"][prompt_index]
            np.savez_compressed(
                pair_path,
                winner_motion=latents[winner].detach().cpu().numpy(),
                loser_motion=latents[loser].detach().cpu().numpy(),
                length=np.int64(lengths[winner]),
                text_vec=text_vec.detach().cpu().numpy(),
                text_ctxt=text_ctxt.detach().cpu().numpy(),
                text_ctxt_length=np.int64(
                    batch["text_ctxt_raw_length"][prompt_index]
                ),
                caption=np.asarray(
                    captions[prompt_index]
                    if prompt_index < len(captions)
                    else ""
                ),
                prompt_id=np.asarray(
                    prompt_ids[prompt_index]
                    if prompt_index < len(prompt_ids)
                    else ""
                ),
                winner_reward=np.float32(rewards[winner]),
                loser_reward=np.float32(rewards[loser]),
                reward_gap=np.float32(rewards[winner] - rewards[loser]),
                winner_tmr_distance=np.float32(candidate_semantic[winner]),
                loser_tmr_distance=np.float32(candidate_semantic[loser]),
                winner_g0_tmr_distance=np.float32(
                    reference_semantic[winner]
                ),
                loser_g0_tmr_distance=np.float32(reference_semantic[loser]),
                winner_root_traj_err_m=np.float32(
                    root_trajectory_errors[winner]
                ),
                loser_root_traj_err_m=np.float32(
                    root_trajectory_errors[loser]
                ),
                pairwise_guard_names=np.asarray(
                    pairwise_guard_names,
                    dtype=np.str_,
                ),
                winner_pairwise_guard_costs=(
                    pairwise_guard_costs[winner].detach().cpu().numpy()
                    if pairwise_guard_costs is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                loser_pairwise_guard_costs=(
                    pairwise_guard_costs[loser].detach().cpu().numpy()
                    if pairwise_guard_costs is not None
                    else np.empty((0,), dtype=np.float32)
                ),
            )
            records.append(
                {
                    "round_id": self.dpo_round_id,
                    "rank": rank,
                    "step": step,
                    "pair_path": str(pair_path),
                    "prompt_index": prompt_index,
                    "winner_index": int(winner),
                    "loser_index": int(loser),
                    "reward_gap": float(rewards[winner] - rewards[loser]),
                }
            )
        if records:
            with jsonl_path.open("a") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if (
            self.gt_anchor_update_interval > 0
            and self.gt_weight > 0
            and self._grpo_updates_since_anchor
            >= self.gt_anchor_update_interval
        ):
            self._grpo_updates_since_anchor = 0
            return self._gt_anchor_step(batch)

        if self._dpo_replay_cache is not None:
            result = self._replay_dpo_step()
            self._grpo_updates_since_anchor += 1
            return result

        self._configure_rank_local_sonic_gpu()

        vtxt = batch["text_vec_raw"]
        ctxt = list(batch["text_ctxt_raw"])
        ctxt_len = batch["text_ctxt_raw_length"]
        num_frames = [int(value) for value in batch["tgt_length"].tolist()]
        batch_size = int(vtxt.shape[0])
        group_size = self.num_samples

        vtxt_rep = vtxt.repeat_interleave(group_size, dim=0)
        ctxt_rep = [value for value in ctxt for _ in range(group_size)]
        ctxt_len_rep = ctxt_len.repeat_interleave(group_size, dim=0)
        lengths_rep = torch.tensor(
            [length for length in num_frames for _ in range(group_size)],
            dtype=torch.long,
        )

        latents, initial_noise = self.bundle.sample_motion(
            vtxt_rep,
            ctxt_rep,
            ctxt_len_rep,
            lengths_rep,
            num_steps=self.diffusion_steps,
            guidance=1.0,
            transformer=self._live_inference_transformer(),
            return_initial_noise=True,
        )
        reference_transformer = self.bundle._reference_transformer()
        if reference_transformer is None:
            raise RuntimeError("Flow-DPO requires an immutable G0 reference")
        reference_latents = self.bundle.sample_motion(
            vtxt_rep,
            ctxt_rep,
            ctxt_len_rep,
            lengths_rep,
            num_steps=self.diffusion_steps,
            guidance=1.0,
            initial_noise=initial_noise,
            transformer=reference_transformer,
        )
        qpos = self.bundle.latents_to_qpos(latents)

        rollout = tempfile.TemporaryDirectory(
            prefix=f"physflow_g1_dpo_step{self._global_step()}_",
            dir=self.rollout_dir,
        )
        try:
            metrics = self._score_samples_centralized(
                qpos,
                num_frames,
                group_size,
                rollout.name,
            )
            reward_errors = [
                str(metric.get("error"))
                for metric in metrics
                if metric.get("error")
            ]
            if reward_errors and len(reward_errors) == len(metrics):
                raise RuntimeError(
                    "All tracker-reward rollouts failed; refusing a Flow-DPO "
                    f"update. First error: {reward_errors[0]}"
                )

            rewards = []
            valid = []
            for metric in metrics:
                score = metric.get("physical_score", metric.get("score"))
                score_is_finite = score is not None and np.isfinite(float(score))
                rewards.append(-float(score) if score_is_finite else 0.0)
                valid.append(not metric.get("error") and score_is_finite)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            valid_t = torch.tensor(valid, dtype=torch.bool)
            root_trajectory_errors = torch.tensor(
                [
                    float(metric.get("root_trajectory_error_mean_m", float("inf")))
                    for metric in metrics
                ],
                dtype=torch.float32,
            )
            pairwise_guard_costs = self._pairwise_guard_costs(
                metrics,
                self.dpo_pairwise_guard_names,
            )

            captions = [str(value) for value in batch.get("caption", [])]
            if len(captions) != batch_size:
                raise ValueError("Flow-DPO requires one caption per prompt")
            captions_rep = [
                caption for caption in captions for _ in range(group_size)
            ]
            if self._tmr_reward_evaluator is not None:
                candidate_semantic = self._semantic_pair_distances(
                    captions_rep,
                    latents,
                    lengths_rep,
                )
                reference_semantic = self._semantic_pair_distances(
                    captions_rep,
                    reference_latents,
                    lengths_rep,
                )
            else:
                candidate_semantic = torch.zeros_like(rewards_t)
                reference_semantic = torch.zeros_like(rewards_t)

            selection = select_reliable_preference_pairs(
                rewards_t,
                group_size=group_size,
                min_reward_gap=self.dpo_min_reward_gap,
                valid_mask=valid_t,
                semantic_distances=(
                    candidate_semantic
                    if self.dpo_require_semantic_guard
                    else None
                ),
                reference_semantic_distances=(
                    reference_semantic
                    if self.dpo_require_semantic_guard
                    else None
                ),
                semantic_floor_margin=self.dpo_semantic_floor_margin,
                pairwise_guard_costs=(
                    pairwise_guard_costs
                ),
                pairwise_guard_margins=(
                    rewards_t.new_tensor(self.dpo_pairwise_guard_margins)
                    if self.dpo_pairwise_guard_names
                    else None
                ),
            )
            winner_indices = selection["winner_indices"]
            loser_indices = selection["loser_indices"]
            pair_count = int(winner_indices.numel())

            if pair_count:
                prompt_indices = (winner_indices // group_size).tolist()
                winner_device = winner_indices.to(latents.device)
                loser_device = loser_indices.to(latents.device)
                prompt_device = torch.tensor(
                    prompt_indices,
                    dtype=torch.long,
                    device=vtxt.device,
                )
                pair_vtxt = vtxt.index_select(0, prompt_device).detach()
                pair_ctxt = [ctxt[index].detach() for index in prompt_indices]
                pair_ctxt_len = ctxt_len.index_select(
                    0,
                    prompt_device.to(ctxt_len.device),
                ).detach()
                winner_latents = latents.index_select(
                    0, winner_device
                ).detach()
                loser_latents = latents.index_select(0, loser_device).detach()
                pair_lengths = torch.tensor(
                    [num_frames[index] for index in prompt_indices],
                    dtype=torch.long,
                )
            else:
                # A direct zero built from DDP parameters bypasses DDP.forward()
                # and leaves reducer hooks in an invalid state. Run one real,
                # identical-pair DPO forward on every zero-pair rank, then mask
                # its scalar loss to zero so all ranks execute one collective-
                # compatible trainable forward per microstep.
                pair_vtxt = vtxt[:1].detach()
                pair_ctxt = [ctxt[0].detach()]
                pair_ctxt_len = ctxt_len[:1].detach()
                winner_latents = latents[:1].detach()
                loser_latents = latents[:1].detach()
                pair_lengths = torch.tensor(
                    [num_frames[0]], dtype=torch.long
                )

            pair_cache = {
                "vtxt": pair_vtxt,
                "ctxt": pair_ctxt,
                "ctxt_len": pair_ctxt_len,
                "winner_latents": winner_latents,
                "loser_latents": loser_latents,
                "lengths": pair_lengths,
            }
            loss_out = self._dpo_loss_from_cache(pair_cache)
            if not pair_count:
                loss_out = dict(loss_out)
                loss_out["loss"] = loss_out["loss"] * 0.0
            loss = loss_out["loss"]

            self._write_pair_audit(
                batch=batch,
                latents=latents,
                lengths=lengths_rep,
                rewards=rewards_t,
                candidate_semantic=candidate_semantic,
                reference_semantic=reference_semantic,
                root_trajectory_errors=root_trajectory_errors,
                pairwise_guard_costs=pairwise_guard_costs,
                pairwise_guard_names=self.dpo_pairwise_guard_names,
                selection=selection,
                group_size=group_size,
            )
            reward_telemetry = self._reward_telemetry(metrics)
            reward_telemetry.update(
                self._paired_motion_telemetry(
                    latents,
                    reference_latents,
                    lengths_rep,
                )
            )
            selected = self._select_tracker_replay(
                metrics,
                batch_size,
                group_size,
            )
            prompt_ids = list(batch.get("prompt_id", [])) or list(
                batch.get("caption", [])
            )
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

        valid_rewards = rewards_t[valid_t]
        reward_mean = (
            valid_rewards.mean()
            if valid_rewards.numel()
            else rewards_t.new_tensor(0.0)
        )
        reward_std = (
            valid_rewards.std(unbiased=False)
            if valid_rewards.numel()
            else rewards_t.new_tensor(0.0)
        )
        reward_gaps = selection["reward_gaps"]
        if pair_count:
            winner_root = root_trajectory_errors.index_select(
                0, winner_indices
            )
            loser_root = root_trajectory_errors.index_select(0, loser_indices)
            pairwise_root_delta = (winner_root - loser_root).mean()
        else:
            pairwise_root_delta = rewards_t.new_tensor(0.0)
        pairwise_guard_deltas: Dict[str, torch.Tensor] = {}
        if pairwise_guard_costs is not None:
            for guard_index, guard_name in enumerate(
                self.dpo_pairwise_guard_names
            ):
                if pair_count:
                    winner_cost = pairwise_guard_costs.index_select(
                        0, winner_indices
                    )[:, guard_index]
                    loser_cost = pairwise_guard_costs.index_select(
                        0, loser_indices
                    )[:, guard_index]
                    delta = (winner_cost - loser_cost).mean()
                else:
                    delta = rewards_t.new_tensor(0.0)
                pairwise_guard_deltas[guard_name] = delta
        semantic_floor_pass = selection["eligible_mask"].float().mean()
        semantic_guard_mask = (
            torch.isfinite(candidate_semantic)
            & torch.isfinite(reference_semantic)
            & (
                candidate_semantic
                <= reference_semantic + self.dpo_semantic_floor_margin
            )
        )
        guard_finite_fraction = (
            torch.isfinite(pairwise_guard_costs).all(dim=1).float().mean()
            if pairwise_guard_costs is not None
            else rewards_t.new_tensor(1.0)
        )
        result: Dict[str, Any] = {
            "loss": loss,
            "loss_dpo": loss.detach(),
            "dpo_pair_count": loss.detach().new_tensor(float(pair_count)),
            "dpo_pair_accept_fraction": loss.detach().new_tensor(
                float(pair_count) / max(batch_size, 1)
            ),
            "dpo_reward_gap_mean": (
                reward_gaps.mean()
                if reward_gaps.numel()
                else rewards_t.new_tensor(0.0)
            ),
            "dpo_reward_gap_min": (
                reward_gaps.min()
                if reward_gaps.numel()
                else rewards_t.new_tensor(0.0)
            ),
            "dpo_pairwise_root_delta_m": pairwise_root_delta,
            "dpo_implicit_accuracy": loss_out["implicit_accuracy"],
            "dpo_model_delta": loss_out["model_delta"],
            "dpo_reference_delta": loss_out["reference_delta"],
            "dpo_preference_delta": loss_out["preference_delta"],
            "dpo_timestep_mean": loss_out["timestep_mean"],
            "dpo_timestep_std": loss_out["timestep_std"],
            "dpo_timesteps_per_pair": loss_out["timesteps_per_pair"],
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "semantic_floor_pass_fraction": semantic_floor_pass,
            "dpo_valid_reward_fraction": valid_t.float().mean(),
            "dpo_semantic_guard_pass_fraction": (
                semantic_guard_mask.float().mean()
            ),
            "dpo_guard_metrics_finite_fraction": guard_finite_fraction,
            "tmr_candidate_distance_mean": candidate_semantic.mean(),
            "tmr_g0_distance_mean": reference_semantic.mean(),
            "candidate_vs_g0_tmr_delta_mean": (
                reference_semantic - candidate_semantic
            ).mean(),
            "n_reward_errors": rewards_t.new_tensor(float(len(reward_errors))),
            "n_pooled": rewards_t.new_tensor(float(n_pooled)),
            "n_qpos_pooled": rewards_t.new_tensor(float(n_qpos_pooled)),
            "sonic_gpu_id": rewards_t.new_tensor(float(self.sonic_gpu_id)),
        }
        for name, value in reward_telemetry.items():
            result[name] = rewards_t.new_tensor(float(value))
        for guard_name, delta in pairwise_guard_deltas.items():
            result[
                "dpo_pairwise_guard_"
                f"{guard_name.replace('.', '_')}_delta"
            ] = delta
        dynamic_names = {
            "loss",
            "loss_dpo",
            "dpo_implicit_accuracy",
            "dpo_model_delta",
            "dpo_reference_delta",
            "dpo_preference_delta",
            "dpo_timestep_mean",
            "dpo_timestep_std",
            "dpo_timesteps_per_pair",
        }
        static_result = self._detach_result(
            {
                name: value
                for name, value in result.items()
                if name not in dynamic_names
            }
        )
        pair_cache["static_result"] = static_result
        pair_cache["next_replay_epoch"] = 2
        if self.dpo_replay_epochs > 1:
            self._dpo_replay_cache = pair_cache
        self._grpo_updates_since_anchor += 1
        return self._format_dpo_result(
            loss_out,
            static_result,
            replay_epoch=1,
            new_rollout=True,
        )
