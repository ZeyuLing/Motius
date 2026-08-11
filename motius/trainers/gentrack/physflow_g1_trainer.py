"""PhysFlowG1Trainer: online best-of-N reward-weighted SFT for the G1-native
HyMotion flow-matching generator.

Subclasses :class:`PhysFlowTrainer` and reuses ALL of its machinery
(``reward`` hot-swap, ``_score_samples``, ``_is_acceptable``,
``_motion_dynamics`` anti-freeze gate, ``_export_to_pool`` trainee co-training
pool).  Only ``train_step`` is overridden, because the HyMotion generator is
conditioned on a DUAL text embedding (CLIP-L 768 ``vtxt`` + Qwen3 4096
``ctxt``) and samples via a flow-matching ODE, whereas the generic base path
uses a single 4096 ``text_feat`` and diffusion.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List

import torch

from motius.registry import TRAINERS
from motius.trainers.gentrack.physflow_trainer import PhysFlowTrainer


@TRAINERS.register_module()
class PhysFlowG1Trainer(PhysFlowTrainer):
    """Online RAFT for the 38-d G1-native flow-matching generator."""

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        vtxt = batch["text_vec_raw"]              # (B, 1, 768)
        ctxt_list = list(batch["text_ctxt_raw"])  # list of (seq_i, 4096)
        ctxt_len = batch["text_ctxt_raw_length"]  # (B,)
        num_frames = [int(x) for x in batch["tgt_length"].tolist()]
        B = vtxt.shape[0]
        N = self.num_samples

        # expand each prompt to N candidates (prompt-major flat order)
        vtxt_rep = vtxt.repeat_interleave(N, dim=0)               # (B*N, 1, 768)
        ctxt_rep = [c for c in ctxt_list for _ in range(N)]       # len B*N
        ctxt_len_rep = ctxt_len.repeat_interleave(N, dim=0)
        lengths_rep = torch.tensor([nf for nf in num_frames for _ in range(N)],
                                   dtype=torch.long)

        # 1) sample candidates from the live policy (no grad)
        sample_noise = None
        if self.relative_to_base:
            latents, sample_noise = self.bundle.sample_motion(
                vtxt_rep, ctxt_rep, ctxt_len_rep, lengths_rep,
                num_steps=self.diffusion_steps,
                return_initial_noise=True,
            )
        else:
            latents = self.bundle.sample_motion(
                vtxt_rep, ctxt_rep, ctxt_len_rep, lengths_rep,
                num_steps=self.diffusion_steps,
            )  # normalized (B*N, Lmax, 38)
        qpos = self.bundle.latents_to_qpos(latents)  # numpy (B*N, Lmax, 36)

        # 2) score with the frozen judge tracker (lower == more trackable)
        ctx = tempfile.TemporaryDirectory(
            prefix=f"physflow_g1_step{self._global_step()}_", dir=self.rollout_dir
        )
        try:
            metrics = self._score_samples(qpos, num_frames, N, ctx.name)
            for b in range(B):
                for j in range(N):
                    flat = b * N + j
                    metrics[flat].update(self._motion_dynamics(qpos[flat], num_frames[b]))
            self._add_style_costs(
                metrics,
                qpos,
                num_frames,
                N,
                captions=list(batch.get("caption", [])),
            )
            scores = torch.tensor([m.get("score", 0.0) for m in metrics], dtype=torch.float32)

            base_metrics = None
            base_scores = None
            if self.relative_to_base:
                base_transformer = None
                if hasattr(self.bundle, "_maybe_init_anchor"):
                    self.bundle._maybe_init_anchor()
                    base_transformer = getattr(self.bundle, "_anchor_transformer", None)
                if base_transformer:
                    base_latents = self.bundle.sample_motion(
                        vtxt_rep, ctxt_rep, ctxt_len_rep, lengths_rep,
                        num_steps=self.diffusion_steps,
                        initial_noise=sample_noise,
                        transformer=base_transformer,
                    )
                    base_qpos = self.bundle.latents_to_qpos(base_latents)
                    base_metrics = self._score_samples(
                        base_qpos, num_frames, N, os.path.join(ctx.name, "base"))
                    for b in range(B):
                        for j in range(N):
                            flat = b * N + j
                            base_metrics[flat].update(
                                self._motion_dynamics(base_qpos[flat], num_frames[b]))
                    base_scores = torch.tensor(
                        [m.get("score", 0.0) for m in base_metrics],
                        dtype=torch.float32)

            # 3) per prompt: pick the SFT target + decide which candidates feed the
            #    trainee pool. Two regimes:
            #    * frontier_mode (B): decouple Q (frozen validity certifier) from T
            #      (trainee difficulty). SFT target = regret-max valid candidate
            #      (Q-valid, hardest for T) so the generator actively explores the
            #      trainee's failure frontier; the trainee pool gets EVERY frontier
            #      candidate (Q-valid AND T-struggles-but-learnable), i.e. the
            #      motions that actually teach the trainee something new.
            #    * legacy: easiest acceptable candidate, single export.
            target_latents, target_lengths = [], []
            sel_vtxt, sel_ctxt, sel_ctxt_len = [], [], []
            hard_good_flags, train_flags, soft_fallback_flags, selected_good = [], [], [], []
            best_scores, mean_scores, sel_joint_stds = [], [], []
            sel_style_costs = []
            base_best_scores, base_mean_scores = [], []
            rel_score_improvements, rel_joint_improvements = [], []
            rel_root_traj_improvements, rel_root_disp_improvements = [], []
            rel_advantages, sel_train_weights = [], []
            n_frontier, sel_t_compl, sel_valid_t_compl = [], [], []
            valid_t_compl = []
            frontier_diag = {
                "total": 0, "valid": 0, "frontier": 0,
                "reject_error": 0, "reject_q_fall": 0,
                "reject_q_low_completion": 0, "reject_trackability": 0,
                "reject_kinematic": 0,
                "t_missing": 0, "t_too_low": 0, "t_too_high": 0,
            }
            accept_diag = {
                "total": 0, "hard": 0, "relative": 0, "soft_fallback": 0,
                "reject_error": 0, "reject_fall": 0,
                "reject_low_completion": 0, "reject_score": 0,
                "reject_trackability": 0, "reject_kinematic": 0,
            }
            for b in range(B):
                g = scores[b * N:(b + 1) * N]
                metrics_b = metrics[b * N:(b + 1) * N]
                base_metrics_b = (
                    base_metrics[b * N:(b + 1) * N]
                    if base_metrics is not None else [None for _ in range(N)]
                )
                base_g = base_scores[b * N:(b + 1) * N] if base_scores is not None else None
                if self.frontier_mode:
                    for m in metrics_b:
                        frontier_diag["total"] += 1
                        reason = self._valid_reject_reason(m)
                        if reason is not None:
                            key = f"reject_{reason}"
                            frontier_diag[key] = frontier_diag.get(key, 0) + 1
                            continue
                        frontier_diag["valid"] += 1
                        _, t = self._qt(m)
                        if t is None:
                            frontier_diag["t_missing"] += 1
                            frontier_diag["frontier"] += 1
                            continue
                        ct = float(t.get("completion", 0.0))
                        valid_t_compl.append(ct)
                        if ct < self.frontier_t_low:
                            frontier_diag["t_too_low"] += 1
                        elif ct >= self.frontier_t_high:
                            frontier_diag["t_too_high"] += 1
                        else:
                            frontier_diag["frontier"] += 1
                    valid = [i for i in range(N) if self._is_valid(metrics_b[i])]
                    frontier = [i for i in range(N) if self._is_frontier(metrics_b[i])]
                    has_trainee = any(self._qt(metrics_b[i])[1] is not None for i in range(N))
                    if valid:
                        if self.sft_target == "regret" and has_trainee:
                            best_local = min(
                                valid, key=lambda i: self._trainee_completion(metrics_b[i], 1.0))
                        else:
                            best_local = min(valid, key=lambda i: float(g[i]))
                        hard_good_flags.append(1.0)
                        train_flags.append(1.0)
                    else:
                        best_local = int(torch.argmin(g).item())
                        hard_good_flags.append(0.0)
                        train_flags.append(0.0)
                    soft_fallback_flags.append(0.0)
                    for i in frontier:
                        selected_good.append((b, i))
                    n_frontier.append(len(frontier))
                    compl = self._trainee_completion(metrics_b[best_local], 1.0)
                    sel_t_compl.append(compl)
                    if valid:
                        sel_valid_t_compl.append(compl)
                else:
                    for i, m in enumerate(metrics_b):
                        accept_diag["total"] += 1
                        reason = self._acceptable_reject_reason(m)
                        if reason is None:
                            accept_diag["hard"] += 1
                            rel_reason = self._relative_reject_reason(m, base_metrics_b[i])
                            if rel_reason is None:
                                accept_diag["relative"] += 1
                            else:
                                key = f"reject_{rel_reason}"
                                accept_diag[key] = accept_diag.get(key, 0) + 1
                        else:
                            key = f"reject_{reason}"
                            accept_diag[key] = accept_diag.get(key, 0) + 1
                    acceptable = [
                        i for i in range(N)
                        if self._is_acceptable(metrics_b[i])
                        and self._relative_reject_reason(metrics_b[i], base_metrics_b[i]) is None
                    ]
                    if acceptable:
                        if self.relative_select_by_advantage and self.relative_to_base:
                            best_local = max(
                                acceptable,
                                key=lambda i: self._relative_advantage(
                                    metrics_b[i], base_metrics_b[i]
                                ) if self._relative_advantage(
                                    metrics_b[i], base_metrics_b[i]
                                ) is not None else float("-inf"),
                            )
                        else:
                            best_local = min(acceptable, key=lambda i: float(g[i]))
                        hard_good_flags.append(1.0)
                        train_flags.append(1.0)
                        soft_fallback_flags.append(0.0)
                        selected_good.append((b, best_local))
                    else:
                        if self.relative_select_by_advantage and self.relative_to_base:
                            best_local = max(
                                range(N),
                                key=lambda i: self._relative_advantage(
                                    metrics_b[i], base_metrics_b[i]
                                ) if self._relative_advantage(
                                    metrics_b[i], base_metrics_b[i]
                                ) is not None else float("-inf"),
                            )
                        else:
                            best_local = int(torch.argmin(g).item())
                        hard_good_flags.append(0.0)
                        fallback_ok = self.accept_soft_fallback
                        if fallback_ok and self.accept_soft_fallback_require_relative:
                            rel_reason = self._relative_reject_reason(
                                metrics_b[best_local], base_metrics_b[best_local]
                            )
                            if rel_reason is not None:
                                fallback_ok = False
                                key = f"reject_{rel_reason}"
                                accept_diag[key] = accept_diag.get(key, 0) + 1
                        if fallback_ok:
                            train_flags.append(1.0)
                            soft_fallback_flags.append(1.0)
                            accept_diag["soft_fallback"] += 1
                        else:
                            train_flags.append(0.0)
                            soft_fallback_flags.append(0.0)
                best_scores.append(float(g[best_local]))
                mean_scores.append(float(g.mean()))
                sel_joint_stds.append(float(metrics_b[best_local].get("joint_std", 0.0)))
                if "style_cost" in metrics_b[best_local]:
                    sel_style_costs.append(float(metrics_b[best_local]["style_cost"]))
                rel_adv = (
                    self._relative_advantage(metrics_b[best_local], base_metrics_b[best_local])
                    if base_metrics_b[best_local] is not None else None
                )
                if rel_adv is not None:
                    rel_advantages.append(float(rel_adv))
                train_weight = 1.0
                if self.relative_weight_by_advantage and rel_adv is not None:
                    train_weight = 1.0 + self.relative_advantage_weight_scale * max(
                        0.0, float(rel_adv)
                    )
                    train_weight = min(
                        self.relative_advantage_weight_max,
                        max(0.1, train_weight),
                    )
                sel_train_weights.append(float(train_weight))
                if base_g is not None:
                    base_best_scores.append(float(base_g.min()))
                    base_mean_scores.append(float(base_g.mean()))
                    bm = base_metrics_b[best_local]
                    cm = metrics_b[best_local]
                    if bm is not None:
                        rel_score_improvements.append(
                            float(bm.get("score", 0.0)) - float(cm.get("score", 0.0)))
                        rel_joint_improvements.append(
                            float(bm.get("max_joint_error_rad", 0.0))
                            - float(cm.get("max_joint_error_rad", 0.0)))
                        rel_root_traj_improvements.append(
                            float(bm.get("root_trajectory_error_mean_m", 0.0))
                            - float(cm.get("root_trajectory_error_mean_m", 0.0)))
                        rel_root_disp_improvements.append(
                            float(bm.get("root_displacement_error_m", 0.0))
                            - float(cm.get("root_displacement_error_m", 0.0)))
                flat = b * N + best_local
                target_latents.append(latents[flat])
                target_lengths.append(int(num_frames[b]))
                sel_vtxt.append(vtxt[b])
                sel_ctxt.append(ctxt_list[b])
                sel_ctxt_len.append(int(ctxt_len[b]))

            target = torch.stack(target_latents, dim=0).detach()  # (B, Lmax, 38)
            sel_vtxt_t = torch.stack(sel_vtxt, dim=0)             # (B, 1, 768)
            sel_ctxt_len_t = torch.tensor(sel_ctxt_len, dtype=torch.long)
            lengths = torch.tensor(target_lengths, dtype=torch.long)
            train_mask = torch.tensor(train_flags, dtype=torch.float32)

            n_pooled = 0
            n_qpos_pooled = 0
            pool_ids = list(batch.get("prompt_id", [])) or list(batch.get("caption", []))
            if self.tracker_pool_dir and selected_good:
                n_pooled = self._export_to_pool(
                    os.path.join(ctx.name, "proto"), selected_good, pool_ids,
                )
            if self.tracker_qpos_pool_dir and selected_good:
                n_qpos_pooled = self._export_qpos_to_pool(
                    qpos, num_frames, selected_good, pool_ids,
                )
        finally:
            if self.keep_rollouts:
                try:
                    ctx._finalizer.detach()
                except Exception:
                    pass
            else:
                ctx.cleanup()

        # 3b) GT-as-special-candidate -> trainee pool. Treat each GT clip as one
        #     extra candidate: judge-score + accept-filter it exactly like a
        #     generated sample, and export accepted GT into the SAME pool so the
        #     trainee tracker co-trains on a generator+GT mix (anti-collapse real
        #     -data anchor). GT never feeds the generator's reward SFT here -- the
        #     generator's pull toward GT is the separate ``gt_weight`` term.
        n_gt_pooled = 0
        n_gt_qpos_pooled = 0
        if (self.export_gt_to_pool
                and (self.tracker_pool_dir or self.tracker_qpos_pool_dir)
                and self.enable_reward
                and "motion" in batch and self._global_step() % self.gt_pool_freq == 0):
            gt_qpos = self.bundle.latents_to_qpos(
                self.bundle.normalize_motion(
                    batch["motion"].to(self.bundle._device()).float()
                )
            )  # (B, Lclip, 36)
            gt_ctx = tempfile.TemporaryDirectory(
                prefix=f"physflow_g1_gt_step{self._global_step()}_", dir=self.rollout_dir
            )
            try:
                gt_metrics = self._score_samples(gt_qpos, num_frames, 1, gt_ctx.name)
                gt_selected = []
                for b in range(B):
                    m = gt_metrics[b]
                    m.update(self._motion_dynamics(gt_qpos[b], num_frames[b]))
                    # GT is the anti-forgetting / hard-skill real-data anchor. For
                    # generated motions we require Q-validity (a strong tracker can
                    # execute it) to avoid teaching artifacts. Real GT, however, is
                    # already a trusted reference: if we also require Q to solve it,
                    # we filter out exactly the hard clips (runs/stairs/falls) the
                    # trainee is supposed to learn. ``kinematic`` keeps only the
                    # tracker-independent artifact gate and admits hard real GT.
                    if self.frontier_mode and self.gt_pool_accept_mode == "kinematic":
                        ok = self._is_kinematically_valid(m)
                    else:
                        ok = self._is_valid(m) if self.frontier_mode else self._is_acceptable(m)
                    if ok:
                        gt_selected.append((b, 0))
                if gt_selected:
                    base_pids = (list(batch.get("prompt_id", []))
                                 or list(batch.get("caption", [])))
                    gt_pids = [
                        f"gt_{base_pids[b] if b < len(base_pids) and base_pids[b] else f'b{b}'}"
                        for b in range(B)
                    ]
                    if self.tracker_pool_dir:
                        n_gt_pooled = self._export_to_pool(
                            os.path.join(gt_ctx.name, "proto"), gt_selected, gt_pids,
                        )
                    if self.tracker_qpos_pool_dir:
                        n_gt_qpos_pooled = self._export_qpos_to_pool(
                            gt_qpos, num_frames, gt_selected, gt_pids,
                        )
            finally:
                if self.keep_rollouts:
                    try:
                        gt_ctx._finalizer.detach()
                    except Exception:
                        pass
                else:
                    gt_ctx.cleanup()

        # 4) reward-filtered + anchored flow-matching SFT toward accepted motions,
        #    plus an optional GT supervised term (shares the same batch prompts as
        #    the reward target, so the bundle fuses both into ONE forward).
        gt_target = gt_lengths = None
        if self.gt_weight > 0 and "motion" in batch:
            gt_target = self.bundle.normalize_motion(
                batch["motion"].to(self.bundle._device()).float()
            ).detach()
            gt_lengths = batch["tgt_length"]
        out = self.bundle.sft_loss_g1(
            sel_vtxt_t, sel_ctxt, sel_ctxt_len_t, target, lengths,
            good_mask=train_mask,
            sample_weights=torch.tensor(sel_train_weights, dtype=torch.float32),
            anchor_weight=self.anchor_weight,
            gt_target=gt_target, gt_lengths=gt_lengths, gt_weight=self.gt_weight,
        )

        result: Dict[str, Any] = {"loss": out["loss"], "loss_sft": out["sft_mse"]}
        result["n_good"] = torch.tensor(float(sum(hard_good_flags)))
        result["n_train_sft"] = out.get("n_good", torch.tensor(float(sum(train_flags))))
        result["n_soft_fallback"] = torch.tensor(float(sum(soft_fallback_flags)))
        if self.tracker_pool_dir:
            result["n_pooled"] = torch.tensor(float(n_pooled))
            if self.export_gt_to_pool:
                result["n_gt_pooled"] = torch.tensor(float(n_gt_pooled))
        if self.tracker_qpos_pool_dir:
            result["n_qpos_pooled"] = torch.tensor(float(n_qpos_pooled))
            if self.export_gt_to_pool:
                result["n_gt_qpos_pooled"] = torch.tensor(float(n_gt_qpos_pooled))
        if "gt_mse" in out:
            result["loss_gt"] = out["gt_mse"]
        if "anchor_mse" in out:
            result["loss_anchor"] = out["anchor_mse"]
        result["reward_best_mean"] = torch.tensor(sum(best_scores) / max(B, 1))
        result["reward_cand_mean"] = torch.tensor(sum(mean_scores) / max(B, 1))
        result["sel_joint_std_mean"] = torch.tensor(sum(sel_joint_stds) / max(B, 1))
        if sel_style_costs:
            result["style_cost_sel_mean"] = torch.tensor(
                sum(sel_style_costs) / len(sel_style_costs)
            )
        if base_best_scores:
            result["base_reward_best_mean"] = torch.tensor(
                sum(base_best_scores) / max(len(base_best_scores), 1))
            result["base_reward_cand_mean"] = torch.tensor(
                sum(base_mean_scores) / max(len(base_mean_scores), 1))
        if rel_score_improvements:
            result["rel_score_improvement_mean"] = torch.tensor(
                sum(rel_score_improvements) / len(rel_score_improvements))
            result["rel_joint_improvement_mean"] = torch.tensor(
                sum(rel_joint_improvements) / len(rel_joint_improvements))
            result["rel_root_traj_improvement_mean"] = torch.tensor(
                sum(rel_root_traj_improvements) / len(rel_root_traj_improvements))
            result["rel_root_disp_improvement_mean"] = torch.tensor(
                sum(rel_root_disp_improvements) / len(rel_root_disp_improvements))
        if rel_advantages:
            result["rel_advantage_mean"] = torch.tensor(
                sum(rel_advantages) / len(rel_advantages))
        if sel_train_weights:
            result["rel_train_weight_mean"] = torch.tensor(
                sum(sel_train_weights) / len(sel_train_weights))
        if self.frontier_mode:
            # learnability telemetry: avg #frontier candidates exported per prompt
            # and avg trainee-completion of the SFT target (lower == generator is
            # successfully targeting the trainee's failure frontier).
            result["n_frontier_mean"] = torch.tensor(sum(n_frontier) / max(B, 1)) \
                if n_frontier else torch.tensor(0.0)
            result["sel_trainee_compl"] = torch.tensor(sum(sel_t_compl) / max(B, 1)) \
                if sel_t_compl else torch.tensor(1.0)
            result["sel_valid_trainee_compl"] = (
                torch.tensor(sum(sel_valid_t_compl) / max(len(sel_valid_t_compl), 1))
                if sel_valid_t_compl else torch.tensor(1.0)
            )
            for key, value in frontier_diag.items():
                result[f"frontier_{key}"] = torch.tensor(float(value))
            if valid_t_compl:
                result["frontier_t_valid_min"] = torch.tensor(float(min(valid_t_compl)))
                result["frontier_t_valid_mean"] = torch.tensor(
                    float(sum(valid_t_compl) / len(valid_t_compl))
                )
                result["frontier_t_valid_max"] = torch.tensor(float(max(valid_t_compl)))
        else:
            for key, value in accept_diag.items():
                result[f"accept_{key}"] = torch.tensor(float(value))
        return result
