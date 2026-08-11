"""PhysFlowTrainer: online physical-feedback tuning of a G1 generator.

Strategy (Stage 1 -- online best-of-N reward-weighted SFT):
  For each training step, with the *current* generator:
    1. sample N motions per prompt from cached text embeddings (no 8B encoder);
    2. score each with a lagged same-data judge tracker in MuJoCo;
    3. select the best (most trackable) motion per prompt;
    4. take a supervised x0 diffusion step toward the selected motions
       (optionally reward-weighted across the N candidates).

This is genuinely *online* (samples come from the live policy every step) and
*online* in both directions: the lagged judge updates between rounds, while the
current trainee receives generated references selected from its relative-hard
frontier. The paper-facing implementation uses Flow-GRPO; the best-of-N trainer
in this file remains available only for ablations.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import torch

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class PhysFlowTrainer(BaseTrainer):
    """Online best-of-N reward-weighted SFT for PhysFlow."""

    def __init__(
        self,
        bundle,
        num_samples: int = 4,
        diffusion_steps: int = 30,
        cfg_weight: Optional[List[float]] = None,
        cfg_type: Optional[str] = None,
        reward_weighted: bool = False,
        reward_temperature: float = 0.5,
        judge_onnx: Optional[str] = None,
        judge_mjcf: Optional[str] = None,
        proto_use_unified_tracking_score: bool = False,
        # ---- judge backend: "protomotions" (frozen ONNX+MuJoCo), "hgpt"
        #      (Humanoid-GPT zero-shot tracker, scored in its own venv worker),
        #      or "any2track" (OpenTrack/Any2Track MuJoCo ONNX evaluator). ----
        judge_backend: str = "protomotions",
        hgpt_python: Optional[str] = None,
        hgpt_freq: int = 50,
        hgpt_input_fps: int = 30,
        any2track_config: Optional[str] = None,
        any2track_input_fps: int = 30,
        any2track_max_steps: Optional[int] = None,
        sonic_gpu_id: int = 1,
        sonic_num_envs: int = 16,
        sonic_eval_timeout_s: int = 1800,
        sonic_service_startup_timeout_s: int = 5400,
        sonic_trainee_checkpoint: Optional[str] = None,
        sonic_persistent_reward: bool = True,
        sonic_postprocess_python: Optional[str] = None,
        sonic_runner: Optional[str] = None,
        sonic_materializer: Optional[str] = None,
        sonic_evaluator: Optional[str] = None,
        sonic_mujoco_workers: int = 8,
        sonic_mujoco_case_timeout_s: int = 300,
        sonic_mujoco_runner: Optional[str] = None,
        rollout_dir: Optional[str] = None,
        keep_rollouts: bool = False,
        enable_reward: bool = True,
        # ---- anti-collapse (RAFT/ReST) controls ----
        accept_min_completion: float = 0.9,
        accept_require_no_fall: bool = True,
        accept_max_score: Optional[float] = None,
        # Optional qpos-level robot-style cost.  When enabled, candidate score is
        # ``physical_score + style_reward_weight * style_cost``.  Lower remains
        # better, and the existing accept/selection logic is unchanged.
        style_reward_bank: Optional[str] = None,
        style_reward_weight: float = 0.0,
        # Optional direct trackability caps. ``accept_max_score`` is useful as a
        # blended scalar, but heldout validation can still regress on the exact
        # metrics we care about. These caps let experiments require the SFT
        # target itself to satisfy the final tracker-quality definition.
        accept_max_joint_error_rad: Optional[float] = None,
        accept_max_root_trajectory_error_mean_m: Optional[float] = None,
        accept_max_root_displacement_error_m: Optional[float] = None,
        accept_require_root_metrics: bool = False,
        # Keep the hard accept gate as telemetry/final-quality accounting, but
        # optionally continue SFT toward the best candidate when no candidate
        # clears that gate. This avoids zero-gradient reward training at startup.
        accept_soft_fallback: bool = False,
        accept_soft_fallback_require_relative: bool = False,
        # Optional relative-to-base filter. A candidate can be absolutely
        # trackable yet still not improve over the frozen base generator. When
        # enabled, reward-SFT targets must beat the same-noise frozen-base sample
        # by explicit margins before being treated as useful improvement signal.
        relative_to_base: bool = False,
        relative_min_score_improvement: float = 0.0,
        relative_min_joint_error_improvement: float = 0.0,
        relative_min_root_trajectory_improvement: float = 0.0,
        relative_min_root_displacement_improvement: float = 0.0,
        relative_max_completion_drop: float = 0.02,
        relative_require_no_fall_regression: bool = True,
        relative_mode: str = "all_margins",
        relative_min_advantage: float = 0.0,
        relative_score_weight: float = 1.0,
        relative_joint_weight: float = 1.0,
        relative_root_trajectory_weight: float = 1.0,
        relative_root_displacement_weight: float = 0.25,
        relative_completion_weight: float = 1.0,
        relative_fall_weight: float = 2.0,
        relative_select_by_advantage: bool = False,
        relative_weight_by_advantage: bool = False,
        relative_advantage_weight_scale: float = 1.0,
        relative_advantage_weight_max: float = 3.0,
        anchor_weight: float = 0.5,
        # ---- GT mixing: supervised FM term toward real motion (anti-collapse,
        #      cold-start signal when no candidate passes the accept gate). The
        #      reward term keeps implicit weight 1.0; GT is a persistent
        #      stabiliser at this (lower) weight so it never dominates the
        #      self-generated trackable target once candidates start passing. ----
        gt_weight: float = 0.0,
        # ---- anti-freeze: reject degenerate "frozen-pose glide" candidates ----
        # A static pose is trivially trackable (no fall, completion 1.0, ~0 joint
        # error), so a pure trackability reward has a degenerate optimum at "don't
        # move the joints". We reject candidates whose articulation (temporal std
        # of joint angles over the valid window, rad) falls below this floor, and
        # optionally reject pure-translation slides (large root displacement with
        # near-frozen joints). Base HYMotion locomotion has joint_std ~0.09-0.16;
        # collapsed glides ~0.013 -- a 0.05 floor separates them cleanly.
        accept_min_joint_std: float = 0.0,
        accept_max_root_disp_if_frozen: Optional[float] = None,
        accept_frozen_joint_std: float = 0.03,
        # ---- LEARNABILITY-FRONTIER co-evolution (the real tracker-improvement
        #      mechanism). The old loop exported the EASIEST-to-track motions to
        #      the trainee pool, so the trainee only ever saw motions it already
        #      solved -> zero learning signal. Here we DECOUPLE two judges:
        #        Q (``quality_judge``, the strong FROZEN reference) certifies a
        #          motion is physically valid / a competent robot CAN track it;
        #        T (``trainee_judge``, the policy being improved) measures the
        #          motion's CURRENT difficulty.
        #      We export to the trainee pool only motions that are Q-valid AND on
        #      T's learnability frontier (T struggles but does not catastrophically
        #      fail), and (optionally) pull the generator's SFT target toward the
        #      same regret-maximising frontier so it actively produces valid-but-
        #      hard motions. This is the minimax co-evolution that can make the
        #      tracker genuinely better. Off by default (preserves Stage-1). ----
        frontier_mode: bool = False,
        quality_judge: str = "frozen",
        trainee_judge: str = "trainee",
        frontier_t_low: float = 0.2,
        frontier_t_high: float = 0.9,
        frontier_selection: str = "completion_band",
        frontier_topk_per_prompt: int = 1,
        sft_target: str = "easiest",  # "easiest" | "regret"
        # Tracker replay must not be upper-bounded by the lagged reward judge.
        # ``quality_valid`` preserves the original Q-success gate for ablations;
        # ``on_policy_all`` exports samples after only transport and reference-
        # integrity checks. ``kinematic_all`` is a compatibility alias for runs
        # created before the paper-facing name was fixed. Generator quality is
        # still optimized by Flow-GRPO, while the trainee is free to learn
        # motions that its lagged predecessor could not execute.
        tracker_replay_selection: str = "quality_valid",
        tracker_replay_samples_per_prompt: int = 0,
        tracker_replay_max_joint_vel: float = 0.0,
        tracker_replay_max_root_vel: float = 0.0,
        # kinematic validity (anti-artifact) gate, rad/s and m/s peak limits on the
        # generated reference itself (independent of any tracker). Artifact motions
        # with impossible velocities teach the trainee garbage even if a judge can
        # momentarily track them. <=0 disables.
        quality_max_joint_vel: float = 0.0,
        quality_max_root_vel: float = 0.0,
        # ---- trainee co-training: export accepted motions to a growing pool ----
        tracker_pool_dir: Optional[str] = None,
        # Optional qpos replay pool for tracker stacks whose native training data
        # is qpos NPZ rather than ProtoMotions .motion, e.g. Any2Track/HGPT.
        tracker_qpos_pool_dir: Optional[str] = None,
        tracker_qpos_pool_fps: float = 30.0,
        pool_max_motions: int = 4000,
        # ---- GT-as-special-candidate: also stream real GT motions into the
        #      trainee pool (judge-scored + accept-filtered like a generated
        #      candidate) so the tracker co-trains on a generator+GT MIX. This
        #      is the anti-collapse "real-data mixing" the co-evolution review
        #      flagged; GT does NOT optimise the generator (that is the separate
        #      ``gt_weight`` supervised term). ``gt_pool_freq`` throttles cost:
        #      score GT every Nth step (judge rollout is the bottleneck). ----
        export_gt_to_pool: bool = False,
        gt_pool_freq: int = 1,
        # Real GT is a trusted reference source, so for tracker replay we must
        # not require the frozen tracker Q to already solve it; otherwise the very
        # hard clips we want the trainee to learn are filtered out. Generated
        # motions still use Q-valid frontier gating. Modes:
        #   "valid"      legacy: Q-valid + kinematic gate.
        #   "kinematic"  real GT only: kinematic gate, no Q completion/fall gate.
        gt_pool_accept_mode: str = "valid",
        **kwargs,
    ) -> None:
        super().__init__(bundle)
        self.num_samples = int(num_samples)
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_weight = cfg_weight
        self.cfg_type = cfg_type
        self.reward_weighted = bool(reward_weighted)
        self.reward_temperature = float(reward_temperature)
        self.judge_onnx = judge_onnx
        self.judge_mjcf = judge_mjcf
        self.proto_use_unified_tracking_score = bool(
            proto_use_unified_tracking_score
        )
        self.judge_backend = str(judge_backend)
        self.hgpt_python = hgpt_python
        self.hgpt_freq = int(hgpt_freq)
        self.hgpt_input_fps = int(hgpt_input_fps)
        self.any2track_config = any2track_config
        self.any2track_input_fps = int(any2track_input_fps)
        self.any2track_max_steps = any2track_max_steps
        self.sonic_gpu_id = int(sonic_gpu_id)
        self.sonic_num_envs = int(sonic_num_envs)
        self.sonic_eval_timeout_s = int(sonic_eval_timeout_s)
        self.sonic_service_startup_timeout_s = int(sonic_service_startup_timeout_s)
        self.sonic_trainee_checkpoint = sonic_trainee_checkpoint
        self.sonic_persistent_reward = bool(sonic_persistent_reward)
        self.sonic_postprocess_python = sonic_postprocess_python
        self.sonic_runner = sonic_runner
        self.sonic_materializer = sonic_materializer
        self.sonic_evaluator = sonic_evaluator
        self.sonic_mujoco_workers = max(1, int(sonic_mujoco_workers))
        self.sonic_mujoco_case_timeout_s = max(
            30, int(sonic_mujoco_case_timeout_s)
        )
        self.sonic_mujoco_runner = sonic_mujoco_runner
        self.rollout_dir = (
            os.path.abspath(os.path.expanduser(rollout_dir))
            if rollout_dir
            else None
        )
        self.keep_rollouts = bool(keep_rollouts)
        self.enable_reward = bool(enable_reward)
        self.accept_min_completion = float(accept_min_completion)
        self.accept_require_no_fall = bool(accept_require_no_fall)
        self.accept_max_score = accept_max_score
        self.style_reward_bank = style_reward_bank
        self.style_reward_weight = float(style_reward_weight)
        self._style_bank = None
        if self.style_reward_bank and self.style_reward_weight > 0.0:
            from motius.models.gentrack.g1_style_reward import G1StyleBank

            self._style_bank = G1StyleBank.load(self.style_reward_bank)
        self.accept_max_joint_error_rad = accept_max_joint_error_rad
        self.accept_max_root_trajectory_error_mean_m = accept_max_root_trajectory_error_mean_m
        self.accept_max_root_displacement_error_m = accept_max_root_displacement_error_m
        self.accept_require_root_metrics = bool(accept_require_root_metrics)
        self.accept_soft_fallback = bool(accept_soft_fallback)
        self.accept_soft_fallback_require_relative = bool(accept_soft_fallback_require_relative)
        self.relative_to_base = bool(relative_to_base)
        self.relative_min_score_improvement = float(relative_min_score_improvement)
        self.relative_min_joint_error_improvement = float(relative_min_joint_error_improvement)
        self.relative_min_root_trajectory_improvement = float(relative_min_root_trajectory_improvement)
        self.relative_min_root_displacement_improvement = float(relative_min_root_displacement_improvement)
        self.relative_max_completion_drop = float(relative_max_completion_drop)
        self.relative_require_no_fall_regression = bool(relative_require_no_fall_regression)
        self.relative_mode = str(relative_mode)
        self.relative_min_advantage = float(relative_min_advantage)
        self.relative_score_weight = float(relative_score_weight)
        self.relative_joint_weight = float(relative_joint_weight)
        self.relative_root_trajectory_weight = float(relative_root_trajectory_weight)
        self.relative_root_displacement_weight = float(relative_root_displacement_weight)
        self.relative_completion_weight = float(relative_completion_weight)
        self.relative_fall_weight = float(relative_fall_weight)
        self.relative_select_by_advantage = bool(relative_select_by_advantage)
        self.relative_weight_by_advantage = bool(relative_weight_by_advantage)
        self.relative_advantage_weight_scale = float(relative_advantage_weight_scale)
        self.relative_advantage_weight_max = float(relative_advantage_weight_max)
        self.anchor_weight = float(anchor_weight)
        self.gt_weight = float(gt_weight)
        self.accept_min_joint_std = float(accept_min_joint_std)
        self.accept_max_root_disp_if_frozen = accept_max_root_disp_if_frozen
        self.accept_frozen_joint_std = float(accept_frozen_joint_std)
        self.frontier_mode = bool(frontier_mode)
        self.quality_judge = str(quality_judge)
        self.trainee_judge = str(trainee_judge)
        self.frontier_t_low = float(frontier_t_low)
        self.frontier_t_high = float(frontier_t_high)
        self.frontier_selection = str(frontier_selection)
        if self.frontier_selection not in {"completion_band", "relative_hard"}:
            raise ValueError(
                "frontier_selection must be 'completion_band' or 'relative_hard', "
                f"got {self.frontier_selection!r}"
            )
        self.frontier_topk_per_prompt = max(1, int(frontier_topk_per_prompt))
        self.sft_target = str(sft_target)
        self.tracker_replay_selection = str(tracker_replay_selection)
        if self.tracker_replay_selection not in {
            "quality_valid",
            "frontier",
            "on_policy_all",
            "kinematic_all",
        }:
            raise ValueError(
                "tracker_replay_selection must be 'quality_valid', 'frontier', "
                "'on_policy_all', or the deprecated 'kinematic_all' alias, "
                f"got {self.tracker_replay_selection!r}"
            )
        self.tracker_replay_samples_per_prompt = int(
            tracker_replay_samples_per_prompt
        )
        if self.tracker_replay_samples_per_prompt < 0:
            raise ValueError("tracker_replay_samples_per_prompt must be nonnegative")
        self.tracker_replay_max_joint_vel = float(tracker_replay_max_joint_vel)
        self.tracker_replay_max_root_vel = float(tracker_replay_max_root_vel)
        self.quality_max_joint_vel = float(quality_max_joint_vel)
        self.quality_max_root_vel = float(quality_max_root_vel)
        self.tracker_pool_dir = (
            os.path.abspath(os.path.expanduser(tracker_pool_dir))
            if tracker_pool_dir
            else None
        )
        self.tracker_qpos_pool_dir = (
            os.path.abspath(os.path.expanduser(tracker_qpos_pool_dir))
            if tracker_qpos_pool_dir
            else None
        )
        self.tracker_qpos_pool_fps = float(tracker_qpos_pool_fps)
        self.pool_max_motions = int(pool_max_motions)
        self.export_gt_to_pool = bool(export_gt_to_pool)
        self.gt_pool_freq = max(1, int(gt_pool_freq))
        self.gt_pool_accept_mode = str(gt_pool_accept_mode)
        self._reward = None
        if self.rollout_dir:
            os.makedirs(self.rollout_dir, exist_ok=True)
        if self.tracker_pool_dir:
            os.makedirs(self.tracker_pool_dir, exist_ok=True)
        if self.tracker_qpos_pool_dir:
            os.makedirs(self.tracker_qpos_pool_dir, exist_ok=True)

    @staticmethod
    def _safe_pool_token(value: Any) -> str:
        text = str(value) if value is not None else ""
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
        return text[:120] or "item"

    def _export_to_pool(self, proto_dir: str, selected: List[tuple], prompt_ids: List[str]) -> int:
        """Copy accepted (trackable) ``.motion`` files into the shared tracker
        pool so the trainee (ProtoMotions PPO+AMP) can co-train on the live
        generator distribution. ``selected`` is a list of (b, best_local) for
        prompts whose best candidate was acceptable. Returns #exported."""
        import glob
        import shutil

        if not self.tracker_pool_dir:
            return 0
        step = self._global_step()
        n = 0
        for b, best_local in selected:
            stem = f"p{b:03d}_s{best_local:02d}"
            srcs = glob.glob(os.path.join(proto_dir, f"{stem}*.motion"))
            if not srcs:
                continue
            pid = prompt_ids[b] if b < len(prompt_ids) and prompt_ids[b] else f"b{b}"
            dst = os.path.join(self.tracker_pool_dir, f"it{step:06d}_{pid}_{stem}.motion")
            try:
                shutil.copy2(srcs[0], dst)
                n += 1
            except Exception:
                pass
        # cap pool size (keep most recent by mtime)
        try:
            allm = sorted(
                glob.glob(os.path.join(self.tracker_pool_dir, "*.motion")),
                key=lambda p: os.path.getmtime(p),
            )
            for old in allm[: max(0, len(allm) - self.pool_max_motions)]:
                os.remove(old)
        except Exception:
            pass
        return n

    def _export_qpos_to_pool(
        self,
        qpos: Any,
        lengths: List[int],
        selected: List[tuple],
        prompt_ids: List[str],
    ) -> int:
        """Save accepted qpos references as NPZ for non-Proto tracker trainers."""
        if not self.tracker_qpos_pool_dir:
            return 0
        import glob
        import numpy as np

        os.makedirs(self.tracker_qpos_pool_dir, exist_ok=True)
        step = self._global_step()
        arr = np.asarray(qpos)
        n = 0
        for b, best_local in selected:
            flat = b * self.num_samples + best_local
            if flat < 0 or flat >= arr.shape[0]:
                continue
            length = int(lengths[b]) if b < len(lengths) else arr.shape[1]
            sample = np.asarray(arr[flat])[:length].astype(np.float32)
            if sample.ndim != 2 or sample.shape[1] < 7:
                continue
            stem = f"p{b:03d}_s{best_local:02d}"
            pid = self._safe_pool_token(
                prompt_ids[b] if b < len(prompt_ids) and prompt_ids[b] else f"b{b}"
            )
            dst = os.path.join(self.tracker_qpos_pool_dir, f"it{step:06d}_{pid}_{stem}.npz")
            try:
                np.savez(dst, qpos=sample, frequency=np.float32(self.tracker_qpos_pool_fps))
                n += 1
            except Exception:
                pass
        try:
            allm = sorted(
                glob.glob(os.path.join(self.tracker_qpos_pool_dir, "*.npz")),
                key=lambda p: os.path.getmtime(p),
            )
            for old in allm[: max(0, len(allm) - self.pool_max_motions)]:
                os.remove(old)
        except Exception:
            pass
        return n

    @staticmethod
    def _motion_dynamics(qpos: "np.ndarray", length: int) -> Dict[str, float]:
        """Per-candidate articulation/translation stats from generated qpos.

        ``qpos`` is [T, 36] (root pos[:3] + root quat[3:7] + 29 joints[7:]).
        ``joint_std`` is the mean over joints of the temporal std of joint angles
        over the valid window -- a direct measure of how much the body actually
        moves. ``root_disp`` is the start->end root translation (m). A frozen-pose
        glide has tiny ``joint_std`` with large ``root_disp``.
        """
        import numpy as np
        a = np.asarray(qpos)[: max(int(length), 1)]
        if a.ndim == 1:
            a = a[None]
        joints = a[:, 7:] if a.shape[1] > 7 else a
        joint_std = float(np.std(joints, axis=0).mean()) if a.shape[0] > 1 else 0.0
        root_disp = float(np.linalg.norm(a[-1, :3] - a[0, :3])) if a.shape[1] >= 3 else 0.0
        # peak kinematic rates (per-frame finite diff at fps=30) -- artifact
        # detector independent of any tracker: impossible joint/root velocities
        # mark a reference as unusable training data even if a judge tracks it.
        fps = 30.0
        if a.shape[0] > 1:
            jvel = np.abs(np.diff(joints, axis=0)) * fps
            max_joint_vel = float(jvel.max()) if jvel.size else 0.0
            rvel = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=-1) * fps
            max_root_vel = float(rvel.max()) if rvel.size else 0.0
        else:
            max_joint_vel = 0.0
            max_root_vel = 0.0
        return {"joint_std": joint_std, "root_disp": root_disp,
                "max_joint_vel": max_joint_vel, "max_root_vel": max_root_vel}

    def _add_style_costs(
        self,
        metrics: List[Dict[str, float]],
        qpos: "np.ndarray",
        num_frames: List[int],
        group_size: int,
        captions: Optional[List[str]] = None,
    ) -> None:
        if self._style_bank is None or self.style_reward_weight <= 0.0:
            return
        from motius.models.gentrack.g1_style_reward import categorize_style_text

        for flat_idx, record in enumerate(metrics):
            b = flat_idx // group_size
            label_source = captions[b] if captions and b < len(captions) else ""
            category = categorize_style_text(label_source)
            cost = self._style_bank.style_cost(
                qpos[flat_idx],
                length=int(num_frames[b]),
                category=category,
            )
            record["physical_score"] = float(record.get("score", 0.0))
            record["style_cost"] = float(cost)
            record["style_category"] = category
            record["score"] = float(record["physical_score"] + self.style_reward_weight * cost)

    # ----------------------------------------------- Q/T frontier helpers (B)
    def _qt(self, m: Dict[str, float]):
        """Split a candidate's metrics into Q (strong frozen quality certifier)
        and T (current trainee difficulty) sub-metrics using the per-judge
        breakdown the ensemble reward writes. Falls back to the combined metrics
        for Q when no per-judge breakdown exists (single-judge / round 0), and
        returns ``t=None`` when the trainee judge is not in the ensemble yet."""
        pj = m.get("per_judge") or {}
        q = pj.get(self.quality_judge)
        if q is None:
            q = {"completion": m.get("completion"), "fall_detected": m.get("fall_detected"),
                 "score": m.get("score")}
        t = pj.get(self.trainee_judge)
        return q, t

    def _is_kinematically_valid(self, m: Dict[str, float]) -> bool:
        """Tracker-independent artifact gate: reject references with impossible
        joint/root velocities or a degenerate frozen pose."""
        js = m.get("joint_std")
        if js is not None and self.accept_min_joint_std > 0.0 and float(js) < self.accept_min_joint_std:
            return False
        if (self.accept_max_root_disp_if_frozen is not None and js is not None
                and float(js) < self.accept_frozen_joint_std
                and float(m.get("root_disp", 0.0)) > self.accept_max_root_disp_if_frozen):
            return False
        if self.quality_max_joint_vel > 0.0 and float(m.get("max_joint_vel", 0.0)) > self.quality_max_joint_vel:
            return False
        if self.quality_max_root_vel > 0.0 and float(m.get("max_root_vel", 0.0)) > self.quality_max_root_vel:
            return False
        return True

    def _passes_trackability_caps(self, m: Dict[str, float]) -> bool:
        return self._trackability_reject_reason(m) is None

    def _trackability_reject_reason(self, m: Dict[str, float]) -> Optional[str]:
        if self.accept_require_root_metrics and not (
            "root_trajectory_error_mean_m" in m or "root_displacement_error_m" in m
        ):
            return "track_missing_root"
        if (
            self.accept_max_joint_error_rad is not None
            and float(m.get("max_joint_error_rad", 1e9)) > float(self.accept_max_joint_error_rad)
        ):
            return "track_joint_error"
        if (
            self.accept_max_root_trajectory_error_mean_m is not None
            and float(m.get("root_trajectory_error_mean_m", 1e9))
            > float(self.accept_max_root_trajectory_error_mean_m)
        ):
            return "track_root_trajectory"
        if (
            self.accept_max_root_displacement_error_m is not None
            and float(m.get("root_displacement_error_m", 1e9))
            > float(self.accept_max_root_displacement_error_m)
        ):
            return "track_root_displacement"
        return None

    def _is_valid(self, m: Dict[str, float]) -> bool:
        """A motion is a VALID reference iff the strong frozen judge Q can execute
        it (physical-validity / correctness certificate) AND it passes the
        tracker-independent kinematic gate. Difficulty for the trainee is NOT a
        validity criterion here -- that is the whole point of decoupling Q and T."""
        if not self.enable_reward:
            return True
        if "error" in m:
            return False
        q, _ = self._qt(m)
        if self.accept_require_no_fall and bool(q.get("fall_detected", True)):
            return False
        if float(q.get("completion", 0.0)) < self.accept_min_completion:
            return False
        if not self._passes_trackability_caps(m):
            return False
        return self._is_kinematically_valid(m)

    def _valid_reject_reason(self, m: Dict[str, float]) -> Optional[str]:
        """Return the first Q/kinematic gate that rejects a generated candidate.

        This is telemetry-only; it mirrors ``_is_valid`` so generator frontier
        yield can be debugged without changing accept/reject behavior.
        """
        if not self.enable_reward:
            return None
        if "error" in m:
            return "error"
        q, _ = self._qt(m)
        if self.accept_require_no_fall and bool(q.get("fall_detected", True)):
            return "q_fall"
        if float(q.get("completion", 0.0)) < self.accept_min_completion:
            return "q_low_completion"
        track_reason = self._trackability_reject_reason(m)
        if track_reason is not None:
            return track_reason
        if not self._is_kinematically_valid(m):
            return "kinematic"
        return None

    def _is_frontier(self, m: Dict[str, float]) -> bool:
        """Learnability frontier for the trainee pool: Q-valid AND the current
        trainee T struggles (completion in (low, high)) -- not already solved,
        not catastrophically unlearnable. When T is absent (round 0) nothing is
        on the frontier yet, so we fall back to exporting valid motions."""
        if not self._is_valid(m):
            return False
        _, t = self._qt(m)
        if t is None:
            return True  # no trainee yet: any valid motion seeds the pool
        ct = float(t.get("completion", 0.0))
        return (self.frontier_t_low <= ct < self.frontier_t_high)

    def _select_frontier(
        self,
        metrics: List[Dict[str, float]],
        batch_size: int,
        group_size: int,
    ) -> List[tuple[int, int]]:
        """Select quality-valid references that are hard for the current tracker.

        ``completion_band`` preserves the original absolute completion rule.
        ``relative_hard`` avoids treating a fully played rollout as solved: for
        each prompt it keeps the highest-cost trainee rollouts among candidates
        certified by the lagged quality judge. The only absolute trainee gate is
        the lower completion bound, which removes catastrophic/non-learning
        rollouts; difficulty itself is defined by within-prompt ranking.
        """
        if self.frontier_selection == "completion_band":
            return [
                (prompt_index, sample_index)
                for prompt_index in range(batch_size)
                for sample_index in range(group_size)
                if self._is_frontier(metrics[prompt_index * group_size + sample_index])
            ]

        selected: List[tuple[int, int]] = []
        for prompt_index in range(batch_size):
            candidates = []
            no_trainee = []
            for sample_index in range(group_size):
                metric = metrics[prompt_index * group_size + sample_index]
                if not self._is_valid(metric):
                    continue
                _, trainee = self._qt(metric)
                if trainee is None:
                    no_trainee.append((prompt_index, sample_index))
                    continue
                completion = float(trainee.get("completion", 0.0))
                if completion < self.frontier_t_low:
                    continue
                # Larger physical score means poorer tracking and thus a harder
                # reference. Fall/contact penalties are already part of score.
                hardness = float(trainee.get("score", 0.0))
                candidates.append((hardness, sample_index))

            if candidates:
                candidates.sort(key=lambda item: (-item[0], item[1]))
                selected.extend(
                    (prompt_index, sample_index)
                    for _, sample_index in candidates[: self.frontier_topk_per_prompt]
                )
            elif no_trainee:
                # Round 0 seeds the replay pool before a distinct trainee exists.
                selected.extend(no_trainee[: self.frontier_topk_per_prompt])
        return selected

    def _select_tracker_replay(
        self,
        metrics: List[Dict[str, float]],
        batch_size: int,
        group_size: int,
    ) -> List[tuple[int, int]]:
        """Select references exported to the adapting tracker.

        The paper-facing ``on_policy_all`` contract deliberately does not ask a
        lagged tracker to certify success.  Such a gate would make the adapting
        tracker's training support a subset of what its predecessor can already
        execute.  We only reject scorer failures and malformed/degenerate
        references here; the lagged tracker remains the generator's soft reward.
        """
        if self.tracker_replay_selection == "frontier":
            return self._select_frontier(metrics, batch_size, group_size)

        selected: List[tuple[int, int]] = []
        per_prompt = self.tracker_replay_samples_per_prompt or group_size
        for prompt_index in range(batch_size):
            prompt_selected: List[tuple[int, int]] = []
            for sample_index in range(group_size):
                metric = metrics[prompt_index * group_size + sample_index]
                if self.tracker_replay_selection == "quality_valid":
                    keep = self._is_valid(metric)
                else:
                    keep = self._is_replay_integrity_valid(metric)
                if keep:
                    prompt_selected.append((prompt_index, sample_index))
            selected.extend(prompt_selected[:per_prompt])
        return selected

    def _is_replay_integrity_valid(self, metric: Dict[str, float]) -> bool:
        """Reject only malformed generated references before tracker training.

        Motion-amplitude, fall, and lagged-tracker success are intentionally not
        replay gates.  Optional velocity ceilings are disabled by default and
        exist only for a separately reported corruption-filter ablation.
        """
        if "error" in metric:
            return False
        for key in ("joint_std", "root_disp", "max_joint_vel", "max_root_vel"):
            value = metric.get(key)
            if value is not None and not math.isfinite(float(value)):
                return False
        if (
            self.tracker_replay_max_joint_vel > 0.0
            and float(metric.get("max_joint_vel", 0.0))
            > self.tracker_replay_max_joint_vel
        ):
            return False
        if (
            self.tracker_replay_max_root_vel > 0.0
            and float(metric.get("max_root_vel", 0.0))
            > self.tracker_replay_max_root_vel
        ):
            return False
        return True

    def _trainee_completion(self, m: Dict[str, float], default: float = 1.0) -> float:
        """Current trainee completion for a candidate (higher == trainee already
        tracks it). Returns ``default`` when the trainee judge is absent."""
        _, t = self._qt(m)
        if t is None:
            return float(default)
        return float(t.get("completion", default))

    def _is_acceptable(self, m: Dict[str, float]) -> bool:
        """A candidate is an acceptable SFT target only if the robot can actually
        execute it AND it is a non-degenerate motion: no fall + sufficient
        completion (+ optional score ceiling) + enough articulation (anti-freeze).

        Rejecting fallen/failed motions stops collapse onto *untrackable* modes;
        rejecting frozen-pose glides stops collapse onto the opposite degenerate
        mode -- a static pose that is trivially trackable but is not the motion the
        prompt asked for (legs frozen while the root slides across the floor)."""
        return self._acceptable_reject_reason(m) is None

    def _acceptable_reject_reason(self, m: Dict[str, float]) -> Optional[str]:
        """Return the first hard-accept gate that rejects a candidate."""
        if not self.enable_reward:
            return None
        if "error" in m:
            return "error"
        if self.accept_require_no_fall and bool(m.get("fall_detected", True)):
            return "fall"
        if float(m.get("completion", 0.0)) < self.accept_min_completion:
            return "low_completion"
        if self.accept_max_score is not None and float(m.get("score", 1e9)) > self.accept_max_score:
            return "score"
        track_reason = self._trackability_reject_reason(m)
        if track_reason is not None:
            return track_reason
        if not self._is_kinematically_valid(m):
            return "kinematic"
        return None

    def _relative_reject_reason(
        self,
        cand: Dict[str, float],
        base: Optional[Dict[str, float]],
    ) -> Optional[str]:
        """Return why ``cand`` is not a useful improvement over same-noise base."""
        if not self.relative_to_base:
            return None
        if base is None:
            return "relative_missing_base"
        if "error" in base:
            return None
        if (
            self.relative_require_no_fall_regression
            and not bool(base.get("fall_detected", False))
            and bool(cand.get("fall_detected", False))
        ):
            return "relative_fall_regression"
        if (
            float(cand.get("completion", 0.0))
            + self.relative_max_completion_drop
            < float(base.get("completion", 0.0))
        ):
            return "relative_completion_drop"
        if self.relative_mode in {"advantage", "net_advantage", "weighted_advantage"}:
            adv = self._relative_advantage(cand, base)
            if adv is None:
                return None
            if adv < self.relative_min_advantage:
                return "relative_advantage_margin"
            return None
        if (
            float(base.get("score", 1e9)) - float(cand.get("score", 1e9))
            < self.relative_min_score_improvement
        ):
            return "relative_score_margin"
        if (
            float(base.get("max_joint_error_rad", 1e9))
            - float(cand.get("max_joint_error_rad", 1e9))
            < self.relative_min_joint_error_improvement
        ):
            return "relative_joint_margin"
        if (
            float(base.get("root_trajectory_error_mean_m", 1e9))
            - float(cand.get("root_trajectory_error_mean_m", 1e9))
            < self.relative_min_root_trajectory_improvement
        ):
            return "relative_root_trajectory_margin"
        if (
            float(base.get("root_displacement_error_m", 1e9))
            - float(cand.get("root_displacement_error_m", 1e9))
            < self.relative_min_root_displacement_improvement
        ):
            return "relative_root_displacement_margin"
        return None

    def _relative_advantage(
        self,
        cand: Dict[str, float],
        base: Optional[Dict[str, float]],
    ) -> Optional[float]:
        """Weighted same-noise improvement score; positive means cand is better."""
        if base is None or "error" in base:
            return None
        score_gain = float(base.get("score", 0.0)) - float(cand.get("score", 0.0))
        joint_gain = (
            float(base.get("max_joint_error_rad", 0.0))
            - float(cand.get("max_joint_error_rad", 0.0))
        )
        root_gain = (
            float(base.get("root_trajectory_error_mean_m", 0.0))
            - float(cand.get("root_trajectory_error_mean_m", 0.0))
        )
        root_disp_gain = (
            float(base.get("root_displacement_error_m", 0.0))
            - float(cand.get("root_displacement_error_m", 0.0))
        )
        completion_gain = float(cand.get("completion", 0.0)) - float(base.get("completion", 0.0))
        fall_gain = float(bool(base.get("fall_detected", False))) - float(
            bool(cand.get("fall_detected", False))
        )
        return (
            self.relative_score_weight * score_gain
            + self.relative_joint_weight * joint_gain
            + self.relative_root_trajectory_weight * root_gain
            + self.relative_root_displacement_weight * root_disp_gain
            + self.relative_completion_weight * completion_gain
            + self.relative_fall_weight * fall_gain
        )

    # ----------------------------------------------------------- reward (lazy)
    @property
    def reward(self):
        # NOTE: this trainer is an nn.Module, whose __getattr__ MASKS any
        # AttributeError raised *inside* this property getter into a misleading
        # "object has no attribute 'reward'". Re-raise such internal failures as
        # RuntimeError so the true cause (e.g. a failed judge import/build) is
        # visible in the traceback instead of being swallowed.
        try:
            return self._build_reward()
        except AttributeError as e:
            import traceback
            raise RuntimeError(
                "PhysFlowTrainer.reward getter failed (real cause below):\n"
                + traceback.format_exc()
            ) from e

    def _build_reward(self):
        # The co-evolution orchestrator hot-swaps the judge between outer rounds
        # by writing a JSON judge spec and pointing PHYSFLOW_JUDGE_SPEC at it
        # (frozen / latest-trainee / blended). We rebuild the reward whenever the
        # spec file changes so a relaunched-per-round generator always scores
        # against the *current* judge ensemble. Falls back to the single frozen
        # judge_onnx when no spec is set (the original Stage-1 behaviour).
        # Humanoid-GPT judge runs in its own venv worker (jax/mujoco-mjx); it does
        # not use the ProtoMotions ONNX-ensemble spec mechanism.
        if self.judge_backend == "hgpt":
            from motius.models.gentrack.hgpt_reward import HgptJudgeReward

            if self._reward is None:
                self._reward = HgptJudgeReward(
                    onnx_path=self.judge_onnx,
                    hgpt_python=self.hgpt_python,
                    freq=self.hgpt_freq,
                    input_fps=self.hgpt_input_fps,
                )
            return self._reward

        if self.judge_backend in {"any2track", "opentrack"}:
            from motius.models.gentrack.any2track_reward import Any2TrackJudgeReward

            if self._reward is None:
                self._reward = Any2TrackJudgeReward(
                    onnx_path=self.judge_onnx,
                    mjcf_path=self.judge_mjcf,
                    config_path=self.any2track_config,
                    input_fps=self.any2track_input_fps,
                    max_steps=self.any2track_max_steps,
                )
            return self._reward

        if self.judge_backend == "sonic":
            from motius.models.gentrack.sonic_reward import SonicJudgeReward

            if self._reward is None:
                self._reward = SonicJudgeReward(
                    checkpoint=self.judge_onnx,
                    trainee_checkpoint=self.sonic_trainee_checkpoint,
                    gpu_id=self.sonic_gpu_id,
                    num_envs=self.sonic_num_envs,
                    input_fps=self.tracker_qpos_pool_fps,
                    eval_timeout_s=self.sonic_eval_timeout_s,
                    persistent=self.sonic_persistent_reward,
                    service_startup_timeout_s=self.sonic_service_startup_timeout_s,
                    postprocess_python=self.sonic_postprocess_python,
                    runner=self.sonic_runner,
                    materializer=self.sonic_materializer,
                    evaluator=self.sonic_evaluator,
                )
            return self._reward

        if self.judge_backend == "sonic_mujoco":
            from motius.models.gentrack.sonic_mujoco_reward import (
                SonicMujocoJudgeReward,
            )

            if self._reward is None:
                self._reward = SonicMujocoJudgeReward(
                    checkpoint=self.judge_onnx,
                    trainee_checkpoint=self.sonic_trainee_checkpoint,
                    gpu_id=self.sonic_gpu_id,
                    input_fps=self.tracker_qpos_pool_fps,
                    eval_timeout_s=self.sonic_eval_timeout_s,
                    postprocess_python=self.sonic_postprocess_python,
                    parallel_workers=self.sonic_mujoco_workers,
                    case_timeout_s=self.sonic_mujoco_case_timeout_s,
                    runner=self.sonic_mujoco_runner,
                )
            return self._reward

        from motius.models.gentrack.reward import PhysicsJudgeReward

        spec = os.environ.get("PHYSFLOW_JUDGE_SPEC")
        if spec and os.path.isfile(spec):
            sig = (spec, os.path.getmtime(spec))
            if self._reward is None or getattr(self, "_judge_sig", None) != sig:
                self._reward = PhysicsJudgeReward.from_spec_file(
                    spec,
                    mjcf_path=self.judge_mjcf,
                    use_unified_tracking_score=(
                        self.proto_use_unified_tracking_score
                    ),
                )
                self._judge_sig = sig
            return self._reward
        if self._reward is None:
            self._reward = PhysicsJudgeReward(
                onnx_path=self.judge_onnx,
                mjcf_path=self.judge_mjcf,
                use_unified_tracking_score=(
                    self.proto_use_unified_tracking_score
                ),
            )
        return self._reward

    # ----------------------------------------------------------------- helpers
    def _global_step(self) -> int:
        try:
            return int(self.get_global_step())
        except Exception:
            return 0

    def _score_samples(
        self, qpos: "torch.Tensor", num_frames: List[int], group_size: int, work_dir: str
    ) -> List[Dict[str, float]]:
        """Write per-sample CSVs (trimmed to length) and score them. Returns a
        list of metric dicts aligned with the flat [B*N] sample order."""
        import numpy as np

        csv_dir = os.path.join(work_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        stems = []
        for flat_idx in range(qpos.shape[0]):
            b = flat_idx // group_size
            length = int(num_frames[b])
            stem = f"p{b:03d}_s{flat_idx % group_size:02d}"
            stems.append(stem)
            sample = np.asarray(qpos[flat_idx])[:length]
            self.bundle.save_qpos_csv(sample, os.path.join(csv_dir, f"{stem}.csv"))

        if not self.enable_reward:
            return [{"score": 0.0} for _ in stems]

        scored = self.reward.score_csv_dir(csv_dir, work_dir)
        return [scored.get(stem, {"score": self.reward.error_penalty}) for stem in stems]

    # -------------------------------------------------------------- train step
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        text_feat = batch["text_feat"]            # [B, seq, 4096]
        text_pad_mask = batch["text_pad_mask"]    # [B, seq]
        num_frames = list(batch["num_frames"])    # [B]
        B = text_feat.shape[0]
        N = self.num_samples

        # Expand each prompt to N candidates: flat order = prompt-major.
        feat_rep = text_feat.repeat_interleave(N, dim=0)          # [B*N, seq, 4096]
        mask_rep = text_pad_mask.repeat_interleave(N, dim=0)      # [B*N, seq]
        lengths_rep = torch.tensor(
            [nf for nf in num_frames for _ in range(N)], dtype=torch.long
        )

        # 1) sample candidate motions from the live policy (no grad)
        latents = self.bundle.sample_latents(
            feat_rep, mask_rep, lengths_rep,
            diffusion_steps=self.diffusion_steps,
            cfg_weight=self.cfg_weight, cfg_type=self.cfg_type,
        )  # [B*N, Tmax, D]
        qpos = self.bundle.latents_to_qpos(latents)  # numpy [B*N, Tmax, 36]

        # 2) score with the frozen judge tracker (lower == more trackable)
        ctx = tempfile.TemporaryDirectory(
            prefix=f"physflow_step{self._global_step()}_", dir=self.rollout_dir
        )
        try:
            metrics = self._score_samples(qpos, num_frames, N, ctx.name)
            # attach articulation/translation stats so the accept filter can
            # reject degenerate frozen-pose glides (anti-freeze gate).
            for b in range(B):
                for j in range(N):
                    flat = b * N + j
                    metrics[flat].update(self._motion_dynamics(qpos[flat], num_frames[b]))
            scores = torch.tensor([m.get("score", 0.0) for m in metrics], dtype=torch.float32)

            # 3) per prompt: prefer the best *acceptable* candidate; mark whether
            #    any candidate was acceptable (good_mask) so unacceptable prompts
            #    contribute zero SFT gradient (only the anchor regularizes them).
            target_latents = []
            target_lengths = []
            sel_text_feat = []
            sel_text_mask = []
            good_flags = []
            selected_good = []   # (b, best_local) for accepted prompts -> pool
            best_scores, mean_scores, sel_joint_stds = [], [], []
            for b in range(B):
                g = scores[b * N:(b + 1) * N]
                metrics_b = metrics[b * N:(b + 1) * N]
                acceptable = [i for i in range(N) if self._is_acceptable(metrics_b[i])]
                if acceptable:
                    best_local = min(acceptable, key=lambda i: float(g[i]))
                    good_flags.append(1.0)
                    selected_good.append((b, best_local))
                else:
                    best_local = int(torch.argmin(g).item())
                    good_flags.append(0.0)
                best_scores.append(float(g[best_local]))
                mean_scores.append(float(g.mean()))
                sel_joint_stds.append(float(metrics_b[best_local].get("joint_std", 0.0)))
                flat = b * N + best_local
                target_latents.append(latents[flat])
                target_lengths.append(int(num_frames[b]))
                sel_text_feat.append(text_feat[b])
                sel_text_mask.append(text_pad_mask[b])

            target = torch.stack(target_latents, dim=0).detach()      # [B, Tmax, D]
            sel_feat = torch.stack(sel_text_feat, dim=0)              # [B, seq, 4096]
            sel_mask = torch.stack(sel_text_mask, dim=0)
            lengths = torch.tensor(target_lengths, dtype=torch.long)
            good_mask = torch.tensor(good_flags, dtype=torch.float32)

            # export accepted motions to the shared trainee pool (closed loop)
            n_pooled = 0
            n_qpos_pooled = 0
            if self.tracker_pool_dir and selected_good:
                n_pooled = self._export_to_pool(
                    os.path.join(ctx.name, "proto"),
                    selected_good,
                    list(batch.get("prompt_id", [])),
                )
            if self.tracker_qpos_pool_dir and selected_good:
                n_qpos_pooled = self._export_qpos_to_pool(
                    qpos, num_frames, selected_good, list(batch.get("prompt_id", [])),
                )
        finally:
            if self.keep_rollouts:
                try:
                    ctx._finalizer.detach()
                except Exception:
                    pass
            else:
                ctx.cleanup()

        # 4) reward-filtered + anchored x0 step toward the accepted motions
        out = self.bundle.sft_loss(
            sel_feat, sel_mask, target, lengths,
            good_mask=good_mask, anchor_weight=self.anchor_weight,
        )

        result: Dict[str, Any] = {"loss": out["loss"]}
        result["loss_sft"] = out["sft_mse"]
        result["n_good"] = out.get("n_good", torch.tensor(float(sum(good_flags))))
        if self.tracker_pool_dir:
            result["n_pooled"] = torch.tensor(float(n_pooled))
        if self.tracker_qpos_pool_dir:
            result["n_qpos_pooled"] = torch.tensor(float(n_qpos_pooled))
        if "anchor_mse" in out:
            result["loss_anchor"] = out["anchor_mse"]
        result["reward_best_mean"] = torch.tensor(sum(best_scores) / max(B, 1))
        result["reward_cand_mean"] = torch.tensor(sum(mean_scores) / max(B, 1))
        # articulation telemetry: mean joint_std of the SELECTED targets. If this
        # trends toward ~0 the policy is collapsing into frozen-pose glides.
        result["sel_joint_std_mean"] = torch.tensor(sum(sel_joint_stds) / max(B, 1))
        return result
