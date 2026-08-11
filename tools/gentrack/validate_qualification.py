#!/usr/bin/env python3
"""Fail closed when a Full GenTrack qualification run has no real update signal."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


STEP_RE = re.compile(r"step \[(\d+)/(\d+)\]")
KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)="
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_training_rows(paths: Iterable[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(errors="replace").splitlines():
            match = STEP_RE.search(line)
            if match is None:
                continue
            row = {
                "step": float(match.group(1)),
                "step_total": float(match.group(2)),
            }
            row.update({key: float(value) for key, value in KV_RE.findall(line)})
            rows.append(row)
    return rows


def latest_checkpoint(root: Path) -> Path:
    candidates = []
    for path in root.glob("checkpoint-iter_*"):
        try:
            step = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if (path / "model.safetensors").is_file():
            candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(f"no generator checkpoint under {root}")
    return max(candidates)[1]


def _flatten_tensor_mapping(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten Motius' nested ``model.pt`` bundle into named tensors."""
    import torch

    tensors: dict[str, Any] = {}
    if isinstance(value, torch.Tensor):
        tensors[prefix] = value
        return tensors
    if not isinstance(value, Mapping):
        return tensors
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        tensors.update(_flatten_tensor_mapping(child, prefix=child_prefix))
    return tensors


def _model_delta_safetensors(
    initial_file: Path,
    trained_file: Path,
) -> dict[str, float | int]:
    from safetensors import safe_open

    delta_sq = 0.0
    initial_sq = 0.0
    max_abs = 0.0
    changed_tensors = 0
    compared_tensors = 0
    with safe_open(initial_file, framework="pt", device="cpu") as before, safe_open(
        trained_file, framework="pt", device="cpu"
    ) as after:
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        common = sorted(before_keys & after_keys)
        if not common:
            raise RuntimeError("generator checkpoints have no common tensors")
        for key in common:
            left = before.get_tensor(key).float()
            right = after.get_tensor(key).float()
            if left.shape != right.shape:
                raise ValueError(f"tensor shape mismatch for {key}: {left.shape} != {right.shape}")
            diff = right - left
            local_max = float(diff.abs().max()) if diff.numel() else 0.0
            if local_max > 0.0:
                changed_tensors += 1
            max_abs = max(max_abs, local_max)
            delta_sq += float(diff.double().square().sum())
            initial_sq += float(left.double().square().sum())
            compared_tensors += 1
    relative_l2 = math.sqrt(delta_sq / max(initial_sq, 1e-30))
    return {
        "checkpoint_format": "safetensors",
        "compared_tensors": compared_tensors,
        "changed_tensors": changed_tensors,
        "max_abs_delta": max_abs,
        "relative_l2_delta": relative_l2,
    }


def _model_delta_torch(
    initial_file: Path,
    trained_file: Path,
) -> dict[str, float | int]:
    import torch

    before = _flatten_tensor_mapping(
        torch.load(initial_file, map_location="cpu", weights_only=False)
    )
    after = _flatten_tensor_mapping(
        torch.load(trained_file, map_location="cpu", weights_only=False)
    )
    common = sorted(before.keys() & after.keys())
    if not common:
        raise RuntimeError("generator checkpoints have no common tensors")

    delta_sq = 0.0
    initial_sq = 0.0
    max_abs = 0.0
    changed_tensors = 0
    for key in common:
        left = before[key].float()
        right = after[key].float()
        if left.shape != right.shape:
            raise ValueError(f"tensor shape mismatch for {key}: {left.shape} != {right.shape}")
        diff = right - left
        local_max = float(diff.abs().max()) if diff.numel() else 0.0
        if local_max > 0.0:
            changed_tensors += 1
        max_abs = max(max_abs, local_max)
        delta_sq += float(diff.double().square().sum())
        initial_sq += float(left.double().square().sum())

    return {
        "checkpoint_format": "torch",
        "compared_tensors": len(common),
        "changed_tensors": changed_tensors,
        "max_abs_delta": max_abs,
        "relative_l2_delta": math.sqrt(delta_sq / max(initial_sq, 1e-30)),
    }


def model_delta(initial: Path, trained: Path) -> dict[str, float | int]:
    initial_safetensors = initial / "model.safetensors"
    trained_safetensors = trained / "model.safetensors"
    if initial_safetensors.is_file() and trained_safetensors.is_file():
        return _model_delta_safetensors(initial_safetensors, trained_safetensors)

    initial_torch = initial / "model.pt"
    trained_torch = trained / "model.pt"
    for path in (initial_torch, trained_torch):
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint has neither a matched model.safetensors pair nor {path}"
            )
    return _model_delta_torch(initial_torch, trained_torch)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("standalone", "proto", "sonic", "sonic-loop"),
        required=True,
    )
    parser.add_argument(
        "--objective-mode",
        choices=("flowgrpo", "reward-weighted-sft", "dpo"),
        default="flowgrpo",
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--initial-generator", type=Path, required=True)
    parser.add_argument("--trained-generator", type=Path)
    parser.add_argument("--expected-rounds", type=int, default=1)
    parser.add_argument("--min-new-rollouts", type=int, default=2)
    parser.add_argument("--min-reward-std", type=float, default=1e-4)
    parser.add_argument(
        "--expect-zero-execution-reward",
        action="store_true",
        help=(
            "Validate the no-execution ablation: rollout reward, reward spread, "
            "physical reward, and policy advantage must all remain numerically zero."
        ),
    )
    parser.add_argument("--max-zero-reward-abs", type=float, default=1e-12)
    parser.add_argument("--min-ratio-deviation", type=float, default=1e-8)
    parser.add_argument("--min-dpo-pairs", type=int, default=1)
    parser.add_argument("--min-dpo-reward-gap", type=float, default=0.2)
    parser.add_argument("--min-relative-model-delta", type=float, default=1e-12)
    parser.add_argument("--require-pooled", action="store_true")
    parser.add_argument("--require-tracker-update", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state_events: list[dict] = []
    loop_record: dict = {}
    if args.mode == "proto":
        if args.run_root is None:
            parser.error("--run-root is required in proto mode")
        logs = sorted(args.run_root.glob("gen/r*/gen.log"))
        if len(logs) < args.expected_rounds:
            raise RuntimeError(
                f"expected {args.expected_rounds} generator logs, found {len(logs)}"
            )
        final_generator = latest_checkpoint(
            args.run_root / "gen" / f"r{args.expected_rounds - 1}"
        )
        state_path = args.run_root / "state.jsonl"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state_events = [
            json.loads(line) for line in state_path.read_text().splitlines() if line.strip()
        ]
    elif args.mode == "sonic-loop":
        if args.run_root is None:
            parser.error("--run-root is required in sonic-loop mode")
        logs = sorted(args.run_root.glob("rounds/round_*/generator.log"))
        if len(logs) < args.expected_rounds:
            raise RuntimeError(
                f"expected {args.expected_rounds} generator logs, found {len(logs)}"
            )
        completed = args.run_root / "completed.json"
        state_path = args.run_root / "state.jsonl"
        for path in (completed, state_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        loop_record = json.loads(completed.read_text())
        final_generator = Path(loop_record["final_generator_checkpoint"])
        state_events = [
            json.loads(line) for line in state_path.read_text().splitlines() if line.strip()
        ]
    elif args.mode in {"standalone", "sonic"}:
        if args.log is None or args.trained_generator is None:
            parser.error(
                f"--log and --trained-generator are required in {args.mode} mode"
            )
        logs = [args.log]
        final_generator = args.trained_generator

    rows = parse_training_rows(logs)
    if not rows:
        raise RuntimeError("no optimizer telemetry found")
    if any(not math.isfinite(row.get("loss", 0.0)) for row in rows):
        raise RuntimeError("non-finite training loss")

    if args.objective_mode == "dpo":
        rollout_rows = [
            row for row in rows if row.get("dpo_new_rollout", 0.0) > 0.5
        ]
        replay_rows = [
            row for row in rows if row.get("dpo_replay_epoch", 0.0) > 1.5
        ]
    else:
        rollout_rows = [
            row for row in rows if row.get("grpo_new_rollout", 0.0) > 0.5
        ]
        replay_rows = [
            row for row in rows if row.get("grpo_replay_epoch", 0.0) > 1.5
        ]
    anchor_rows = [row for row in rows if row.get("update_is_gt_anchor", 0.0) > 0.5]
    reward_errors = sum(row.get("n_reward_errors", 0.0) for row in rollout_rows)
    reward_stds = [row.get("reward_std", 0.0) for row in rollout_rows]
    zero_reward_fields = (
        "reward_mean",
        "reward_std",
        "physical_reward_group_mean",
        "advantage_abs_mean",
    )
    ratio_deviations = (
        [
            row.get("grpo_ratio_abs_deviation_mean", 0.0)
            for row in replay_rows
        ]
        if args.objective_mode != "dpo"
        else []
    )
    pooled = sum(
        row.get("n_pooled", 0.0) + row.get("n_qpos_pooled", 0.0)
        for row in rollout_rows
    )

    failures = []
    zero_reward_max_abs: float | None = None
    objective_evidence: dict[str, float | int | None] = {}
    if len(rollout_rows) < args.min_new_rollouts:
        failures.append(
            f"new rollouts {len(rollout_rows)} < {args.min_new_rollouts}"
        )
    if not replay_rows:
        failures.append("no replay optimizer epochs observed")
    if reward_errors != 0:
        failures.append(f"tracker reward errors={reward_errors:g}")
    if args.expect_zero_execution_reward:
        missing_zero_fields = sorted(
            {
                field
                for row in rollout_rows
                for field in zero_reward_fields
                if field not in row
            }
        )
        if missing_zero_fields:
            failures.append(
                "zero-execution telemetry missing fields: "
                + ", ".join(missing_zero_fields)
            )
        zero_reward_max_abs = max(
            (
                abs(row[field])
                for row in rollout_rows
                for field in zero_reward_fields
                if field in row
            ),
            default=math.inf,
        )
        if zero_reward_max_abs > args.max_zero_reward_abs:
            failures.append(
                "no-execution reward signal is nonzero: "
                f"max abs={zero_reward_max_abs:.3e} > "
                f"{args.max_zero_reward_abs:.3e}"
            )
    elif not reward_stds or max(reward_stds) < args.min_reward_std:
        failures.append(f"reward_std never exceeded {args.min_reward_std:g}")
    if args.objective_mode == "flowgrpo":
        ratio_max = max(ratio_deviations, default=0.0)
        objective_evidence["ratio_abs_deviation_max"] = ratio_max
        if not ratio_deviations or ratio_max < args.min_ratio_deviation:
            failures.append(
                "behavior/current policy ratio did not move: "
                f"max |ratio-1|={ratio_max:.3e}"
            )
    elif args.objective_mode == "reward-weighted-sft":
        rwsft_rows = [
            row for row in rows if row.get("grpo_replay_epoch", 0.0) >= 1.0
        ]
        zero_control_fields = (
            "grpo_ratio_abs_deviation_mean",
            "grpo_behavior_log_prob",
            "grpo_current_log_prob",
            "grpo_timesteps_per_update",
        )
        missing_rwsft_fields = sorted(
            {
                field
                for row in rwsft_rows
                for field in zero_control_fields
                if field not in row
            }
        )
        if missing_rwsft_fields:
            failures.append(
                "reward-weighted SFT telemetry missing fields: "
                + ", ".join(missing_rwsft_fields)
            )
        rwsft_control_max_abs = max(
            (
                abs(row[field])
                for row in rwsft_rows
                for field in zero_control_fields
                if field in row
            ),
            default=math.inf,
        )
        rwsft_advantage_max = max(
            (row.get("advantage_abs_mean", 0.0) for row in rollout_rows),
            default=0.0,
        )
        rwsft_loss_max = max(
            (abs(row.get("loss_grpo", 0.0)) for row in rwsft_rows),
            default=0.0,
        )
        objective_evidence.update(
            {
                "rwsft_control_signal_max_abs": rwsft_control_max_abs,
                "rwsft_advantage_abs_mean_max": rwsft_advantage_max,
                "rwsft_loss_abs_max": rwsft_loss_max,
            }
        )
        if not rwsft_rows:
            failures.append("no reward-weighted SFT optimizer rows observed")
        if rwsft_control_max_abs > args.max_zero_reward_abs:
            failures.append(
                "reward-weighted SFT unexpectedly emitted PPO transition signal: "
                f"max abs={rwsft_control_max_abs:.3e}"
            )
        if rwsft_advantage_max <= args.min_reward_std:
            failures.append(
                "reward-weighted SFT observed no nonzero advantage weights"
            )
        if rwsft_loss_max <= 0.0:
            failures.append("reward-weighted SFT objective loss stayed zero")
    else:
        dpo_pair_counts = [
            int(row.get("dpo_pair_count", 0.0)) for row in rollout_rows
        ]
        accepted_dpo_rows = [
            row for row in rollout_rows if row.get("dpo_pair_count", 0.0) > 0.0
        ]
        dpo_reward_gaps = [
            row.get("dpo_reward_gap_min", 0.0) for row in accepted_dpo_rows
        ]
        dpo_timesteps = [
            row.get("dpo_timesteps_per_pair", 0.0) for row in accepted_dpo_rows
        ]
        dpo_loss_max = max(
            (abs(row.get("loss_dpo", 0.0)) for row in accepted_dpo_rows),
            default=0.0,
        )
        objective_evidence.update(
            {
                "dpo_pair_count_max": max(dpo_pair_counts, default=0),
                "dpo_reward_gap_min": min(dpo_reward_gaps, default=None),
                "dpo_timesteps_per_pair_min": min(dpo_timesteps, default=None),
                "dpo_loss_abs_max": dpo_loss_max,
            }
        )
        if max(dpo_pair_counts, default=0) < args.min_dpo_pairs:
            failures.append(
                "Flow-DPO accepted too few preference pairs: "
                f"max={max(dpo_pair_counts, default=0)} < {args.min_dpo_pairs}"
            )
        if (
            not dpo_reward_gaps
            or min(dpo_reward_gaps) < args.min_dpo_reward_gap
        ):
            failures.append(
                "Flow-DPO preference gap below contract: "
                f"min={min(dpo_reward_gaps, default=0.0):.4f} < "
                f"{args.min_dpo_reward_gap:.4f}"
            )
        if not dpo_timesteps or min(dpo_timesteps) < 1.0:
            failures.append("Flow-DPO used no diffusion timesteps per pair")
        if dpo_loss_max <= 0.0:
            failures.append("Flow-DPO objective loss stayed zero")
    if not anchor_rows:
        failures.append("no GT anchor update observed")
    if args.require_pooled and pooled < 1:
        failures.append("no accepted generated reference entered tracker replay")

    if args.require_tracker_update or args.mode == "sonic-loop":
        event_names = [str(row.get("event")) for row in state_events]
        required_events = (
            ("generator_done", "tracker_start", "round_done")
            if args.mode == "sonic-loop"
            else ("trainee_done", "judge_synced", "round_done")
        )
        for name in required_events:
            count = event_names.count(name)
            if count < args.expected_rounds:
                failures.append(
                    f"{name} events {count} < expected rounds {args.expected_rounds}"
                )
        if "orchestrator_done" not in event_names:
            failures.append("orchestrator_done event missing")
        if args.mode == "sonic-loop":
            final_tracker = Path(str(loop_record.get("final_tracker_checkpoint", "")))
            if not final_tracker.is_file():
                failures.append(f"final SONIC tracker checkpoint missing: {final_tracker}")
            if int(loop_record.get("final_global_step", 0)) <= 0:
                failures.append("SONIC tracker global step did not advance")
            round_records = []
            for round_index in range(args.expected_rounds):
                path = args.run_root / "rounds" / f"round_{round_index:02d}" / "round.json"
                if not path.is_file():
                    failures.append(f"missing SONIC round record: {path}")
                    continue
                round_records.append(json.loads(path.read_text()))
            if len(round_records) >= 2:
                first = round_records[0]
                second = round_records[1]
                if second.get("quality_checkpoint_before") != first.get(
                    "quality_checkpoint_after"
                ):
                    failures.append("SONIC lagged quality checkpoint clock is inconsistent")
                if second.get("trainee_checkpoint_before") != first.get(
                    "trainee_checkpoint_after"
                ):
                    failures.append("SONIC trainee checkpoint clock is inconsistent")
                if second.get("quality_checkpoint_before") == second.get(
                    "trainee_checkpoint_before"
                ):
                    failures.append("SONIC round 1 quality and trainee are not lag-separated")

    delta = model_delta(args.initial_generator.resolve(), final_generator.resolve())
    if delta["changed_tensors"] < 1:
        failures.append("generator checkpoint has no changed tensors")
    if delta["relative_l2_delta"] < args.min_relative_model_delta:
        failures.append(
            f"generator relative L2 delta {delta['relative_l2_delta']:.3e} is too small"
        )

    report = {
        "status": "failed" if failures else "qualified",
        "mode": args.mode,
        "objective_mode": args.objective_mode,
        "objective_evidence": objective_evidence,
        "logs": [str(path.resolve()) for path in logs],
        "initial_generator": str(args.initial_generator.resolve()),
        "trained_generator": str(final_generator.resolve()),
        "optimizer_rows": len(rows),
        "new_rollouts": len(rollout_rows),
        "replay_rows": len(replay_rows),
        "gt_anchor_rows": len(anchor_rows),
        "reward_error_count": reward_errors,
        "reward_expectation": (
            "zero_execution" if args.expect_zero_execution_reward else "nondegenerate"
        ),
        "zero_reward_signal_max_abs": zero_reward_max_abs,
        "reward_std_min": min(reward_stds, default=None),
        "reward_std_max": max(reward_stds, default=None),
        "ratio_abs_deviation_max": max(ratio_deviations, default=None),
        "pooled_references": pooled,
        "model_delta": delta,
        "loop_record": loop_record,
        "failures": failures,
    }
    atomic_json(args.output, report)
    if failures:
        raise SystemExit("qualification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
