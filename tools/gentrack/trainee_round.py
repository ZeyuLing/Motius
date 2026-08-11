#!/usr/bin/env python3
"""GenTrack trainee co-training round runner.

The generator streams accepted, structurally valid G1 motions
into a shared pool:
    outputs/training/gentrack_g1/tracker_motion_pool/*.motion

This runner consumes that pool in rounds. Each round:
  1. wait until the pool has enough (new) motions;
  2. snapshot the current pool into a per-round directory;
  3. train the *trainee* G1 tracker on it with ProtoMotions PPO+AMP+BeyondMimic
     (experiment physflow_g1_xy_offset.py, include_xy_offset for global
     displacement), warm-started from the previous round's checkpoint;
  4. the next round warm-starts from this round's checkpoint.

So the trainee continuously chases the generator's *evolving* motion distribution
-> generator <-> trainee online co-training. The FROZEN judge that scores the
generator is deliberately separate (unbiased reward), per the adversarial design.

Runs the heavy train_agent.py in the IsaacGym py3.8 venv; the runner itself is
dependency-light and can run under any python.

Example:
  CUDA_VISIBLE_DEVICES=0 python tools/gentrack/trainee_round.py \
      --pool-dir outputs/training/gentrack_g1/tracker_motion_pool \
      --out-root outputs/training/gentrack_g1/trainee_rounds \
      --min-motions 24 --min-new 16 --steps-per-round 1500 --max-rounds 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from motius.models.gentrack.tracker_paths import PROTOMOTIONS_ROOT

DEFAULT_EXPERIMENT = (
    PROTOMOTIONS_ROOT
    / "examples"
    / "experiments"
    / "mimic"
    / "gentrack_g1_xy_offset.py"
)
DEFAULT_WARMSTART = Path(
    os.environ.get(
        "MOTIUS_GENTRACK_TRAINEE_CHECKPOINT",
        PROJECT_ROOT
        / "outputs"
        / "training"
        / "protomotions"
        / "gentrack_g1_t0"
        / "last.ckpt",
    )
)
DEFAULT_TRACKER_PYTHON = os.environ.get(
    "MOTIUS_GENTRACK_PROTO_PYTHON",
    sys.executable,
)
NUM_STEPS_PER_EPOCH = 32  # ProtoMotions PPO rollout horizon (base_agent num_steps)


def _ckpt_epoch(path: str) -> int:
    """Best-effort read of the PPO epoch already trained in a warm-start ckpt.

    ProtoMotions' ``--training-max-steps`` is a GLOBAL cap and a warm-started run
    continues from its own epoch, so each round must target a CUMULATIVE budget
    (prev_epoch + epochs_per_round); a flat per-round value would make every
    round after the first stop immediately.
    """
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0))
    except Exception:
        return 0


def _tracker_env(output_root: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROTOMOTIONS_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MOTIUS_PROTOMOTIONS_OUTPUT_ROOT"] = str(output_root)
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("WANDB_DISABLE_SENTRY", "true")
    env.setdefault("ISAACGYM_GRAPHICS_DEVICE_ID", "-1")
    for version in range(14, 8, -1):
        root = Path(f"/opt/rh/gcc-toolset-{version}/root/usr")
        if (root / "bin").exists():
            env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
            env["CC"] = str(root / "bin" / "gcc")
            env["CXX"] = str(root / "bin" / "g++")
            if (root / "lib64").exists():
                env["LD_LIBRARY_PATH"] = f"{root / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
            break
    return env


def _pool_motions(pool_dir: Path) -> list:
    return sorted(glob.glob(str(pool_dir / "*.motion")))


def _snapshot_pool(pool_dir: Path, round_dir: Path,
                   sample: int = 0, recent_frac: float = 0.5,
                   seed: int = 0) -> Path:
    """Snapshot (a subset of) the live pool into the round dir via symlinks.

    Re-loading the FULL, ever-growing pool every round makes each round
    increasingly I/O-bound (thousands of tiny .motion files) while only a tiny
    fraction of compute goes to PPO. When ``sample>0`` and the pool is larger,
    we take a RECENCY-BIASED subset: the newest ``recent_frac`` (the current
    generator/GT frontier) plus a random draw from the rest (anti-forgetting /
    diversity). Symlinks avoid duplicating I/O.
    """
    snap = round_dir / "pool"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)
    motions = _pool_motions(pool_dir)
    if sample and len(motions) > sample:
        motions.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # newest first
        n_recent = min(len(motions), int(sample * recent_frac))
        recent = motions[:n_recent]
        rest_pool = motions[n_recent:]
        rng = random.Random(seed)
        k_rest = min(len(rest_pool), sample - n_recent)
        rest = rng.sample(rest_pool, k_rest) if k_rest > 0 else []
        motions = recent + rest
    for m in motions:
        dst = snap / Path(m).name
        try:
            os.symlink(os.path.abspath(m), dst)
        except FileExistsError:
            pass
        except Exception:
            try:
                shutil.copy2(m, dst)
            except Exception:
                pass
    return snap


def _build_train_cmd(args, snap_dir: Path, warm_ckpt: str, exp_name: str, max_steps: int) -> list:
    cmd = [
        args.tracker_python,
        str(PROTOMOTIONS_ROOT / "protomotions" / "train_agent.py"),
        "--robot-name", "g1",
        "--simulator", args.simulator,
        "--experiment-path", str(Path(args.experiment).resolve()),
        "--experiment-name", exp_name,
        "--motion-file", str(snap_dir.resolve()),
        "--checkpoint", str(warm_ckpt),
        "--num-envs", str(args.num_envs),
        "--batch-size", str(args.batch_size),
        "--training-max-steps", str(max_steps),
        "--headless", "True",
        "--skip-initial-eval",
        "--overrides", f"agent.save_last_checkpoint_every={args.save_every}",
    ]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pool-dir",
        default="outputs/training/gentrack_g1/tracker_motion_pool",
    )
    ap.add_argument(
        "--out-root",
        default="outputs/training/gentrack_g1/trainee_rounds",
    )
    ap.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT))
    ap.add_argument("--warmstart-ckpt", default=str(DEFAULT_WARMSTART))
    ap.add_argument("--tracker-python", default=DEFAULT_TRACKER_PYTHON)
    ap.add_argument("--simulator", default="isaacgym")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=512)
    # Warm-start (--checkpoint) RESETS ProtoMotions' step_count to 0 each round
    # (verified: round-N ckpts save epoch=0/step_count=0), so --training-max-steps
    # is effectively a per-round budget. 1 PPO epoch = num_envs * NUM_STEPS_PER_EPOCH
    # env steps, so the old 1500-9000 values were <1 epoch => the tracker barely
    # trained. Budget a meaningful number of epochs PER ROUND instead.
    ap.add_argument("--epochs-per-round", type=int, default=30,
                    help="PPO epochs to train each round (flat; counter resets on warm-start)")
    ap.add_argument("--steps-per-round", type=int, default=0,
                    help="explicit per-round env-step budget; overrides --epochs-per-round if >0")
    # Pool subsampling: cap per-round motion-loading cost and train on the fresh
    # frontier + a diversity sample (anti-forgetting).
    ap.add_argument("--pool-sample", type=int, default=800,
                    help="max motions loaded per round (0 = full pool)")
    ap.add_argument("--recent-frac", type=float, default=0.5,
                    help="fraction of the per-round sample taken as NEWEST motions")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--min-motions", type=int, default=24, help="pool size required for round 0")
    ap.add_argument("--min-new", type=int, default=16, help="new motions required between rounds")
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--start-round", type=int, default=0)
    # CRITICAL: ProtoMotions AUTO-RESUMES results/<experiment-name> if it exists and
    # then IGNORES all CLI overrides (--motion-file/--num-envs/--checkpoint). A fixed
    # name like physflow_online_g1_trainee_rNN collides with prior runs' result dirs,
    # silently training on stale data/configs. Tagging the experiment name per launch
    # guarantees a fresh dir so our pool + warm-start chain are actually used. Pin
    # --run-tag on relaunch only if you intend to resume the SAME results dirs.
    ap.add_argument("--run-tag", default=time.strftime("%Y%m%d_%H%M%S"),
                    help="unique tag for experiment-name to avoid stale auto-resume")
    args = ap.parse_args()

    pool_dir = (PROJECT_ROOT / args.pool_dir) if not os.path.isabs(args.pool_dir) else Path(args.pool_dir)
    out_root = (PROJECT_ROOT / args.out_root) if not os.path.isabs(args.out_root) else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tracker_output_root = out_root / "protomotions_runtime"
    tracker_output_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "trainee_rounds.jsonl"

    def logj(rec):
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[trainee-runner {rec['timestamp']}] {rec}", flush=True)

    if not Path(args.warmstart_ckpt).is_file():
        logj({"event": "fatal", "msg": f"warmstart ckpt missing: {args.warmstart_ckpt}"})
        sys.exit(1)

    warm = str(Path(args.warmstart_ckpt).resolve())
    rnd = args.start_round
    last_count = 0
    logj({"event": "start", "pool_dir": str(pool_dir), "warmstart": warm,
          "min_motions": args.min_motions, "min_new": args.min_new})

    while rnd < args.max_rounds:
        # wait for enough (new) motions
        while True:
            n = len(_pool_motions(pool_dir))
            need = args.min_motions if rnd == args.start_round else (last_count + args.min_new)
            if n >= need:
                break
            time.sleep(args.poll_sec)

        round_dir = out_root / f"r{rnd:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        snap = _snapshot_pool(pool_dir, round_dir, sample=args.pool_sample,
                              recent_frac=args.recent_frac, seed=rnd)
        n_motions = len(_pool_motions(snap))
        exp_name = f"physflow_g1coevo_{args.run_tag}_r{rnd:02d}"
        # flat per-round budget (warm-start resets the step counter each round)
        if args.steps_per_round > 0:
            max_steps = args.steps_per_round
        else:
            max_steps = args.epochs_per_round * args.num_envs * NUM_STEPS_PER_EPOCH
        cmd = _build_train_cmd(args, snap, warm, exp_name, max_steps)
        logj({"event": "round_start", "round": rnd, "n_motions": n_motions,
              "pool_total": len(_pool_motions(pool_dir)), "max_steps": max_steps,
              "warmstart": warm, "experiment_name": exp_name, "cmd": cmd})

        t0 = time.time()
        round_log = round_dir / "train_agent.log"
        with open(round_log, "w") as lf:
            ret = subprocess.run(
                cmd,
                cwd=str(PROTOMOTIONS_ROOT),
                env=_tracker_env(tracker_output_root),
                stdout=lf,
                stderr=subprocess.STDOUT,
            ).returncode
        dt = time.time() - t0

        ckpt = tracker_output_root / exp_name / "last.ckpt"
        # Guard against the silent stale-resume failure mode: if ProtoMotions
        # resumed a pre-existing output/<exp_name> it prints this warning and
        # ignores our --motion-file/--num-envs/--checkpoint. Treat as fatal.
        stale_resume = False
        try:
            with open(round_log, "r", errors="ignore") as lf:
                head = lf.read(20000)
            if "overrides provided during RESUME will be IGNORED" in head \
               or str(snap.resolve()) not in head:
                stale_resume = True
        except Exception:
            pass
        if stale_resume:
            logj({"event": "fatal", "round": rnd,
                  "msg": "ProtoMotions resumed a stale results dir and ignored CLI "
                         "(wrong pool/num_envs). experiment-name collision; aborting.",
                  "experiment_name": exp_name, "log": str(round_log)})
            sys.exit(2)
        ok = (ret == 0 and ckpt.is_file())
        logj({"event": "round_done", "round": rnd, "returncode": ret,
              "elapsed_sec": round(dt, 1), "ckpt": str(ckpt), "ckpt_exists": ckpt.is_file(),
              "log": str(round_log)})
        if ok:
            warm = str(ckpt.resolve())   # continuous co-training: warm-start next round
            last_count = n_motions
            rnd += 1
        else:
            logj({"event": "round_failed", "round": rnd,
                  "hint": f"see {round_log}; not advancing warmstart"})
            time.sleep(args.poll_sec)

    logj({"event": "finished", "rounds": rnd})


if __name__ == "__main__":
    main()
