"""Humanoid-GPT physics-realism reward for the PhysFlow online loop.

Drop-in alternative to ``PhysicsJudgeReward`` (the ProtoMotions ONNX judge) that
scores generated G1 motions with the **Humanoid-GPT** zero-shot tracker instead.

Why a separate process: HGPT's rollout needs jax + mujoco-mjx + onnxruntime in a
py3.11 env that cannot be imported into the HYMotion training process. So we talk to
a long-lived bundled worker (``trackers/humanoid_gpt/physflow_hgpt_judge_server.py``,
launched in HGPT's venv) over a line-based JSON protocol; the worker loads the
policy ONCE and only pays the rollout cost per step.

Contract parity with ``PhysicsJudgeReward``: exposes ``error_penalty`` and
``score_csv_dir(csv_dir, work_dir) -> {stem: metrics}`` where ``metrics`` carries
``score`` (LOWER == better/more trackable), ``completion``, ``fall_detected``,
``max_joint_error_rad`` and ``root_trajectory_error_mean_m`` -- exactly what the
trainer's best-of-N selection + anti-freeze accept gate read. The scalar score
reuses the *same* ``compute_g1_adversarial_score`` formula/config as the
ProtoMotions judge so accept thresholds (e.g. ``accept_max_score``) stay
comparable across judges.

The generator already emits ``qpos`` [T, 36] (root_xyz + quat_wxyz + 29 dof) at
30 fps -- which is exactly HGPT's npz input -- so we feed it directly and skip the
ProtoMotions CSV->.motion conversion entirely.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from motius.models.gentrack.tracker_paths import (
    HUMANOID_GPT_ONNX,
    HUMANOID_GPT_ROOT,
    HUMANOID_GPT_VENV_PYTHON,
)


class HgptJudgeReward:
    def __init__(
        self,
        onnx_path: Optional[str] = None,
        hgpt_python: Optional[str] = None,
        hgpt_root: Optional[str] = None,
        hgpt_worker: Optional[str] = None,
        input_fps: int = 30,
        freq: int = 50,
        error_penalty: float = 5.0,
        fall_length_ratio: float = 0.5,
        startup_timeout_s: float = 600.0,
        job_timeout_s: float = 600.0,
        **kwargs,
    ) -> None:
        self.hgpt_root = Path(
            hgpt_root
            or os.environ.get("MOTIUS_GENTRACK_HGPT_RUNTIME")
            or HUMANOID_GPT_ROOT
        ).expanduser().absolute()
        # The worker runs with cwd=hgpt_root, so a path relative to the repo root
        # (e.g. a project-relative config path) would
        # resolve against the wrong base. Make it absolute up front.
        _onnx = Path(onnx_path or HUMANOID_GPT_ONNX)
        if not _onnx.is_absolute():
            _onnx = (PROJECT_ROOT / _onnx).resolve()
        self.onnx_path = str(_onnx)
        # The Humanoid-GPT rollout worker owns a separate JAX runtime. Keep its
        # interpreter and worker path explicit instead of assuming a local venv.
        _hgpt_python = Path(
            os.environ.get("MOTIUS_GENTRACK_HGPT_PYTHON")
            or os.environ.get("PHYSFLOW_HGPT_PYTHON")
            or hgpt_python
            or HUMANOID_GPT_VENV_PYTHON
        )
        if not _hgpt_python.is_absolute():
            _hgpt_python = (PROJECT_ROOT / _hgpt_python).resolve()
        self.hgpt_python = str(_hgpt_python)
        _worker = Path(
            hgpt_worker
            or os.environ.get("MOTIUS_GENTRACK_HGPT_WORKER")
            or self.hgpt_root / "physflow_hgpt_judge_server.py"
        )
        if not _worker.is_absolute():
            _worker = PROJECT_ROOT / _worker
        self.hgpt_worker = _worker.expanduser().absolute()
        self.input_fps = int(input_fps)
        self.freq = int(freq)
        self.error_penalty = float(error_penalty)
        self.fall_length_ratio = float(fall_length_ratio)
        self.startup_timeout_s = float(startup_timeout_s)
        self.job_timeout_s = float(job_timeout_s)

        from motius.evaluation.gentrack.scoring import (
            DEFAULT_G1_SCORE_CONFIG,
            compute_g1_adversarial_score,
        )

        self._score_config = DEFAULT_G1_SCORE_CONFIG
        self._compute_score = compute_g1_adversarial_score
        self._proc: Optional[subprocess.Popen] = None
        self._worker_log = None

    # ----------------------------------------------------------- worker mgmt
    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.exists(self.hgpt_python):
            raise FileNotFoundError(
                f"HGPT judge python not found: {self.hgpt_python} "
                "(set MOTIUS_GENTRACK_HGPT_PYTHON)"
            )
        if not self.hgpt_worker.is_file():
            raise FileNotFoundError(
                f"HGPT judge worker not found: {self.hgpt_worker} "
                "(set MOTIUS_GENTRACK_HGPT_WORKER)"
            )
        cmd = [
            self.hgpt_python,
            str(self.hgpt_worker),
            "--load_path", self.onnx_path,
            "--freq", str(self.freq),
            "--device", "cpu",
            "--policy_type", "mlp",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.hgpt_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        # Headless workers may not expose EGL. The judge never renders.
        env.setdefault("MUJOCO_GL", "disable")
        log_path = Path(
            os.environ.get(
                "MOTIUS_GENTRACK_HGPT_LOG",
                PROJECT_ROOT
                / "outputs"
                / "training"
                / "gentrack_runtime"
                / "hgpt_judge.log",
            )
        ).expanduser().absolute()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_log = open(log_path, "w")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.hgpt_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._worker_log,
            text=True,
            bufsize=1,
        )
        # block until the model is loaded and the worker reports ready
        line = self._readline_timeout(self.startup_timeout_s)
        if not line:
            raise RuntimeError(
                f"HGPT judge worker did not become ready (see {log_path})"
            )
        resp = json.loads(line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"HGPT judge worker bad startup: {resp} (see {log_path})")

    def _readline_timeout(self, timeout_s: float) -> str:
        # Popen stdout.readline has no timeout; guard the whole job with a thread.
        import threading

        result = {"line": None}

        def _read():
            try:
                result["line"] = self._proc.stdout.readline()
            except Exception:  # noqa: BLE001
                result["line"] = None

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            return ""
        return result["line"] or ""

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        if self._worker_log is not None:
            try:
                self._worker_log.close()
            except Exception:  # noqa: BLE001
                pass

    # ----------------------------------------------------------- scoring
    def _map_metrics(self, m: Dict[str, float]) -> Dict[str, float]:
        lr = float(m["length_ratio"])
        completion = float(min(max(lr, 0.0), 1.0))
        joint = float(m.get("joint_pos_mae", 0.0))            # rad (avg over time)
        root_traj = float(m.get("root_pos_err_mm", 0.0)) / 1000.0  # mm -> m
        fall = lr < self.fall_length_ratio
        score = float(
            self._compute_score(
                completion=completion,
                max_joint_error_rad=joint,
                root_trajectory_error_mean_m=root_traj,
                root_displacement_error_m=0.0,
                fall_detected=fall,
                config=self._score_config,
            )
        )
        return {
            "score": score,
            "completion": completion,
            "fall_detected": bool(fall),
            "max_joint_error_rad": joint,
            "root_trajectory_error_mean_m": root_traj,
            "length_ratio": lr,
            "kpt_pos_mae": float(m.get("kpt_pos_mae", 0.0)),
            "root_pos_err_mm": float(m.get("root_pos_err_mm", 0.0)),
            "root_yaw_err": float(m.get("root_yaw_err", 0.0)),
        }

    def score_csv_dir(self, csv_dir, work_dir) -> Dict[str, Dict[str, float]]:
        import numpy as np

        csv_dir = Path(csv_dir)
        work_dir = Path(work_dir)
        in_dir = work_dir / "hgpt_in"
        in_dir.mkdir(parents=True, exist_ok=True)

        stems: List[str] = []
        for csv in sorted(csv_dir.glob("*.csv")):
            stem = csv.stem
            stems.append(stem)
            qpos = np.loadtxt(csv, delimiter=",", dtype=np.float32)
            if qpos.ndim == 1:
                qpos = qpos[None]
            np.savez(
                in_dir / f"{stem}.npz",
                qpos=qpos.astype(np.float32),
                frequency=np.float32(self.input_fps),
            )

        out_json = work_dir / "hgpt_metrics.json"
        try:
            self._ensure_worker()
            req = {"job_dir": str(in_dir.resolve()), "out": str(out_json.resolve())}
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            resp_line = self._readline_timeout(self.job_timeout_s)
            if not resp_line:
                raise RuntimeError("HGPT judge worker timed out / died")
            resp = json.loads(resp_line)
            if resp.get("status") != "ok":
                raise RuntimeError(f"HGPT judge worker error: {resp}")
            raw = json.loads(out_json.read_text())
        except Exception as exc:  # noqa: BLE001
            try:
                (work_dir / "hgpt_debug.json").write_text(json.dumps({
                    "stage": "worker",
                    "stems": stems,
                    "error": f"{type(exc).__name__}: {exc}",
                    "worker_log": str(self._worker_log.name) if self._worker_log else None,
                    "hgpt_python": self.hgpt_python,
                    "onnx_path": self.onnx_path,
                }, indent=2))
            except Exception:
                pass
            return {s: {"score": self.error_penalty, "error": f"hgpt: {exc}"} for s in stems}

        results: Dict[str, Dict[str, float]] = {}
        debug = {"stems": stems, "raw_keys": sorted(raw.keys()), "missing": [], "errors": {}}
        for stem in stems:
            m = raw.get(stem)
            if not m or "error" in m:
                if not m:
                    debug["missing"].append(stem)
                else:
                    debug["errors"][stem] = m.get("error", "unknown")
                results[stem] = {
                    "score": self.error_penalty,
                    "error": (m or {}).get("error", "missing"),
                }
                continue
            try:
                results[stem] = self._map_metrics(m)
            except Exception as exc:  # noqa: BLE001
                debug["errors"][stem] = f"map: {exc}; keys={sorted(m.keys())}"
                results[stem] = {
                    "score": self.error_penalty,
                    "error": debug["errors"][stem],
                }
        if debug["missing"] or debug["errors"] or os.environ.get("PHYSFLOW_HGPT_DEBUG"):
            try:
                (work_dir / "hgpt_debug.json").write_text(json.dumps(debug, indent=2))
            except Exception:
                pass
        return results
