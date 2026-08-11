"""Released-SONIC MuJoCo reward for deployment-aligned generator tuning."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from motius.models.gentrack.sonic_reward import SonicJudgeReward


class SonicMujocoJudgeReward(SonicJudgeReward):
    """Score a candidate batch with the released SONIC MuJoCo deployment.

    The existing :class:`SonicJudgeReward` uses the vectorized IsaacLab policy
    during training. This backend deliberately executes the released controller
    in the same MuJoCo stack used by the independent paper gate, then applies the
    same released SONIC/BeyondMimic tracking kernels. It is slower, but removes
    simulator-domain mismatch from a controlled reward ablation.
    """

    def __init__(
        self,
        checkpoint: str,
        trainee_checkpoint: Optional[str] = None,
        parallel_workers: int = 8,
        case_timeout_s: int = 300,
        runner: Optional[str] = None,
        **kwargs,
    ) -> None:
        if trainee_checkpoint is not None and Path(trainee_checkpoint) != Path(checkpoint):
            raise ValueError(
                "SonicMujocoJudgeReward currently supports one frozen quality "
                "controller; a distinct trainee belongs in the tracker-update path"
            )
        super().__init__(
            checkpoint=checkpoint,
            trainee_checkpoint=None,
            runner=None,
            persistent=False,
            require_runner=False,
            **kwargs,
        )
        self.parallel_workers = max(1, int(parallel_workers))
        self.case_timeout_s = max(30, int(case_timeout_s))
        configured_runner = runner or os.environ.get(
            "MOTIUS_GENTRACK_SONIC_MUJOCO_RUNNER"
        )
        if not configured_runner:
            raise FileNotFoundError(
                "SONIC MuJoCo runner is required; pass runner=... or set "
                "MOTIUS_GENTRACK_SONIC_MUJOCO_RUNNER"
            )
        self.mujoco_runner = Path(configured_runner).expanduser()
        if not self.mujoco_runner.is_absolute():
            self.mujoco_runner = self.project_root / self.mujoco_runner
        self.mujoco_runner = self.mujoco_runner.absolute()
        if not self.mujoco_runner.is_file():
            raise FileNotFoundError(f"SONIC MuJoCo runner not found: {self.mujoco_runner}")

    def _mujoco_parallel_env(self, expected_cases: int) -> Dict[str, str]:
        env = self._postprocess_env()
        physical_gpu = self._physical_gpu_token()
        try:
            gpu_index = int(physical_gpu)
        except ValueError:
            gpu_index = self.gpu_id

        # DDS domain IDs must be unique for concurrent reward arms on one host.
        # Pair launchers assign a distinct physical judge GPU to each arm.
        domain_base = 20 + (gpu_index % 8) * 24
        port_base = 18000 + (gpu_index % 8) * 64
        env.update(
            {
                "EXPECTED_CASES": str(int(expected_cases)),
                "GPU_LIST": " ".join([physical_gpu] * self.parallel_workers),
                "CASE_TIMEOUT": str(self.case_timeout_s),
                "MAX_ATTEMPTS": "2",
                "DOMAIN_BASE": str(domain_base),
                "PORT_BASE": str(port_base),
                "SONIC_PYTHON": str(self.postprocess_python),
                "SONIC_LOCAL_STAGE": "1",
            }
        )
        return env

    def _parse_mujoco_metrics(
        self,
        stems: list[str],
        cases_dir: Path,
        unified_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        rows: Dict[str, dict] = {}
        metrics_path = unified_dir / "case_metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["case_id"])] = row

        results: Dict[str, Dict[str, float]] = {}
        for stem in stems:
            row = rows.get(stem)
            case_path = cases_dir / f"{stem}.npz"
            if row is None or not case_path.is_file():
                results[stem] = {
                    "score": self.error_penalty,
                    "error": "missing MuJoCo rollout or unified metric",
                }
                continue
            try:
                with np.load(case_path, allow_pickle=False) as case:
                    reference_pos = np.asarray(case["reference_joints"], dtype=np.float32)
                    execution_pos = np.asarray(case["execution_joints"], dtype=np.float32)
                    reference_quat = np.asarray(
                        case["reference_body_quat"], dtype=np.float32
                    )
                    execution_quat = np.asarray(
                        case["execution_body_quat"], dtype=np.float32
                    )
                    fps = float(np.asarray(case["fps"]).reshape(-1)[0])
                unexpected_fall = bool(row["unexpected_fall"])
                score, reward_components = self._compute_sonic_tracking_score(
                    reference_body_pos=reference_pos,
                    execution_body_pos=execution_pos,
                    reference_body_quat=reference_quat,
                    execution_body_quat=execution_quat,
                    fps=fps,
                    completion=float(row["completion"]),
                    fall_detected=unexpected_fall,
                )
                results[stem] = {
                    "score": float(score),
                    "score_protocol": "released_sonic_mujoco_deployment_v1",
                    "reward_components": reward_components,
                    "completion": float(row["completion"]),
                    "fall_detected": unexpected_fall,
                    "max_joint_error_rad": float(row["max_joint_err_rad"]),
                    "root_trajectory_error_mean_m": float(row["root_traj_err_m"]),
                    "e_joint_rad": float(row["e_joint_rad"]),
                    "er_mpjpe_mm": float(row["er_mpjpe_mm"]),
                    "evel_mps": float(row["evel_mps"]),
                    "eacc_mps2": float(row["eacc_mps2"]),
                    "mpjpe_mm": float(row["mpjpe_mm"]),
                    "mpjve_mps": float(row["mpjve_mps"]),
                    "root_vel_err_mps": float(row["root_vel_err_mps"]),
                    "sonic_success": bool(row["success_unified"]),
                }
            except Exception as exc:  # noqa: BLE001 - preserve reward batch shape.
                results[stem] = {
                    "score": self.error_penalty,
                    "error": f"MuJoCo reward parse: {exc!r}",
                }
        return results

    def score_csv_dir(
        self,
        csv_dir: Path,
        work_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        csv_dir = Path(csv_dir)
        work_dir = Path(work_dir)
        protocol_root = work_dir / "sonic_mujoco_protocol"
        output_root = work_dir / "sonic_mujoco_reward"
        stems = sorted(path.stem for path in csv_dir.glob("*.csv"))
        try:
            stems = self._build_protocol(csv_dir, protocol_root)
            input_dir = protocol_root / "inputs" / "reward_batch" / "npz"
            env = self._mujoco_parallel_env(len(stems))
            env.update(
                {
                    "INPUT_DIR": str(input_dir),
                    "OUTPUT_ROOT": str(output_root),
                    "METHOD": "sonic_mujoco_reward",
                }
            )
            log_path = work_dir / "sonic_mujoco_reward.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(
                    ["bash", str(self.mujoco_runner)],
                    cwd=self.project_root,
                    env=env,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=(self.case_timeout_s + 60)
                    * max(
                        1,
                        (len(stems) + self.parallel_workers - 1)
                        // self.parallel_workers,
                    )
                    * 2
                    + 120,
                )
            return self._parse_mujoco_metrics(
                stems,
                output_root / "cases",
                output_root / "unified_v021",
            )
        except Exception as exc:  # noqa: BLE001 - preserve group-relative batch shape.
            return {
                stem: {
                    "score": self.error_penalty,
                    "error": f"sonic MuJoCo reward: {exc!r}",
                }
                for stem in stems
            }
