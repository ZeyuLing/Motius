"""Any2Track/OpenTrack physics-realism reward for PhysFlow G1.

This adapter gives the online PhysFlow trainer the same ``score_csv_dir``
contract as the ProtoMotions and HumanoidGPT judges. Generated G1 ``qpos`` CSVs
are packed as temporary NPZ reference motions and rolled out by the validated
MuJoCo Any2Track evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Any2TrackJudgeReward:
    def __init__(
        self,
        onnx_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        config_path: Optional[str] = None,
        input_fps: int = 30,
        error_penalty: float = 5.0,
        max_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        import json
        from motius.models.gentrack.any2track_runtime import OpenTrackRollout
        from motius.evaluation.gentrack.scoring import (
            DEFAULT_G1_SCORE_CONFIG,
            compute_g1_adversarial_score,
        )
        from motius.models.gentrack.tracker_paths import (
            ANY2TRACK_CONFIG,
            ANY2TRACK_G1_MJCF,
            ANY2TRACK_ONNX,
        )

        self.onnx_path = self._abs(
            onnx_path,
            ANY2TRACK_ONNX,
        )
        self.mjcf_path = self._abs(
            mjcf_path,
            ANY2TRACK_G1_MJCF,
        )
        self.config_path = self._abs(
            config_path,
            ANY2TRACK_CONFIG,
        )
        self.input_fps = int(input_fps)
        self.error_penalty = float(error_penalty)
        self.max_steps = max_steps
        self._compute_score = compute_g1_adversarial_score
        self._score_config = DEFAULT_G1_SCORE_CONFIG
        self._runner = OpenTrackRollout(
            self.mjcf_path,
            json.loads(self.config_path.read_text()),
            self.onnx_path,
        )

    @staticmethod
    def _abs(path: Optional[str], default: Path) -> Path:
        p = Path(path) if path else default
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    def _pack_csv(self, csv_path: Path, npz_path: Path) -> None:
        qpos = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if qpos.ndim == 1:
            qpos = qpos[None]
        np.savez(npz_path, qpos=qpos.astype(np.float32), frequency=np.float32(self.input_fps))

    def _map_metrics(self, row: Dict[str, float]) -> Dict[str, float]:
        success = bool(row.get("success", False))
        joint_err = float(row.get("max_joint_err_mean", row.get("joint_err_mean", 999.0)))
        root_err = float(row.get("root_err_mean", 999.0))
        displacement_err = float(row.get("root_err_max", root_err))
        score = float(
            self._compute_score(
                completion=1.0,
                max_joint_error_rad=joint_err,
                root_trajectory_error_mean_m=root_err,
                root_displacement_error_m=displacement_err,
                fall_detected=not success,
                config=self._score_config,
            )
        )
        return {
            "score": score,
            "completion": 1.0,
            "fall_detected": not success,
            "max_joint_error_rad": joint_err,
            "root_trajectory_error_mean_m": root_err,
            "root_displacement_error_m": displacement_err,
            "any2track_success": success,
            "any2track_paper_success": bool(row.get("paper_success", False)),
            "any2track_mpjpe_mm": float(row.get("mpjpe_mm", float("nan"))),
            "any2track_local_mpjpe_mm": float(row.get("local_mpjpe_mm", float("nan"))),
            "any2track_mpjve_mps": float(row.get("mpjve_mps", float("nan"))),
        }

    def score_csv_dir(self, csv_dir: Path, work_dir: Path) -> Dict[str, Dict[str, float]]:
        csv_dir = Path(csv_dir)
        work_dir = Path(work_dir)
        npz_dir = work_dir / "any2track_npz"
        npz_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, float]] = {}
        for csv_path in sorted(csv_dir.glob("*.csv")):
            stem = csv_path.stem
            npz_path = npz_dir / f"{stem}.npz"
            try:
                self._pack_csv(csv_path, npz_path)
                row = self._runner.evaluate_motion(npz_path, max_steps=self.max_steps)
                results[stem] = self._map_metrics(row)
            except Exception as exc:  # noqa: BLE001
                results[stem] = {
                    "score": self.error_penalty,
                    "completion": 0.0,
                    "fall_detected": True,
                    "error": f"any2track: {exc}",
                }
        return results
