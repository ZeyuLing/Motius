"""Lagged-judge physics reward for the PhysFlow online co-training loop.

The generator is scored by how well a one-round-lagged G1 tracker can physically
execute the generated motion in MuJoCo. The current trainee has zero reward
weight and is used only to identify relative-hard references for tracker replay;
the lag prevents the generator and trainee from exploiting one another within a
single update.

It reuses the already-validated scoring stack:
  qpos CSV  --convert-->  ProtoMotions .motion  --MuJoCo+ONNX-->  stats
  stats  -->  compute_g1_adversarial_score   (lower == more trackable == better)

All of this runs in the HYMotion environment (MuJoCo + onnxruntime), so the whole
online loop lives in a single process -- no IsaacGym required for the reward.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from motius.models.gentrack.tracker_paths import PROTOMOTIONS_ROOT


_INPROCESS_CONVERTER = None
_INPROCESS_CONVERTER_LOCK = threading.Lock()


def _default_convert_python() -> str:
    """Resolve a Python that contains the ProtoMotions converter stack."""
    configured = os.environ.get("PHYSFLOW_CONVERT_PYTHON")
    configured = os.environ.get("MOTIUS_GENTRACK_PROTO_PYTHON") or configured
    if configured:
        return configured

    candidates = (Path(os.sys.executable),)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return str(candidates[-1])


def _convert_pythonpath(env: Dict[str, str]) -> str:
    """Build a py3.8-safe path even when the parent is the T2M py3.10 shim."""
    configured = env.get("PHYSFLOW_PY38_PYTHONPATH", "")
    inherited = configured or env.get("PYTHONPATH", "")
    compatible = []
    for entry in inherited.split(os.pathsep):
        if not entry:
            continue
        normalized = entry.replace("\\", "/")
        if "/python3.10/site-packages" in normalized:
            continue
        compatible.append(entry)
    ordered = [str(PROTOMOTIONS_ROOT), str(PROJECT_ROOT), *compatible]
    return os.pathsep.join(dict.fromkeys(ordered))


def _simulate_motion_worker(payload: Dict[str, object], result_queue) -> None:
    """Run one MuJoCo rollout in a killable child process.

    This is only used when ``PHYSFLOW_PROTO_SIM_SUBPROCESS`` is enabled. The
    direct in-process path stays the default for existing evaluators.
    """
    try:
        from motius.models.gentrack.protomotions_runtime import (
            parse_body_mesh_mapping,
            simulate_and_export,
        )

        mjcf_path = str(payload["mjcf_path"])
        stats = simulate_and_export(
            onnx_path=str(payload["onnx_path"]),
            motion_file=str(payload["motion_file"]),
            output_json_path=str(payload["output_json_path"]),
            mjcf_path=mjcf_path,
            body_mesh_mapping=parse_body_mesh_mapping(Path(mjcf_path)),
            subsample_factor=int(payload["subsample_factor"]),
            terminate_on_unexpected_fall=False,
            export_frames=False,
            collect_tracking_trajectories=bool(
                payload.get("collect_tracking_trajectories", False)
            ),
        )
        result_queue.put({"status": "ok", "stats": stats})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


class PhysicsJudgeReward:
    """Score generated G1 motions with a frozen MuJoCo + ONNX tracker.

    Args:
        onnx_path: Frozen judge tracker (defaults to released ``g1-bones-deploy``).
        mjcf_path: G1 MuJoCo model.
        input_fps / output_fps: CSV / motion fps (HYMotion-G1 is 30).
        error_penalty: score assigned when a motion fails to convert/simulate.
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        input_fps: int = 30,
        output_fps: int = 30,
        robot_json_subsample: int = 4,
        error_penalty: float = 5.0,
        convert_python: Optional[str] = None,
        judges: Optional[List[Dict[str, object]]] = None,
        use_unified_tracking_score: bool = False,
    ) -> None:
        from motius.models.gentrack.protomotions_runtime import DEFAULT_MJCF, DEFAULT_ONNX

        # ---- judge ensemble -------------------------------------------------
        # ``judges`` is an optional list of {"onnx", "mjcf"?, "weight"?, "name"?}.
        # When given, every generated motion is rolled out under EACH judge and
        # the per-judge scores are combined (weighted mean); acceptance gating
        # (no-fall + completion) is taken CONSERVATIVELY across the ensemble so a
        # motion only counts as trackable if *all* judges can execute it. This is
        # what lets the online-adversarial loop swap / blend the frozen released
        # tracker with the co-trained trainee (see physflow_coevolve_orchestrator).
        # With no ``judges`` (the default) behaviour is identical to a single
        # frozen judge at ``onnx_path``.
        if judges:
            self._judges = []
            for j in judges:
                self._judges.append({
                    "onnx": str(j["onnx"]),
                    "mjcf": str(j.get("mjcf") or mjcf_path or DEFAULT_MJCF),
                    "weight": float(j.get("weight", 1.0)),
                    "name": str(j.get("name", os.path.basename(os.path.dirname(str(j["onnx"]))) or "judge")),
                })
        else:
            self._judges = [{
                "onnx": str(onnx_path or DEFAULT_ONNX),
                "mjcf": str(mjcf_path or DEFAULT_MJCF),
                "weight": 1.0,
                "name": "frozen",
            }]
        # primary judge (used for frame-count / control_dt lookup)
        self.onnx_path = self._judges[0]["onnx"]
        self.mjcf_path = self._judges[0]["mjcf"]
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.robot_json_subsample = robot_json_subsample
        self.error_penalty = float(error_penalty)
        self.use_unified_tracking_score = bool(use_unified_tracking_score)
        # The CSV->.motion converter imports dm_control + protomotions, which
        # can live in a simulator-owned environment separate from HYMotion.
        self.convert_python = convert_python or _default_convert_python()
        self.convert_timeout_s = float(os.environ.get("PHYSFLOW_CONVERT_TIMEOUT", "600"))
        self.simulate_timeout_s = float(os.environ.get("PHYSFLOW_PROTO_SIM_TIMEOUT", "0"))
        self.simulate_subprocess = _env_flag(
            "PHYSFLOW_PROTO_SIM_SUBPROCESS",
            default=self.simulate_timeout_s > 0,
        )
        self.simulate_mp_context = os.environ.get("PHYSFLOW_PROTO_MP_CONTEXT", "fork")

        from motius.evaluation.gentrack.scoring import (
            DEFAULT_G1_SCORE_CONFIG,
            DEFAULT_SONIC_TRACKING_SCORE_CONFIG,
            compute_g1_adversarial_score,
            compute_sonic_tracking_score,
        )
        from motius.models.gentrack.protomotions_runtime import (
            parse_body_mesh_mapping,
            simulate_and_export,
        )

        self._score_config = DEFAULT_G1_SCORE_CONFIG
        self._compute_score = compute_g1_adversarial_score
        self._unified_score_config = DEFAULT_SONIC_TRACKING_SCORE_CONFIG
        self._compute_unified_score = compute_sonic_tracking_score
        self._simulate = simulate_and_export
        self._parse_body_mesh_mapping = parse_body_mesh_mapping
        # cache body<->mesh mapping per (unique) mjcf so multi-judge ensembles
        # that share the G1 model don't re-parse the XML for every judge.
        self._mesh_cache: Dict[str, object] = {}
        self._body_mesh_mapping = self._mesh_mapping_for(self.mjcf_path)

    def _mesh_mapping_for(self, mjcf_path: str):
        if mjcf_path not in self._mesh_cache:
            self._mesh_cache[mjcf_path] = self._parse_body_mesh_mapping(Path(mjcf_path))
        return self._mesh_cache[mjcf_path]

    @classmethod
    def from_spec_file(cls, spec_path: str, **kwargs) -> "PhysicsJudgeReward":
        """Build a reward from a JSON judge spec written by the co-evolution
        orchestrator: ``{"judges": [{"onnx": ..., "weight": ...}, ...]}``."""
        import json

        with open(spec_path) as f:
            spec = json.load(f)
        return cls(judges=spec.get("judges"), **kwargs)

    # --- CSV dir -> .motion dir (one subprocess for the whole batch) ---
    def _convert_csv_dir(self, csv_dir: Path, proto_dir: Path) -> None:
        if _env_flag("PHYSFLOW_PROTO_CONVERT_IN_PROCESS", default=False):
            self._convert_csv_dir_in_process(csv_dir, proto_dir)
            return

        cmd = [
            self.convert_python,
            "data/scripts/convert_g1_csv_to_proto.py",
            "--input-dir", str(csv_dir.resolve()),
            "--output-dir", str(proto_dir.resolve()),
            "--input-fps", str(self.input_fps),
            "--output-fps", str(self.output_fps),
            "--pos-units", "m",
            "--rot-format", "quat_wxyz",
            "--joint-units", "rad",
            "--no-has-header",
            "--no-has-frame-column",
            "--force-remake",
        ]
        env = _os_environ()
        env["MUJOCO_GL"] = "disable"
        py38_ld = env.get("PHYSFLOW_PY38_LD_LIBRARY_PATH")
        if py38_ld:
            current_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = py38_ld + ((os.pathsep + current_ld) if current_ld else "")
        # The convert script imports the ``protomotions`` package, so the
        # subprocess needs PROTOMOTIONS_ROOT on PYTHONPATH.
        env["PYTHONPATH"] = _convert_pythonpath(env)
        log_path = proto_dir.parent / "convert.log"
        with open(log_path, "w") as lf:
            try:
                subprocess.run(
                    cmd,
                    cwd=str(PROTOMOTIONS_ROOT),
                    env=env,
                    check=True,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=self.convert_timeout_s if self.convert_timeout_s > 0 else None,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                lf.flush()
                try:
                    lines = log_path.read_text(errors="replace").splitlines()
                    log_tail = "\n".join(lines[-80:])
                except OSError as read_exc:
                    log_tail = f"<failed to read {log_path}: {read_exc}>"
                raise RuntimeError(
                    f"Proto conversion failed via {self.convert_python}: {exc}\n"
                    f"convert log ({log_path}) tail:\n{log_tail}"
                ) from exc

    def _convert_csv_dir_in_process(self, csv_dir: Path, proto_dir: Path) -> None:
        """Reuse the official converter without a cold Python start per step.

        Importing the ProtoMotions converter from network storage can dominate
        short jobs. The conversion itself is unchanged; this path loads the
        same module once and serializes its cwd-sensitive invocation.
        """
        global _INPROCESS_CONVERTER

        import contextlib
        import importlib.util
        import sys

        # The official converter changes cwd to the ProtoMotions checkout.
        # Resolve caller-owned rollout paths first so that logs, CSV inputs, and
        # converted motions stay in the experiment directory.
        csv_dir = Path(csv_dir).resolve()
        proto_dir = Path(proto_dir).resolve()
        log_path = (proto_dir.parent / "convert.log").resolve()

        converter_path = (
            Path(PROTOMOTIONS_ROOT) / "data" / "scripts" / "convert_g1_csv_to_proto.py"
        )
        converter_script_dir = str(converter_path.parent)
        proto_site = os.environ.get(
            "MOTIUS_GENTRACK_PROTO_SITE_PACKAGES",
            os.environ.get("PHYSFLOW_PROTO_SITE_PACKAGES", ""),
        )

        with _INPROCESS_CONVERTER_LOCK:
            if _INPROCESS_CONVERTER is None:
                for entry in (str(PROTOMOTIONS_ROOT), converter_script_dir, proto_site):
                    if entry and entry not in sys.path:
                        # Append so the generator runtime keeps ownership of
                        # torch/numpy while ProtoMotions supplies missing deps.
                        sys.path.append(entry)
                spec = importlib.util.spec_from_file_location(
                    "physflow_inprocess_g1_converter", converter_path
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot import Proto converter: {converter_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _INPROCESS_CONVERTER = module

            previous_cwd = Path.cwd()
            try:
                os.chdir(PROTOMOTIONS_ROOT)
                with open(log_path, "w") as log_file, contextlib.redirect_stdout(
                    log_file
                ), contextlib.redirect_stderr(log_file):
                    with _INPROCESS_CONVERTER.torch.no_grad():
                        _INPROCESS_CONVERTER.main(
                            input_dir=csv_dir,
                            output_dir=proto_dir,
                            input_fps=self.input_fps,
                            output_fps=self.output_fps,
                            force_remake=True,
                            ignore_first_n_frames=0,
                            apply_motion_filter=False,
                            min_height_threshold=-0.05,
                            max_velocity_threshold=15.0,
                            max_dof_vel_threshold=40.0,
                            duration_height_filter=0.1,
                            duration_height_seconds=0.6,
                            robot_type="g1",
                            contact_labels_dir=None,
                            pos_units="m",
                            rot_format="quat_wxyz",
                            joint_units="rad",
                            has_header=False,
                            has_frame_column=False,
                            euler_order="xyz",
                            num_rank=1,
                            slurm_rank=0,
                        )
            except Exception as exc:
                try:
                    log_tail = "\n".join(
                        log_path.read_text(errors="replace").splitlines()[-80:]
                    )
                except OSError as read_exc:
                    log_tail = f"<failed to read {log_path}: {read_exc}>"
                raise RuntimeError(
                    f"in-process Proto conversion failed: {exc}\n"
                    f"convert log ({log_path}) tail:\n{log_tail}"
                ) from exc
            finally:
                os.chdir(previous_cwd)

    def _expected_frames(self, motion_path: Path) -> int:
        import yaml
        from deployment.motion_utils import MotionPlayer

        with open(self.onnx_path.replace(".onnx", ".yaml")) as f:
            meta = yaml.safe_load(f)
        control_dt = meta["timing"]["control_dt"]
        return int(MotionPlayer(str(motion_path), control_dt=control_dt).total_frames)

    def score_motion_file(self, motion_path: Path, out_json: Path) -> Dict[str, float]:
        """Roll out the motion under EVERY judge and combine.

        - ``score``: weighted mean of per-judge adversarial scores (lower=better).
        - ``completion``: min across judges (conservative).
        - ``fall_detected``: any across judges (conservative -> a motion is only
          'trackable' if no judge in the ensemble falls).
        Per-judge breakdown is returned under ``per_judge`` for logging.
        """
        total_frames = self._expected_frames(motion_path)
        out_json = Path(out_json)
        per_judge: Dict[str, Dict[str, float]] = {}
        wsum = 0.0
        score_acc = 0.0
        legacy_score_acc = 0.0
        reward_component_acc: Dict[str, float] = {}
        completions: List[float] = []
        falls: List[bool] = []
        joint_errs: List[float] = []
        traj_errs: List[float] = []
        disp_errs: List[float] = []
        absolute_low_roots: List[bool] = []
        root_height_deficits: List[float] = []
        for ji, j in enumerate(self._judges):
            jout = out_json if len(self._judges) == 1 else out_json.with_name(
                f"{out_json.stem}__{j['name']}{out_json.suffix}"
            )
            stats = self._simulate_judge(j, motion_path, jout)
            completion = float(stats["total_steps"] / max(total_frames, 1))
            legacy_score = float(self._compute_score(
                completion=completion,
                max_joint_error_rad=float(stats.get("max_joint_error_rad", 0.0)),
                root_trajectory_error_mean_m=float(stats.get("root_trajectory_error_mean_m", 0.0)),
                root_displacement_error_m=float(stats.get("root_displacement_error_m", 0.0)),
                fall_detected=bool(stats.get("fall_detected", False)),
                config=self._score_config,
            ))
            reward_components: Dict[str, float] = {}
            if self.use_unified_tracking_score:
                s, reward_components = self._compute_unified_score(
                    reference_body_pos=stats["reference_body_pos"],
                    execution_body_pos=stats["execution_body_pos"],
                    reference_body_quat=stats["reference_body_quat"],
                    execution_body_quat=stats["execution_body_quat"],
                    fps=float(stats["tracking_fps"]),
                    completion=completion,
                    fall_detected=bool(stats.get("fall_detected", False)),
                    config=self._unified_score_config,
                )
                s = float(s)
            else:
                s = legacy_score
            w = float(j["weight"])
            score_acc += w * s
            legacy_score_acc += w * legacy_score
            wsum += w
            for name, value in reward_components.items():
                reward_component_acc[name] = (
                    reward_component_acc.get(name, 0.0) + w * float(value)
                )
            # A zero-weight judge is diagnostic-only (the current trainee used
            # for frontier difficulty). It must not silently tighten the
            # quality/reward aggregate, otherwise the trainee becomes its own
            # validity gate and the generated frontier can be empty by design.
            if w > 0.0:
                completions.append(completion)
                falls.append(bool(stats.get("fall_detected", False)))
                joint_errs.append(float(stats.get("max_joint_error_rad", 0.0)))
                traj_errs.append(float(stats.get("root_trajectory_error_mean_m", 0.0)))
                disp_errs.append(float(stats.get("root_displacement_error_m", 0.0)))
                absolute_low_roots.append(bool(stats.get("absolute_low_root_detected", False)))
                root_height_deficits.append(float(stats.get("max_root_height_deficit_m", 0.0)))
            per_judge[j["name"]] = {
                                    "score": s,
                                    "legacy_root_aware_score": legacy_score,
                                    "completion": completion,
                                    "fall_detected": bool(stats.get("fall_detected", False)),
                                    "absolute_low_root_detected": bool(
                                        stats.get("absolute_low_root_detected", False)
                                    ),
                                    "max_root_height_deficit_m": float(
                                        stats.get("max_root_height_deficit_m", 0.0)
                                    )}
        if not completions:
            raise ValueError("judge ensemble must contain at least one positive-weight judge")
        result = {
            "score": float(score_acc / max(wsum, 1e-9)),
            "physical_score": float(score_acc / max(wsum, 1e-9)),
            "legacy_root_aware_score": float(
                legacy_score_acc / max(wsum, 1e-9)
            ),
            "score_protocol": (
                "protomotions_mujoco_public_tracking_kernels_v1"
                if self.use_unified_tracking_score
                else "protomotions_legacy_root_aware_v1"
            ),
            "completion": float(min(completions)),
            "max_joint_error_rad": float(max(joint_errs)),
            "fall_detected": bool(any(falls)),
            "root_trajectory_error_mean_m": float(max(traj_errs)),
            "root_displacement_error_m": float(max(disp_errs)),
            "fall_protocol": "reference_conditioned_v012_persistent",
            "absolute_low_root_detected": bool(any(absolute_low_roots)),
            "max_root_height_deficit_m": float(max(root_height_deficits, default=0.0)),
        }
        if reward_component_acc:
            result["reward_components"] = {
                name: float(value / max(wsum, 1e-9))
                for name, value in reward_component_acc.items()
            }
            # Keep the paper-facing raw diagnostics directly addressable by
            # the Pareto preference selector. They are diagnostics, not extra
            # scalar reward terms.
            for name in ("er_mpjpe_mm", "evel_mps", "eacc_mps2"):
                if name in reward_component_acc:
                    result[name] = float(
                        reward_component_acc[name] / max(wsum, 1e-9)
                    )
        if len(self._judges) > 1:
            result["per_judge"] = per_judge
        return result

    def _simulate_judge(self, judge: Dict[str, object], motion_path: Path, out_json: Path) -> Dict[str, float]:
        if not self.simulate_subprocess:
            return self._simulate(
                onnx_path=judge["onnx"],
                motion_file=str(motion_path),
                output_json_path=str(out_json),
                mjcf_path=judge["mjcf"],
                body_mesh_mapping=self._mesh_mapping_for(judge["mjcf"]),
                subsample_factor=self.robot_json_subsample,
                terminate_on_unexpected_fall=False,
                export_frames=False,
                collect_tracking_trajectories=self.use_unified_tracking_score,
            )

        payload = {
            "onnx_path": judge["onnx"],
            "motion_file": str(motion_path),
            "output_json_path": str(out_json),
            "mjcf_path": judge["mjcf"],
            "subsample_factor": self.robot_json_subsample,
            "collect_tracking_trajectories": self.use_unified_tracking_score,
        }
        ctx = mp.get_context(self.simulate_mp_context)
        result_queue = ctx.Queue()
        proc = ctx.Process(target=_simulate_motion_worker, args=(payload, result_queue))
        proc.start()
        timeout = self.simulate_timeout_s if self.simulate_timeout_s > 0 else None
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            raise TimeoutError(
                f"simulate timeout after {self.simulate_timeout_s:.1f}s: "
                f"{Path(str(motion_path)).name} judge={judge.get('name', 'judge')}"
            )
        try:
            message = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                f"simulate subprocess exited with code {proc.exitcode} and no result: "
                f"{Path(str(motion_path)).name}"
            ) from exc
        if message.get("status") != "ok":
            raise RuntimeError(
                f"simulate subprocess failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        return message["stats"]

    def score_csv_dir(self, csv_dir: Path, work_dir: Path) -> Dict[str, Dict[str, float]]:
        """Convert all CSVs in ``csv_dir`` and score each. Returns {stem: metrics}.

        Adversarial score is LOWER == better (more trackable). Failures get
        ``error_penalty`` so they are never selected as best-of-N.
        """
        csv_dir = Path(csv_dir)
        work_dir = Path(work_dir)
        proto_dir = work_dir / "proto"
        json_dir = work_dir / "json"
        proto_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, float]] = {}
        stems = sorted(p.stem for p in csv_dir.glob("*.csv"))
        try:
            self._convert_csv_dir(csv_dir, proto_dir)
        except Exception as exc:  # convert failed for the whole batch
            for stem in stems:
                results[stem] = {"score": self.error_penalty, "error": f"convert: {exc}"}
            return results

        def score_stem(stem: str) -> tuple[str, Dict[str, float]]:
            motions = sorted(proto_dir.glob(f"{stem}*.motion"))
            if not motions:
                return stem, {"score": self.error_penalty, "error": "no .motion"}
            try:
                result = self.score_motion_file(motions[0], json_dir / f"{stem}.json")
            except Exception as exc:
                result = {"score": self.error_penalty, "error": str(exc)}
                if os.environ.get("PHYSFLOW_PROTO_DEBUG_ERRORS"):
                    try:
                        debug_path = work_dir / f"{stem}_proto_error.json"
                        debug_path.write_text(json.dumps({
                            "stem": stem,
                            "motion": str(motions[0]),
                            "error": f"{type(exc).__name__}: {exc}",
                        }, indent=2))
                    except Exception:
                        pass
            return stem, result

        score_workers = max(1, int(os.environ.get("PHYSFLOW_PROTO_SCORE_WORKERS", "1")))
        if score_workers == 1 or len(stems) < 2:
            scored_items = map(score_stem, stems)
        else:
            executor = ThreadPoolExecutor(
                max_workers=min(score_workers, len(stems)),
                thread_name_prefix="proto-reward",
            )
            scored_items = executor.map(score_stem, stems)
        try:
            for stem, result in scored_items:
                results[stem] = result
        finally:
            if score_workers > 1 and len(stems) >= 2:
                executor.shutdown(wait=True)
        return results


def _os_environ() -> Dict[str, str]:
    import os

    return os.environ.copy()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
