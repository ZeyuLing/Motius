"""Official IsaacLab SONIC execution reward for G1 generator post-training."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import joblib
import numpy as np


def _single_process_subprocess_env() -> Dict[str, str]:
    """Return an environment that cannot join the parent DDP process group."""

    env = os.environ.copy()
    distributed_keys = {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    }
    for key in list(env):
        if key in distributed_keys or key.startswith("TORCHELASTIC_"):
            env.pop(key, None)
    return env


class SonicJudgeReward:
    """Score one generator candidate batch with an official SONIC checkpoint.

    A single Isaac Sim process evaluates the whole CSV directory. The resulting
    30-FPS canonical qpos/body trajectories are passed through the same unified
    evaluator used by the paper tables, so online reward and final evaluation
    share their continuous tracking definitions.
    """

    def __init__(
        self,
        checkpoint: str,
        trainee_checkpoint: Optional[str] = None,
        project_root: Optional[str] = None,
        gpu_id: int = 1,
        num_envs: int = 16,
        input_fps: int = 30,
        eval_timeout_s: int = 1800,
        eval_attempts: int = 3,
        eval_retry_delay_s: int = 10,
        error_penalty: float = 5.0,
        runner: Optional[str] = None,
        materializer: Optional[str] = None,
        evaluator: Optional[str] = None,
        persistent: bool = True,
        service_startup_timeout_s: int = 5400,
        postprocess_python: Optional[str] = None,
        require_runner: bool = True,
    ) -> None:
        configured_project_root = project_root or os.environ.get(
            "PHYSFLOW_PROJECT_ROOT"
        )
        self.project_root = Path(
            configured_project_root or Path(__file__).resolve().parents[3]
        ).resolve()
        project_root_text = str(self.project_root)
        if project_root_text not in sys.path:
            sys.path.insert(0, project_root_text)
        self.checkpoint = Path(checkpoint).resolve()
        self.trainee_checkpoint = (
            Path(trainee_checkpoint).resolve() if trainee_checkpoint else None
        )
        self.gpu_id = int(gpu_id)
        self.num_envs = max(1, int(num_envs))
        self.input_fps = int(input_fps)
        self.eval_timeout_s = int(eval_timeout_s)
        self.eval_attempts = max(1, int(eval_attempts))
        self.eval_retry_delay_s = max(0, int(eval_retry_delay_s))
        self.error_penalty = float(error_penalty)
        configured_runner = runner or os.environ.get(
            "MOTIUS_GENTRACK_SONIC_RUNNER"
        )
        self.runner = (
            Path(configured_runner).expanduser()
            if configured_runner
            else None
        )
        if self.runner is not None and not self.runner.is_absolute():
            self.runner = self.project_root / self.runner
        if self.runner is not None:
            self.runner = self.runner.absolute()
        configured_materializer = materializer or os.environ.get(
            "MOTIUS_GENTRACK_SONIC_MATERIALIZER"
        )
        self.materializer = (
            Path(configured_materializer).expanduser()
            if configured_materializer
            else None
        )
        if self.materializer is not None and not self.materializer.is_absolute():
            self.materializer = self.project_root / self.materializer
        if self.materializer is not None:
            self.materializer = self.materializer.absolute()
        configured_evaluator = evaluator or "tools/gentrack/evaluate.py"
        self.evaluator = Path(configured_evaluator).expanduser()
        if not self.evaluator.is_absolute():
            self.evaluator = self.project_root / self.evaluator
        self.evaluator = self.evaluator.absolute()
        self.persistent = bool(persistent)
        self.service_startup_timeout_s = max(60, int(service_startup_timeout_s))
        # Preserve an explicitly configured virtualenv launcher. Resolving a
        # ``venv/bin/python`` symlink to the system interpreter discards the
        # virtualenv prefix and can silently remove packages such as mujoco.
        self.postprocess_python = Path(
            postprocess_python or sys.executable
        ).expanduser().absolute()
        self.runtime_env_root = self.postprocess_python.parent.parent.parent
        # The validated Python/IsaacLab venv and the persisted NVIDIA graphics
        # bundle can live on different mounts.
        # Keep those roots independent so selecting a native venv does not
        # redirect the evaluator's EGL lookup to the wrong Ceph share.
        self.graphics_env_root = self._discover_graphics_env_root()
        self._service_process: Optional[subprocess.Popen] = None
        self._service_root: Optional[Path] = None
        self._service_log_handle = None
        self._service_request_count = 0
        self._service_started_at: Optional[float] = None
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SONIC checkpoint not found: {self.checkpoint}")
        if self.trainee_checkpoint is not None and not self.trainee_checkpoint.is_file():
            raise FileNotFoundError(
                f"SONIC trainee checkpoint not found: {self.trainee_checkpoint}"
            )
        if require_runner and (self.runner is None or not self.runner.is_file()):
            raise FileNotFoundError(
                "SONIC runner not found; pass runner=... or set "
                "MOTIUS_GENTRACK_SONIC_RUNNER to the official simulator "
                f"launcher (resolved value: {self.runner})"
            )
        if self.persistent and (
            self.materializer is None or not self.materializer.is_file()
        ):
            raise FileNotFoundError(
                "persistent SONIC reward requires the official trajectory "
                "materializer; pass materializer=... or set "
                "MOTIUS_GENTRACK_SONIC_MATERIALIZER "
                f"(resolved value: {self.materializer})"
            )
        if not self.evaluator.is_file():
            raise FileNotFoundError(
                f"GenTrack unified evaluator not found: {self.evaluator}"
            )
        if not self.postprocess_python.is_file():
            raise FileNotFoundError(
                f"SONIC postprocess Python not found: {self.postprocess_python}"
            )

        from motius.evaluation.gentrack.scoring import (
            DEFAULT_G1_SCORE_CONFIG,
            compute_sonic_tracking_score,
            compute_g1_adversarial_score,
        )

        self._score_config = DEFAULT_G1_SCORE_CONFIG
        self._compute_score = compute_g1_adversarial_score
        self._compute_sonic_tracking_score = compute_sonic_tracking_score
        atexit.register(self.close)

    def _discover_graphics_env_root(self) -> Path:
        """Find the parent directory containing the persisted NVIDIA bundle."""
        configured = os.environ.get("PHYSFLOW_GRAPHICS_ENV_ROOT")
        if configured:
            return Path(configured).expanduser().absolute()

        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            candidate = Path(entry).expanduser()
            if (
                candidate.name.startswith("nvidia_")
                and candidate.name.endswith("_graphics")
                and (
                    (candidate / "libEGL_nvidia.so.0").exists()
                    or any(candidate.glob("libEGL_nvidia.so.*"))
                )
            ):
                return candidate.absolute().parent

        return self.runtime_env_root

    def _postprocess_env(self) -> Dict[str, str]:
        """Use the validated SONIC runtime for canonicalization and scoring.

        The simulator launcher already honors ``PHYSFLOW_SONIC_SITE_PACKAGES``,
        but materialization and unified evaluation are direct Python
        subprocesses. Without the same override they can fall back to the T2M
        runtime and miss native packages such as MuJoCo.
        """
        env = _single_process_subprocess_env()
        entries = [
            env.get("PHYSFLOW_SONIC_SITE_PACKAGES", ""),
            env.get("PHYSFLOW_SONIC_EXTRA_PYTHONPATH", ""),
            str(self.project_root),
            env.get("PYTHONPATH", ""),
        ]
        pythonpath = []
        for entry in entries:
            for path in entry.split(os.pathsep):
                if path and path not in pythonpath:
                    pythonpath.append(path)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONHOME", None)
        return env

    def _sonic_extra_pythonpath(self) -> str:
        """Add the shared Kit kernel only when a staged runtime lacks it."""
        entries = [
            path
            for path in os.environ.get(
                "PHYSFLOW_SONIC_EXTRA_PYTHONPATH", ""
            ).split(os.pathsep)
            if path
        ]
        staged_site = os.environ.get("PHYSFLOW_SONIC_SITE_PACKAGES", "")
        staged_has_kit = bool(
            staged_site and (Path(staged_site) / "omni" / "kit_app.py").is_file()
        )
        if not staged_has_kit:
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            shared_site = (
                self.graphics_env_root
                / f"physflow_sonic_isaaclab_py{sys.version_info.major}{sys.version_info.minor}"
                / "lib"
                / f"python{version}"
                / "site-packages"
            )
            if (
                (shared_site / "omni" / "kit_app.py").is_file()
                and str(shared_site) not in entries
            ):
                entries.append(str(shared_site))
        return os.pathsep.join(entries)

    def _physical_gpu_token(self) -> str:
        """Map the judge's relative GPU index through the parent's visibility.

        The generator launcher commonly exposes a pair such as ``2,3`` and
        assigns the generator to relative GPU 0 and SONIC to relative GPU 1.
        The SONIC shell launches a fresh process and overwrites
        ``CUDA_VISIBLE_DEVICES``, so passing the relative index through would
        incorrectly select physical GPU 1 for every pair.
        """
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible:
            return str(self.gpu_id)
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if not 0 <= self.gpu_id < len(tokens):
            raise ValueError(
                f"SONIC relative gpu_id={self.gpu_id} is outside "
                f"CUDA_VISIBLE_DEVICES={visible!r}"
            )
        return tokens[self.gpu_id]

    @staticmethod
    def _atomic_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)

    def _build_protocol(self, csv_dir: Path, protocol_root: Path) -> list[str]:
        input_root = protocol_root / "inputs" / "reward_batch"
        npz_dir = input_root / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)
        stems = []
        for csv_path in sorted(csv_dir.glob("*.csv")):
            qpos = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
            if qpos.ndim == 1:
                qpos = qpos[None]
            if qpos.ndim != 2 or qpos.shape[1] != 36:
                raise ValueError(f"{csv_path}: expected [T, 36] qpos, got {qpos.shape}")
            stem = csv_path.stem
            np.savez(
                npz_dir / f"{stem}.npz",
                qpos=qpos,
                frequency=np.float32(self.input_fps),
                fps=np.float32(self.input_fps),
                case_id=np.asarray(stem),
            )
            stems.append(stem)
        if not stems:
            raise ValueError(f"no candidate CSV files found in {csv_dir}")
        self._atomic_json(input_root / "manifest.json", stems)
        return stems

    def _run_sonic(
        self,
        protocol_root: Path,
        canonical_root: Path,
        output_root: Path,
        checkpoint: Path,
        method: str,
    ) -> None:
        env = os.environ.copy()
        env.update(
            {
                "PROTOCOL_ROOT": str(protocol_root),
                "CANONICAL_ROOT": str(canonical_root),
                "OUTPUT_ROOT": str(output_root),
                "SPLITS": "reward_batch",
                "TOTAL_SHARDS": "1",
                "SHARD_ID": "0",
                "GPU_ID": self._physical_gpu_token(),
                "NUM_ENVS": str(self.num_envs),
                "SONIC_CHECKPOINT": str(checkpoint),
                "CANONICAL_METHOD": method,
                "PHYSFLOW_SONIC_USE_USD_G1": "0",
                # Keep simulation and post-processing on the same validated
                # interpreter instead of silently crossing runtime mounts.
                "PHYSFLOW_ENV_ROOT": str(self.runtime_env_root),
                "PHYSFLOW_GRAPHICS_ENV_ROOT": str(self.graphics_env_root),
                "SONIC_ISAACLAB_PYTHON": str(self.postprocess_python),
                "PREPARE_PYTHON": str(self.postprocess_python),
                "PHYSFLOW_SONIC_EXTRA_PYTHONPATH": (
                    self._sonic_extra_pythonpath()
                ),
                "SONIC_EVAL_TIMEOUT_SECONDS": str(self.eval_timeout_s),
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "ACCEPT_EULA": "Y",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "OMNI_USER_ACCEPT_EULA": "YES",
                "ISAACSIM_ACCEPT_EULA": "YES",
            }
        )
        output_root.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "sonic_reward.log"
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.eval_attempts + 1):
            # A failed Kit startup can leave descendants alive after the shell
            # exits. Own a process group so retries cannot inherit a stale
            # Vulkan/Isaac process on the reward GPU.
            env["FORCE_EVAL"] = "1"
            with log_path.open("a") as log_file:
                log_file.write(
                    f"\n[physflow-sonic-reward] attempt "
                    f"{attempt}/{self.eval_attempts}\n"
                )
                log_file.flush()
                process = subprocess.Popen(
                    ["bash", str(self.runner)],
                    cwd=self.project_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=self.eval_timeout_s + 120)
                    if return_code == 0:
                        return
                    last_error = subprocess.CalledProcessError(
                        return_code, ["bash", str(self.runner)]
                    )
                except subprocess.TimeoutExpired as exc:
                    last_error = exc
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=15)
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
            if attempt < self.eval_attempts:
                time.sleep(self.eval_retry_delay_s * attempt)
        assert last_error is not None
        raise RuntimeError(
            f"SONIC reward failed after {self.eval_attempts} attempts"
        ) from last_error

    def _service_log_tail(self, max_lines: int = 80) -> str:
        if self._service_root is None:
            return ""
        path = self._service_root / "service.log"
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-max_lines:])

    def close(self) -> None:
        process = self._service_process
        root = self._service_root
        self._service_process = None
        self._service_root = None
        self._service_started_at = None
        if root is not None:
            try:
                (root / "shutdown").touch()
            except OSError:
                pass
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if self._service_log_handle is not None:
            self._service_log_handle.close()
            self._service_log_handle = None

    def _wait_for_service_ready(
        self,
        process: subprocess.Popen,
        service_root: Path,
        timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        ready_path = service_root / "ready.json"
        while True:
            # A live persistent worker may already be ready when a later
            # request re-enters this method.  Check the sentinel before the
            # timeout boundary so a slow initial cold start does not make all
            # subsequent warm requests fail with an immediate zero-second
            # startup timeout.
            if ready_path.is_file():
                ready = json.loads(ready_path.read_text())
                if Path(ready["quality_checkpoint"]) != self.checkpoint:
                    raise RuntimeError(
                        f"SONIC service loaded wrong quality checkpoint: {ready}"
                    )
                expected_trainee = self.trainee_checkpoint or self.checkpoint
                if Path(ready["trainee_checkpoint"]) != expected_trainee:
                    raise RuntimeError(
                        f"SONIC service loaded wrong trainee checkpoint: {ready}"
                    )
                return
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"SONIC service exited during startup with code {return_code}\n"
                    f"{self._service_log_tail()}"
                )
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        raise TimeoutError(
            f"SONIC service did not become ready within "
            f"{self.service_startup_timeout_s}s\n{self._service_log_tail()}"
        )

    @contextmanager
    def _serialized_service_startup(self) -> Iterator[None]:
        """Serialize Isaac Sim cold starts that share one host runtime.

        Isaac Sim 4.5 workers can execute concurrently after initialization,
        but simultaneous scene construction against the shared Kit runtime
        intermittently corrupts native allocator state. The lock covers only
        cold start through ``ready.json``; warm rollout requests remain fully
        parallel across judge GPUs.
        """
        lock_path = Path(
            os.environ.get(
                "PHYSFLOW_SONIC_STARTUP_LOCK",
                "/tmp/physflow_sonic_service_startup.lock",
            )
        ).expanduser().resolve()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _start_service(self, protocol_root: Path, canonical_root: Path) -> None:
        if (
            self._service_process is not None
            and self._service_process.poll() is None
            and self._service_root is not None
        ):
            elapsed = (
                time.monotonic() - self._service_started_at
                if self._service_started_at is not None
                else 0.0
            )
            self._wait_for_service_ready(
                self._service_process,
                self._service_root,
                max(0.0, self.service_startup_timeout_s - elapsed),
            )
            return

        with self._serialized_service_startup():
            self.close()

            # The runner changes cwd to SONIC_REPO before launching Isaac Sim.
            # Keep the IPC endpoint absolute so parent and worker cannot poll
            # two different same-named relative directories.
            service_parent = Path(
                os.environ.get("PHYSFLOW_SONIC_SERVICE_ROOT", "/tmp")
            ).expanduser().resolve()
            service_parent.mkdir(parents=True, exist_ok=True)
            service_root = Path(
                tempfile.mkdtemp(
                    prefix=f"physflow_sonic_reward_{os.getpid()}_",
                    dir=service_parent,
                )
            )
            env = _single_process_subprocess_env()
            env.update(
                {
                    "PROTOCOL_ROOT": str(protocol_root),
                    "CANONICAL_ROOT": str(canonical_root),
                    "OUTPUT_ROOT": str(service_root / "bootstrap"),
                    "SPLITS": "reward_batch",
                    "TOTAL_SHARDS": "1",
                    "SHARD_ID": "0",
                    "GPU_ID": self._physical_gpu_token(),
                    "NUM_ENVS": str(self.num_envs),
                    "SONIC_CHECKPOINT": str(self.checkpoint),
                    "CANONICAL_METHOD": "sonic_service_bootstrap",
                    "PHYSFLOW_SONIC_USE_USD_G1": "0",
                    "PHYSFLOW_ENV_ROOT": str(self.runtime_env_root),
                    "PHYSFLOW_GRAPHICS_ENV_ROOT": str(self.graphics_env_root),
                    "SONIC_ISAACLAB_PYTHON": str(self.postprocess_python),
                    "PREPARE_PYTHON": str(self.postprocess_python),
                    "PHYSFLOW_SONIC_EXTRA_PYTHONPATH": (
                        self._sonic_extra_pythonpath()
                    ),
                    "PHYSFLOW_SONIC_SERVICE_DIR": str(service_root),
                    "PHYSFLOW_SONIC_SERVICE_PARENT_PID": str(os.getpid()),
                    "PHYSFLOW_SONIC_TRAINEE_CHECKPOINT": str(
                        self.trainee_checkpoint or self.checkpoint
                    ),
                    "SONIC_EVAL_TIMEOUT_SECONDS": str(self.eval_timeout_s),
                    "SONIC_SERVICE_LIFETIME_SECONDS": "86400",
                    # vulkaninfo can be a false-negative on headless GPU hosts
                    # even when the native-loader runtime works.
                    "SKIP_VULKAN_PREFLIGHT": os.environ.get(
                        "PHYSFLOW_SONIC_SKIP_VULKAN_PREFLIGHT", "1"
                    ),
                    "NVIDIA_DRIVER_CAPABILITIES": "all",
                    "ACCEPT_EULA": "Y",
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "OMNI_USER_ACCEPT_EULA": "YES",
                    "ISAACSIM_ACCEPT_EULA": "YES",
                }
            )
            log_handle = (service_root / "service.log").open("a")
            process = subprocess.Popen(
                ["bash", str(self.runner)],
                cwd=self.project_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._service_process = process
            self._service_root = service_root
            self._service_log_handle = log_handle
            self._service_started_at = time.monotonic()
            self._wait_for_service_ready(
                process,
                service_root,
                self.service_startup_timeout_s,
            )

    def _prepare_service_motion_dir(
        self, protocol_root: Path, stems: list[str], work_dir: Path
    ) -> Path:
        from motius.models.gentrack.sonic_motion import (
            _load_qpos,
            qpos_to_sonic_motion,
        )

        npz_dir = protocol_root / "inputs" / "reward_batch" / "npz"
        motion_dir = work_dir / "sonic_service_motion"
        motion_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            qpos, _ = _load_qpos(npz_dir / f"{stem}.npz", float(self.input_fps))
            payload = {stem: qpos_to_sonic_motion(qpos, int(self.input_fps))}
            joblib.dump(payload, motion_dir / f"{stem}.pkl")
        return motion_dir

    def _request_service(
        self,
        *,
        motion_dir: Path,
        num_motions: int,
        quality_output_dir: Path,
        trainee_output_dir: Optional[Path],
    ) -> dict:
        if self._service_root is None or self._service_process is None:
            raise RuntimeError("SONIC service is not running")
        self._service_request_count += 1
        request_id = (
            f"p{os.getpid()}_{self._service_request_count:06d}_{uuid.uuid4().hex[:8]}"
        )
        request = {
            "request_id": request_id,
            "motion_dir": str(motion_dir.resolve()),
            "num_motions": int(num_motions),
            "quality_output_dir": str(quality_output_dir.resolve()),
            "trainee_output_dir": (
                str(trainee_output_dir.resolve()) if trainee_output_dir is not None else None
            ),
        }
        request_path = self._service_root / "requests" / f"{request_id}.json"
        response_path = self._service_root / "responses" / f"{request_id}.json"
        self._atomic_json(request_path, request)

        deadline = time.monotonic() + self.eval_timeout_s
        while time.monotonic() < deadline:
            if response_path.is_file():
                response = json.loads(response_path.read_text())
                consumed_path = (
                    self._service_root / "consumed_responses" / response_path.name
                )
                self._atomic_json(consumed_path, response)
                response_path.unlink(missing_ok=True)
                if not response.get("ok"):
                    raise RuntimeError(
                        f"SONIC service request failed: {response.get('error')}\n"
                        f"{response.get('traceback', '')}"
                    )
                return response
            return_code = self._service_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"SONIC service exited with code {return_code}\n{self._service_log_tail()}"
                )
            time.sleep(0.1)
        raise TimeoutError(f"SONIC service request {request_id} exceeded {self.eval_timeout_s}s")

    def _materialize_service_dump(
        self,
        *,
        dump_dir: Path,
        protocol_root: Path,
        canonical_root: Path,
        method: str,
    ) -> None:
        dump = dump_dir / "physflow_sonic_trajectories.npz"
        cmd = [
            str(self.postprocess_python),
            str(self.materializer),
            "--dump",
            str(dump),
            "--canonical-root",
            str(canonical_root),
            "--split",
            "reward_batch",
            "--method",
            method,
            "--protocol-input-dir",
            str(protocol_root / "inputs" / "reward_batch" / "npz"),
            "--manifest",
            str(protocol_root / "inputs" / "reward_batch" / "manifest.json"),
            "--source-fps",
            "50",
            "--output-fps",
            "30",
        ]
        subprocess.run(
            cmd,
            cwd=self.project_root,
            env=self._postprocess_env(),
            check=True,
            timeout=600,
        )

    def _run_unified_evaluator(
        self,
        protocol_root: Path,
        canonical_root: Path,
        eval_root: Path,
        method: str,
    ) -> None:
        table_root = canonical_root / "table_tracker"
        ref_root = table_root / "reference" / "reward_batch"
        exe_root = table_root / method / "reward_batch"
        cmd = [
            str(self.postprocess_python),
            str(self.evaluator),
            "evaluate-canonical-dirs",
            "--reference-dir",
            str(ref_root / "g1_body30"),
            "--execution-dir",
            str(exe_root / "g1_body30"),
            "--reference-qpos-dir",
            str(ref_root / "g1_qpos30"),
            "--execution-qpos-dir",
            str(exe_root / "g1_qpos30"),
            "--manifest",
            str(protocol_root / "inputs" / "reward_batch" / "manifest.json"),
            "--out-dir",
            str(eval_root),
            "--method",
            method,
            "--split",
            "reward_batch",
            "--fps",
            "30",
            "--workers",
            "4",
        ]
        with (eval_root.parent / "unified_eval.log").open("w") as log_file:
            subprocess.run(
                cmd,
                cwd=self.project_root,
                env=self._postprocess_env(),
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=600,
            )

    @staticmethod
    def _root_displacement_error(ref_path: Path, exe_path: Path) -> float:
        with np.load(ref_path, allow_pickle=False) as data:
            ref = np.asarray(data["qpos"], dtype=np.float32)
        with np.load(exe_path, allow_pickle=False) as data:
            exe = np.asarray(data["qpos"], dtype=np.float32)
        count = min(len(ref), len(exe))
        if count < 2:
            return float("inf")
        ref_delta = ref[count - 1, :3] - ref[0, :3]
        exe_delta = exe[count - 1, :3] - exe[0, :3]
        return float(np.linalg.norm(ref_delta - exe_delta))

    def _parse_metrics(
        self,
        stems: list[str],
        canonical_root: Path,
        eval_root: Path,
        method: str = "sonic_reward",
    ) -> Dict[str, Dict[str, float]]:
        rows = {}
        metrics_path = eval_root / "case_metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["case_id"])] = row

        table_root = canonical_root / "table_tracker"
        ref_qpos_dir = table_root / "reference" / "reward_batch" / "g1_qpos30"
        exe_qpos_dir = table_root / method / "reward_batch" / "g1_qpos30"
        ref_body_dir = table_root / "reference" / "reward_batch" / "g1_body30"
        exe_body_dir = table_root / method / "reward_batch" / "g1_body30"
        results: Dict[str, Dict[str, float]] = {}
        for stem in stems:
            row = rows.get(stem)
            if row is None:
                results[stem] = {"score": self.error_penalty, "error": "missing unified metric"}
                continue
            try:
                root_disp = self._root_displacement_error(
                    ref_qpos_dir / f"{stem}.npz", exe_qpos_dir / f"{stem}.npz"
                )
                completion = float(row["completion"])
                joint_error = float(row["max_joint_err_rad"])
                root_error = float(row["root_traj_err_m"])
                fall = bool(row["unexpected_fall"])
                legacy_score = float(
                    self._compute_score(
                        completion=completion,
                        max_joint_error_rad=joint_error,
                        root_trajectory_error_mean_m=root_error,
                        root_displacement_error_m=root_disp,
                        fall_detected=fall,
                        config=self._score_config,
                    )
                )
                with np.load(ref_body_dir / f"{stem}.npz", allow_pickle=False) as data:
                    ref_pos = np.asarray(data["body_pos"], dtype=np.float32)
                    ref_quat = np.asarray(data["body_quat"], dtype=np.float32)
                    ref_fps = float(
                        np.asarray(data["fps" if "fps" in data.files else "frequency"])
                        .reshape(-1)[0]
                    )
                with np.load(exe_body_dir / f"{stem}.npz", allow_pickle=False) as data:
                    exe_pos = np.asarray(data["body_pos"], dtype=np.float32)
                    exe_quat = np.asarray(data["body_quat"], dtype=np.float32)
                score, reward_components = self._compute_sonic_tracking_score(
                    reference_body_pos=ref_pos,
                    execution_body_pos=exe_pos,
                    reference_body_quat=ref_quat,
                    execution_body_quat=exe_quat,
                    fps=ref_fps,
                    completion=completion,
                    fall_detected=fall,
                )
                results[stem] = {
                    "score": score,
                    "legacy_root_aware_score": legacy_score,
                    "score_protocol": "released_sonic_plus_global_anchor_pos_v1",
                    "reward_components": reward_components,
                    "completion": completion,
                    "max_joint_error_rad": joint_error,
                    "fall_detected": fall,
                    "root_trajectory_error_mean_m": root_error,
                    "root_displacement_error_m": root_disp,
                    "e_joint_rad": float(row["e_joint_rad"]),
                    "er_mpjpe_mm": float(row["er_mpjpe_mm"]),
                    "evel_mps": float(row["evel_mps"]),
                    "eacc_mps2": float(row["eacc_mps2"]),
                    "mpjpe_mm": float(row["mpjpe_mm"]),
                    "mpjve_mps": float(row["mpjve_mps"]),
                    "root_vel_err_mps": float(row["root_vel_err_mps"]),
                    "sonic_success": bool(row["success_unified"]),
                }
            except Exception as exc:  # noqa: BLE001 - report one bad candidate.
                results[stem] = {"score": self.error_penalty, "error": repr(exc)}
        return results

    def _score_with_persistent_service(
        self,
        *,
        stems: list[str],
        protocol_root: Path,
        canonical_root: Path,
        work_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        motion_dir = self._prepare_service_motion_dir(protocol_root, stems, work_dir)
        quality_dump = work_dir / "sonic_service_quality_dump"
        trainee_dump = work_dir / "sonic_service_trainee_dump"
        distinct_trainee = (
            self.trainee_checkpoint is not None
            and self.trainee_checkpoint != self.checkpoint
        )

        last_error: Optional[BaseException] = None
        response = None
        for attempt in range(1, min(self.eval_attempts, 2) + 1):
            try:
                self._start_service(protocol_root, canonical_root)
                response = self._request_service(
                    motion_dir=motion_dir,
                    num_motions=len(stems),
                    quality_output_dir=quality_dump,
                    trainee_output_dir=trainee_dump if distinct_trainee else None,
                )
                break
            except Exception as exc:  # noqa: BLE001 - one controlled warm-worker restart.
                last_error = exc
                self.close()
                if attempt < min(self.eval_attempts, 2):
                    time.sleep(self.eval_retry_delay_s * attempt)
        if response is None:
            assert last_error is not None
            raise RuntimeError(
                f"persistent SONIC reward service failed: {last_error!r}"
            ) from last_error

        quality_eval = work_dir / "sonic_unified_quality"
        quality_eval.mkdir(parents=True, exist_ok=True)
        self._materialize_service_dump(
            dump_dir=quality_dump,
            protocol_root=protocol_root,
            canonical_root=canonical_root,
            method="sonic_quality",
        )
        self._run_unified_evaluator(
            protocol_root, canonical_root, quality_eval, "sonic_quality"
        )
        quality = self._parse_metrics(stems, canonical_root, quality_eval, "sonic_quality")

        if not distinct_trainee:
            for stem in stems:
                shared_metric = dict(quality[stem])
                quality[stem]["per_judge"] = {
                    "quality": shared_metric,
                    "trainee": dict(shared_metric),
                }
                quality[stem]["service_timing"] = dict(response)
            return quality

        trainee_eval = work_dir / "sonic_unified_trainee"
        trainee_eval.mkdir(parents=True, exist_ok=True)
        self._materialize_service_dump(
            dump_dir=trainee_dump,
            protocol_root=protocol_root,
            canonical_root=canonical_root,
            method="sonic_trainee",
        )
        self._run_unified_evaluator(
            protocol_root, canonical_root, trainee_eval, "sonic_trainee"
        )
        trainee = self._parse_metrics(stems, canonical_root, trainee_eval, "sonic_trainee")
        for stem in stems:
            quality_metric = quality[stem]
            quality_metric["per_judge"] = {
                "quality": dict(quality_metric),
                "trainee": dict(trainee[stem]),
            }
            quality_metric["service_timing"] = dict(response)
        return quality

    def score_csv_dir(self, csv_dir: Path, work_dir: Path) -> Dict[str, Dict[str, float]]:
        csv_dir = Path(csv_dir)
        work_dir = Path(work_dir)
        protocol_root = work_dir / "sonic_protocol"
        canonical_root = work_dir / "sonic_canonical"
        stems = sorted(path.stem for path in csv_dir.glob("*.csv"))
        try:
            stems = self._build_protocol(csv_dir, protocol_root)
            if self.persistent:
                return self._score_with_persistent_service(
                    stems=stems,
                    protocol_root=protocol_root,
                    canonical_root=canonical_root,
                    work_dir=work_dir,
                )
            quality_output = work_dir / "sonic_run_quality"
            quality_eval = work_dir / "sonic_unified_quality"
            self._run_sonic(
                protocol_root,
                canonical_root,
                quality_output,
                self.checkpoint,
                "sonic_quality",
            )
            quality_eval.mkdir(parents=True, exist_ok=True)
            self._run_unified_evaluator(
                protocol_root, canonical_root, quality_eval, "sonic_quality"
            )
            quality = self._parse_metrics(
                stems, canonical_root, quality_eval, "sonic_quality"
            )
            if self.trainee_checkpoint is None:
                return quality

            if self.trainee_checkpoint == self.checkpoint:
                for stem in stems:
                    quality_metric = quality[stem]
                    shared_metric = dict(quality_metric)
                    quality_metric["per_judge"] = {
                        "quality": shared_metric,
                        "trainee": dict(shared_metric),
                    }
                return quality

            trainee_output = work_dir / "sonic_run_trainee"
            trainee_eval = work_dir / "sonic_unified_trainee"
            self._run_sonic(
                protocol_root,
                canonical_root,
                trainee_output,
                self.trainee_checkpoint,
                "sonic_trainee",
            )
            trainee_eval.mkdir(parents=True, exist_ok=True)
            self._run_unified_evaluator(
                protocol_root, canonical_root, trainee_eval, "sonic_trainee"
            )
            trainee = self._parse_metrics(
                stems, canonical_root, trainee_eval, "sonic_trainee"
            )
            for stem in stems:
                quality_metric = quality[stem]
                quality_metric["per_judge"] = {
                    "quality": dict(quality_metric),
                    "trainee": dict(trainee[stem]),
                }
            return quality
        except Exception as exc:  # noqa: BLE001 - preserve group-relative batch shape.
            return {
                stem: {"score": self.error_penalty, "error": f"sonic reward: {exc}"}
                for stem in stems
            }
