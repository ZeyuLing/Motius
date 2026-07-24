#!/usr/bin/env python3
"""Submit one finite 3DPW monocular evaluation to an elastic Taiji pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


# Preserve the Taiji-visible cq11 mount. Host-side cwd/__file__ canonicalization
# expands it to a path that is not mounted inside Taiji containers.
ROOT = Path(
    os.environ.get(
        "MOTIUS_TAIJI_ROOT",
        "/apdcephfs_cq11/share_1467498/home/zeyuling/Motius",
    )
)
MODEL_SOURCE = (
    "/apdcephfs_cq11/share_1467498/home/chingshuai/Template/empty"
)
IMAGE = "mirrors.tencent.com/jeffryli/tlinux3.2-python3.10-cuda11.8:v0.3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=("gem_smpl", "gem_x", "prompthmr", "gvhmr", "hymotion_v2m"),
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--revision", default="persistent")
    parser.add_argument("--gpu-name")
    parser.add_argument("--business")
    parser.add_argument("--cuda-version")
    parser.add_argument("--priority-level", default="LOW")
    parser.add_argument("--output-method")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    token = os.environ.get("TOKEN")
    if not token:
        raise RuntimeError("TOKEN must be set without placing it in this script.")
    if args.gpus < 1:
        raise ValueError("--gpus must be positive.")

    is_a100 = args.method == "gem_smpl"
    gpu_name = args.gpu_name or ("A100" if is_a100 else "H20")
    business = args.business or "AILab_DHC_DC"
    if business in {"AILab_DHA", "AILab_DHC_DD"}:
        raise ValueError(
            f"{business} compute GPUs are unavailable; use AILab_DHC_DC or "
            "the maintained TaiJi_HYAide_HY3D_ZW4 pool."
        )
    cuda_version = args.cuda_version or (
        "11.0" if gpu_name.upper().startswith("A100") else "12.0"
    )
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    task_flag = f"motius-3dpw-{args.method}-{args.revision}-{stamp}"
    log = (
        ROOT
        / "outputs/evaluation/monocular_capture/3dpw_test/schedulers"
        / f"{args.method}_{args.revision}_{stamp}.log"
    )
    if args.method == "hymotion_v2m":
        bootstrap = "tools/bootstrap_run_3dpw_hymotion_v2m_taiji.sh"
    else:
        bootstrap = "tools/bootstrap_run_3dpw_monocular_taiji.sh"
    snapshot = f"/tmp/motius_bootstrap_{args.method}.sh"
    method_args = (
        f"{args.gpus}"
        if args.method == "hymotion_v2m"
        else f"{args.method} {args.gpus}"
    )
    run_command = (
        f"cp {ROOT / bootstrap} {snapshot} && chmod +x {snapshot} && "
        f"MOTIUS_ROOT={ROOT} "
        f"MOTIUS_OUTPUT_METHOD={args.output_method or args.method} "
        f"MOTIUS_MAX_SEQUENCES={args.max_sequences or ''} "
        f"MOTIUS_MAX_FRAMES={args.max_frames or ''} "
        f"bash {snapshot} {method_args}"
    )
    start_cmd = (
        f"mkdir -p {log.parent} && cd {ROOT} && "
        f"{run_command} > {log} 2>&1"
    )
    config = {
        "Token": token,
        "business_flag": business,
        # Storage authorization for the canonical CQ11 mount, not compute.
        "mount_ceph_business_flag": "AILab_DHA",
        "extra_plat_business": business,
        "is_elasticity": True,
        "priority_level": args.priority_level,
        "elastic_level": 1,
        "enable_evicted_pulled_up": False,
        "enable_evicted_end_task": True,
        "keep_alive": False,
        "keep_running_after_trainer_finish": False,
        "exit_cmd": "",
        "init_cmd": "",
        "exec_start_in_all_mpi_pods": True,
        "cuda_version": cuda_version,
        "report_period": 60,
        "model_local_file_path": MODEL_SOURCE,
        "project_id": 0,
        "host_num": 1,
        "host_gpu_num": args.gpus,
        "image_full_name": IMAGE,
        "task_flag": task_flag,
        "GPUName": gpu_name,
        "start_cmd": start_cmd,
    }
    fd, raw_path = tempfile.mkstemp(prefix="motius-taiji-", suffix=".json")
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(config, stream)
        path.chmod(0o600)
        subprocess.run(
            ["taiji_client", "start", "-scfg", str(path)],
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
