#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:?Usage: tools/run_3dpw_monocular_taiji.sh METHOD [GPU_COUNT]}"
GPU_COUNT="${2:-${GPU_COUNT:-8}}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/motius_3dpw_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/evaluation/monocular_capture/3dpw_test}"
OUTPUT_METHOD="${MOTIUS_OUTPUT_METHOD:-${METHOD}}"
LOG_ROOT="${OUTPUT_ROOT}/${OUTPUT_METHOD}/logs"
TARGET_INDEX="${ROOT}/outputs/evaluation/monocular_capture/ground_truth/3dpw_joint_only/ground_truth_index.json"
SHARD_PLAN="${OUTPUT_ROOT}/${OUTPUT_METHOD}/shard_plan.json"
OCCUPY_PID_FILE="${OCCUPY_PID_FILE:-/tmp/motius_setup_occupy.pid}"

case "${METHOD}" in
  prompthmr)
    METHOD_ROOT="${PROMPTHMR_ROOT:-${ROOT}/outputs/tmp/prompthmr/upstream}"
    PYTHON_BIN="${PROMPTHMR_PYTHON:-${ROOT}/outputs/tmp/conda-envs/phmr_pt2.4/bin/python}"
    METHOD_ARGS=(--prompthmr-root "${METHOD_ROOT}" --prompthmr-python "${PYTHON_BIN}")
    ;;
  gem_smpl)
    METHOD_ROOT="${GEM_SMPL_ROOT:-${ROOT}/outputs/tmp/gem_smpl/upstream}"
    PYTHON_BIN="${GEM_SMPL_PYTHON:-${METHOD_ROOT}/.venv/bin/python}"
    METHOD_ARGS=(--gem-smpl-root "${METHOD_ROOT}")
    ;;
  gem_x)
    METHOD_ROOT="${GEM_X_ROOT:-${ROOT}/outputs/tmp/gem_x/upstream}"
    PYTHON_BIN="${GEM_X_PYTHON:-${METHOD_ROOT}/.venv/bin/python}"
    METHOD_ARGS=(--gem-x-root "${METHOD_ROOT}")
    ;;
  gvhmr)
    METHOD_ROOT="${GVHMR_ROOT:-${ROOT}/outputs/tmp/gvhmr/upstream}"
    PYTHON_BIN="${GVHMR_PYTHON:-${ROOT}/outputs/tmp/gvhmr/conda-env/bin/python}"
    METHOD_ARGS=(
      --gvhmr-root "${METHOD_ROOT}"
      --gvhmr-python "${PYTHON_BIN}"
    )
    ;;
  *) echo "Unsupported method: ${METHOD}" >&2; exit 2 ;;
esac
if (( GPU_COUNT < 1 )); then
  echo "GPU_COUNT must be positive." >&2
  exit 2
fi

cd "${ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Method runtime Python is not ready: ${PYTHON_BIN}" >&2
  exit 3
fi
if [[ ! -f "${STAGE_ROOT}/videos/manifest.json" ]]; then
  tools/stage_3dpw_test_data.sh "${STAGE_ROOT}"
  "${PYTHON_BIN}" tools/materialize_3dpw_test_videos.py \
    --data-root "${STAGE_ROOT}" \
    --output-dir "${STAGE_ROOT}/videos"
fi

mkdir -p "${LOG_ROOT}"
PLAN_ARGS=()
if [[ -n "${MOTIUS_MAX_SEQUENCES:-}" ]]; then
  PLAN_ARGS+=(--max-sequences "${MOTIUS_MAX_SEQUENCES}")
fi
"${PYTHON_BIN}" tools/build_3dpw_monocular_shard_plan.py \
  --video-manifest "${STAGE_ROOT}/videos/manifest.json" \
  --ground-truth-index "${TARGET_INDEX}" \
  --prediction-dir "${OUTPUT_ROOT}/${OUTPUT_METHOD}/predictions" \
  --num-shards "${GPU_COUNT}" \
  --output "${SHARD_PLAN}" \
  "${PLAN_ARGS[@]}"
PENDING_SEQUENCES="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["pending"])' \
    "${SHARD_PLAN}"
)"
if [[ "${PENDING_SEQUENCES}" == "0" ]]; then
  echo "All requested ${METHOD} sequences already have predictions."
  exit 0
fi
if [[ -f "${OCCUPY_PID_FILE}" ]]; then
  occupy_pid="$(<"${OCCUPY_PID_FILE}")"
  if [[ "${occupy_pid}" =~ ^[0-9]+$ ]] && kill -0 "${occupy_pid}" 2>/dev/null; then
    kill -- "-${occupy_pid}"
    for _ in {1..60}; do
      kill -0 "${occupy_pid}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${occupy_pid}" 2>/dev/null; then
      kill -9 -- "-${occupy_pid}"
    fi
  fi
  rm -f "${OCCUPY_PID_FILE}"
fi

pids=()
EXTRA_ARGS=()
if [[ -n "${MOTIUS_MAX_SEQUENCES:-}" ]]; then
  EXTRA_ARGS+=(--max-sequences "${MOTIUS_MAX_SEQUENCES}")
fi
if [[ -n "${MOTIUS_MAX_FRAMES:-}" ]]; then
  EXTRA_ARGS+=(--max-frames "${MOTIUS_MAX_FRAMES}")
fi
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for ((rank = 0; rank < GPU_COUNT; rank++)); do
  CUDA_VISIBLE_DEVICES="${rank}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" tools/run_3dpw_monocular_shard.py \
    --method "${METHOD}" \
    --video-manifest "${STAGE_ROOT}/videos/manifest.json" \
    --video-dir "${STAGE_ROOT}/videos" \
    --output-root "${OUTPUT_ROOT}" \
    --output-method "${OUTPUT_METHOD}" \
    --assignment-plan "${SHARD_PLAN}" \
    --shard-id "${rank}" \
    --num-shards "${GPU_COUNT}" \
    "${METHOD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    >"${LOG_ROOT}/shard_${rank}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - INT TERM

METHOD="${METHOD}" OUTPUT_METHOD="${OUTPUT_METHOD}" GPU_COUNT="${GPU_COUNT}" FAILED="${failed}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"]) / os.environ["OUTPUT_METHOD"]
status = sorted((root / "status").glob("*.json"))
records = [json.loads(path.read_text()) for path in status]
payload = {
    "method": os.environ["METHOD"],
    "output_method": os.environ["OUTPUT_METHOD"],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "shards_failed": bool(int(os.environ["FAILED"])),
    "complete": sum(item.get("status") == "complete" for item in records),
    "failed": sum(item.get("status") == "failed" for item in records),
    "population": len(records),
}
(root / "finish.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY

exit "${failed}"
