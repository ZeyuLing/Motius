#!/usr/bin/env bash
set -euo pipefail

# Run the four HumanML3D M2T baselines on one 8-GPU Taiji worker. Predictions
# live on Ceph and are sample-addressed, so an evicted worker resumes in place.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/HumanML3D}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/m2t/humanml3d/full_v1}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${ROOT}/outputs/m2t/humanml3d/protocol_manifest.json}"
LOCAL_ROOT="${LOCAL_ROOT:-/dev/shm/motius_m2t_suite}"
MOTIONGPT_MODEL="${MOTIONGPT_MODEL:-ZeyuLing/Motius-MotionGPT-HumanML3D}"
MOTIONGPT3_MODEL="${MOTIONGPT3_MODEL:-ZeyuLing/Motius-MotionGPT3-HumanML3D}"
TM2T_MODEL="${TM2T_MODEL:-ZeyuLing/Motius-TM2T-HumanML3D}"
VERMO_MODEL="${VERMO_MODEL:-ZeyuLing/Motius-VerMo-HumanML3D}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-${MOTIUS_BODY_MODEL_DIR:-${ROOT}/checkpoints/body_models}}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${LOCAL_ROOT}/hf"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_ROOT}/logs" "${LOCAL_ROOT}"
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[m2t-suite] HumanML3D not found: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the official HumanML3D directory." >&2
    exit 2
fi

run_shard() {
    local gpu="$1"
    local method="$2"
    local num_shards="$3"
    local shard_index="$4"
    local batch_size="$5"
    local model="$6"
    shift 6

    local log="${OUTPUT_ROOT}/logs/${method}.shard-${shard_index}-of-${num_shards}.log"
    local attempt=1
    while (( attempt <= MAX_ATTEMPTS )); do
        echo "[m2t-suite] ${method} shard ${shard_index}/${num_shards} attempt ${attempt}" | tee -a "${log}"
        if CUDA_VISIBLE_DEVICES="${gpu}" python3 "${ROOT}/tools/run_m2t_humanml3d.py" \
            --method "${method}" \
            --model "${model}" \
            --data-root "${DATA_ROOT}" \
            --protocol-manifest "${PROTOCOL_MANIFEST}" \
            --output-dir "${OUTPUT_ROOT}/${method}" \
            --num-shards "${num_shards}" \
            --shard-index "${shard_index}" \
            --batch-size "${batch_size}" \
            "$@" >>"${log}" 2>&1; then
            echo "[m2t-suite] ${method} shard ${shard_index}/${num_shards} complete" | tee -a "${log}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 30
    done
    echo "[m2t-suite] ${method} shard ${shard_index}/${num_shards} failed" | tee -a "${log}"
    return 1
}

pids=()
run_shard 0 motiongpt 1 0 32 "${MOTIONGPT_MODEL}" & pids+=("$!")
run_shard 1 motiongpt3 1 0 4 "${MOTIONGPT3_MODEL}" & pids+=("$!")
run_shard 2 tm2t 2 0 32 "${TM2T_MODEL}" & pids+=("$!")
run_shard 3 tm2t 2 1 32 "${TM2T_MODEL}" & pids+=("$!")
run_shard 4 vermo 4 0 1 "${VERMO_MODEL}" --smpl-model-dir "${SMPL_MODEL_DIR}" & pids+=("$!")
run_shard 5 vermo 4 1 1 "${VERMO_MODEL}" --smpl-model-dir "${SMPL_MODEL_DIR}" & pids+=("$!")
run_shard 6 vermo 4 2 1 "${VERMO_MODEL}" --smpl-model-dir "${SMPL_MODEL_DIR}" & pids+=("$!")
run_shard 7 vermo 4 3 1 "${VERMO_MODEL}" --smpl-model-dir "${SMPL_MODEL_DIR}" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
