#!/usr/bin/env bash
set -euo pipefail

# Reproduce MotionGPT's released batch-32 M2T inference while retaining
# resumable, sample-addressed outputs. Batch groups are kept intact per shard.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/HumanML3D}"
MODEL="${MODEL:-ZeyuLing/Motius-MotionGPT-HumanML3D}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/m2t/humanml3d/motiongpt_official_batch32_v2}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${ROOT}/outputs/m2t/humanml3d/protocol_manifest.json}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}/logs"
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[motiongpt:m2t] HumanML3D not found: ${DATA_ROOT}" >&2
    echo "Set DATA_ROOT to the official HumanML3D directory." >&2
    exit 2
fi

run_shard() {
    local shard="$1"
    local log="${OUTPUT_DIR}/logs/shard-${shard}-of-${NUM_SHARDS}.log"
    local attempt=1
    while (( attempt <= MAX_ATTEMPTS )); do
        echo "[motiongpt:m2t] shard ${shard}/${NUM_SHARDS} attempt ${attempt}" | tee -a "${log}"
        if CUDA_VISIBLE_DEVICES="${shard}" python3 "${ROOT}/tools/run_m2t_humanml3d.py" \
            --method motiongpt \
            --model "${MODEL}" \
            --data-root "${DATA_ROOT}" \
            --protocol-manifest "${PROTOCOL_MANIFEST}" \
            --output-dir "${OUTPUT_DIR}" \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${shard}" \
            --shard-mode batch_group \
            --motiongpt-official-batch-padding \
            --batch-size 32 >>"${log}" 2>&1; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 20
    done
    return 1
}

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    run_shard "${shard}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
