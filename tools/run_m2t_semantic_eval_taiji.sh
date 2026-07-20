#!/usr/bin/env bash
set -euo pipefail

# Recompute the shared text-to-motion retrieval protocol without repeating
# BERTScore. One method is assigned to each GPU.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/m2t/humanml3d/full_v1}"
MOTIONGPT_DIR="${MOTIONGPT_DIR:-${ROOT}/outputs/m2t/humanml3d/motiongpt_official_batch32_v2}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${ROOT}/outputs/m2t/humanml3d/protocol_manifest.json}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/HumanML3D}"
LOCAL_EVALUATOR="${LOCAL_EVALUATOR:-/dev/shm/motius_m2t_semantic_tm2t}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

rm -rf "${LOCAL_EVALUATOR}"
mkdir -p "${LOCAL_EVALUATOR}" "${OUTPUT_ROOT}/eval_logs"
cp -aL "${ROOT}/checkpoints/tm2t/." "${LOCAL_EVALUATOR}/"

python3 -m pip install --disable-pip-version-check --no-cache-dir \
    'spacy==3.8.7' 'pycocoevalcap==1.2' 'click==8.2.1' \
    'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'

methods=(tm2t motiongpt motiongpt3 vermo)
directories=(
    "${OUTPUT_ROOT}/tm2t"
    "${MOTIONGPT_DIR}"
    "${OUTPUT_ROOT}/motiongpt3"
    "${OUTPUT_ROOT}/vermo"
)

pids=()
for gpu in 0 1 2 3; do
    method="${methods[$gpu]}"
    directory="${directories[$gpu]}"
    log="${OUTPUT_ROOT}/eval_logs/${method}_semantic_fixed.log"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${ROOT}/tools/eval_m2t_humanml3d.py" \
        --prediction-dir "${directory}" \
        --protocol-manifest "${PROTOCOL_MANIFEST}" \
        --data-root "${DATA_ROOT}" \
        --output "${directory}/metrics_semantic_fixed.json" \
        --semantic-artifact "${LOCAL_EVALUATOR}" \
        --semantic-device cuda \
        --chunk-size 32 \
        --n-repeats 1 \
        --io-workers 64 \
        --no-bertscore >"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if ! CUDA_VISIBLE_DEVICES=1 python3 "${ROOT}/tools/eval_m2t_humanml3d.py" \
    --prediction-dir "${OUTPUT_ROOT}/motiongpt" \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT_ROOT}/motiongpt/metrics_semantic_fixed.json" \
    --semantic-artifact "${LOCAL_EVALUATOR}" \
    --semantic-device cuda \
    --chunk-size 32 \
    --n-repeats 1 \
    --io-workers 64 \
    --no-bertscore >"${OUTPUT_ROOT}/eval_logs/motiongpt_stable_semantic_fixed.log" 2>&1; then
    status=1
fi
exit "${status}"
