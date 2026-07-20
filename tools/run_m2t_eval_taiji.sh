#!/usr/bin/env bash
set -euo pipefail

# Evaluate all HumanML3D M2T baselines on one four-GPU Taiji worker. Each
# method gets one GPU; linguistic and semantic metrics share protocol v2.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/m2t/humanml3d/full_v1}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${ROOT}/outputs/m2t/humanml3d/protocol_manifest.json}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/HumanML3D}"
LOCAL_ROOT="${LOCAL_ROOT:-/dev/shm/motius_m2t_eval}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${LOCAL_ROOT}/hf"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${LOCAL_ROOT}" "${OUTPUT_ROOT}/eval_logs"

python3 -m pip install --disable-pip-version-check --no-cache-dir \
    'spacy==3.8.7' 'bert-score==0.3.13' 'pycocoevalcap==1.2' \
    'click==8.2.1' \
    'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'

python3 - <<'PY'
import spacy
import torch
import transformers
import bert_score
import pycocoevalcap

spacy.load("en_core_web_sm")
assert torch.__version__.startswith("2.5.0"), torch.__version__
assert transformers.__version__.startswith("4.53."), transformers.__version__
print("[m2t-eval] dependencies verified", flush=True)
PY

if [[ ! -f "${LOCAL_ROOT}/tm2t/.motius-stage-complete" ]]; then
    rm -rf "${LOCAL_ROOT}/tm2t"
    mkdir -p "${LOCAL_ROOT}/tm2t"
    cp -a "${ROOT}/checkpoints/tm2t/." "${LOCAL_ROOT}/tm2t/"
    touch "${LOCAL_ROOT}/tm2t/.motius-stage-complete"
fi

# Download once before four BERTScore workers start, avoiding concurrent Hub
# locks and duplicate network transfers.
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("FacebookAI/roberta-large")
print("[m2t-eval] roberta-large cached", flush=True)
PY

methods=(motiongpt motiongpt3 tm2t vermo)
pids=()
for gpu in 0 1 2 3; do
    method="${methods[$gpu]}"
    log="${OUTPUT_ROOT}/eval_logs/${method}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${ROOT}/tools/eval_m2t_humanml3d.py" \
        --prediction-dir "${OUTPUT_ROOT}/${method}" \
        --protocol-manifest "${PROTOCOL_MANIFEST}" \
        --data-root "${DATA_ROOT}" \
        --output "${OUTPUT_ROOT}/${method}/metrics.json" \
        --semantic-artifact "${LOCAL_ROOT}/tm2t" \
        --semantic-device cuda \
        --bert-device cuda \
        --chunk-size 32 \
        --n-repeats 1 \
        --io-workers 64 >"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if ! CUDA_VISIBLE_DEVICES=0 python3 "${ROOT}/tools/eval_m2t_humanml3d.py" \
    --gt-from-protocol \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT_ROOT}/gt/metrics.json" \
    --semantic-artifact "${LOCAL_ROOT}/tm2t" \
    --semantic-device cuda \
    --bert-device cuda \
    --chunk-size 32 \
    --n-repeats 1 \
    --io-workers 64 >"${OUTPUT_ROOT}/eval_logs/gt.log" 2>&1; then
    status=1
fi
exit "${status}"
