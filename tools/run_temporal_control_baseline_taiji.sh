#!/usr/bin/env bash
set -euo pipefail

# Generate all eight HumanML3D Temporal Motion Completion tracks for one baseline.
# Taiji sets INDEX to the host rank. Each host launches one shard per local GPU.

METHOD="${METHOD:?Set METHOD=maskcontrol or METHOD=omnicontrol}"
if [[ "${METHOD}" != maskcontrol && "${METHOD}" != omnicontrol ]]; then
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 2
fi

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/Motius}"
HFTRAINER_ROOT="${HFTRAINER_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/outputs/evaluation/temporal_baselines/shared/hml263_gt}"
PROTOCOL_FILE="${PROTOCOL_FILE:-${HFTRAINER_ROOT}/data/eval/m2m_v2/eval_hml3d_official_control_4012.json}"
KEYFRAME_FILE="${KEYFRAME_FILE:-${HFTRAINER_ROOT}/data/eval/m2m_v2/eval_hml3d_official_adaptive_keyframes_4012.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/evaluation/temporal_baselines/${METHOD}}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-${INDEX:-}}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
LOCAL_ROOT="${LOCAL_ROOT:-/dev/shm/motius_temporal_${METHOD}_${NODE_RANK:-unknown}}"

if [[ -z "${NODE_RANK}" ]]; then
    echo "Missing Taiji INDEX/NODE_RANK; refusing to duplicate shards." >&2
    exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "Invalid node rank ${NODE_RANK}/${NUM_NODES}" >&2
    exit 2
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${LOCAL_ROOT}/hf"
mkdir -p "${LOCAL_ROOT}" "${OUTPUT_ROOT}/logs"

python3 -m pip install --disable-pip-version-check -q -e "${ROOT}[${METHOD}]"

if [[ "${METHOD}" == maskcontrol ]]; then
    ARTIFACT="${ARTIFACT:-ZeyuLing/motius-maskcontrol-humanml3d}"
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${ARTIFACT}")
print("[prefetch] MaskControl artifact ready", flush=True)
PY
    RUNNER="${ROOT}/tools/eval_maskcontrol_temporal_humanml3d.py"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    METHOD_ARGS=(--artifact "${ARTIFACT}" --optimization-profile paper)
else
    ARTIFACT="${ARTIFACT:-${ROOT}/checkpoints/omnicontrol/extracted/omnicontrol_ckpt/model_humanml3d.pt}"
    python3 - <<'PY'
import clip

clip.load("ViT-B/32", device="cpu")
print("[prefetch] OpenAI CLIP artifact ready", flush=True)
PY
    RUNNER="${ROOT}/tools/eval_omnicontrol_temporal_humanml3d.py"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    METHOD_ARGS=(--artifact "${ARTIFACT}" --guidance 2.5)
fi

if [[ ! -d "${INPUT_DIR}" || ! -f "${PROTOCOL_FILE}" ]]; then
    echo "Missing protocol input: ${INPUT_DIR} or ${PROTOCOL_FILE}" >&2
    exit 2
fi

TOTAL_SHARDS=$((NUM_NODES * NUM_GPUS))
SETTINGS=(
    "temporal_start_1f:start_1f:normal"
    "temporal_pre20:pre20:normal"
    "temporal_pre20_uncond:pre20:blank"
    "temporal_both_1f:both_1f:normal"
    "temporal_mid80:mid80:normal"
    "temporal_mid80_uncond:mid80:blank"
    "temporal_adaptive_keyframes:adaptive_keyframes:normal"
    "temporal_adaptive_keyframes_uncond:adaptive_keyframes:blank"
)

if [[ -n "${SETTING_IDS:-}" ]]; then
    requested=" ${SETTING_IDS//,/ } "
    filtered=()
    for spec in "${SETTINGS[@]}"; do
        setting_id="${spec%%:*}"
        if [[ "${requested}" == *" ${setting_id} "* ]]; then
            filtered+=("${spec}")
        fi
    done
    if (( ${#filtered[@]} == 0 )); then
        echo "SETTING_IDS did not match a supported setting: ${SETTING_IDS}" >&2
        exit 2
    fi
    SETTINGS=("${filtered[@]}")
fi

run_shard() {
    local gpu="$1"
    local global_shard="$2"
    local setting_id="$3"
    local setting="$4"
    local caption_mode="$5"
    local output="${OUTPUT_ROOT}/${setting_id}/hml263"
    local log="${OUTPUT_ROOT}/logs/${setting_id}.shard-${global_shard}-of-${TOTAL_SHARDS}.log"
    local keyframe_args=()
    if [[ "${setting}" == adaptive_keyframes ]]; then
        keyframe_args=(--keyframe-file "${KEYFRAME_FILE}")
    fi
    mkdir -p "${output}"
    echo "[start] ${METHOD} ${setting_id} shard=${global_shard}/${TOTAL_SHARDS}" >"${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${RUNNER}" \
        "${METHOD_ARGS[@]}" \
        --ids "${PROTOCOL_FILE}" \
        --captions "${PROTOCOL_FILE}" \
        --gt-hml263-dir "${INPUT_DIR}" \
        --out-dir "${output}" \
        --setting "${setting}" \
        --caption-mode "${caption_mode}" \
        --device cuda \
        --batch-size "${BATCH_SIZE}" \
        --num-shards "${TOTAL_SHARDS}" \
        --shard-index "${global_shard}" \
        --max-samples "${MAX_SAMPLES}" \
        --seed "${SEED}" \
        --skip-existing \
        "${keyframe_args[@]}" >>"${log}" 2>&1
}

for spec in "${SETTINGS[@]}"; do
    IFS=: read -r setting_id setting caption_mode <<<"${spec}"
    pids=()
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        global_shard=$((NODE_RANK * NUM_GPUS + gpu))
        run_shard "${gpu}" "${global_shard}" "${setting_id}" "${setting}" "${caption_mode}" &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    if (( status != 0 )); then
        echo "[failed] ${METHOD} ${setting_id} node=${NODE_RANK}" >&2
        exit "${status}"
    fi
    echo "[complete] ${METHOD} ${setting_id} node=${NODE_RANK}" | tee -a "${OUTPUT_ROOT}/logs/run.log"
done

date -Is >"${OUTPUT_ROOT}/host_${NODE_RANK}.done"
echo "[done] ${METHOD} host=${NODE_RANK}/${NUM_NODES}"
