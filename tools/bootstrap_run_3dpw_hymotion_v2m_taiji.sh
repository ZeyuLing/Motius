#!/usr/bin/env bash
set -euo pipefail

ROOT="${MOTIUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU_COUNT="${1:-8}"
OUTPUT_METHOD="${MOTIUS_OUTPUT_METHOD:-hymotion_v2m_gtcrop_v1}"
ENV_ROOT="/tmp/motius_hymotion_v2m_env"
SAM3D_ROOT="/tmp/motius_sam_3d_body"
YOLOX_ROOT="/tmp/motius_yolox"
STAGE_ROOT="/tmp/motius_3dpw_eval"
ASSET_CACHE="/tmp/motius_hymotion_v2m_assets"
OUTPUT_ROOT="${ROOT}/outputs/evaluation/monocular_capture/3dpw_test"
METHOD_ROOT="${OUTPUT_ROOT}/${OUTPUT_METHOD}"
TARGET_INDEX="${ROOT}/outputs/evaluation/monocular_capture/ground_truth/3dpw_joint_only/ground_truth_index.json"
SHARD_PLAN="${METHOD_ROOT}/shard_plan.json"
OCCUPY_PID_FILE="/tmp/motius_setup_occupy.pid"
OCCUPY_PYTHON="$(command -v python3)"

SAM3D_REVISION="b5c765a0d89d789985e186d396315e7590887b94"
YOLOX_REVISION="e1052df71842031413f6030723c3607b839c80ce"
DINOV3_REVISION="6876159a11b4df116f30f667f8c9888617df0751"
HYMOTION_CKPT="${ROOT}/checkpoints/hymotion_v2m/epoch100.ckpt"
HYMOTION_MEAN_STD="${ROOT}/checkpoints/hymotion_v2m/mean_std.json"
SMPLH_MODEL="${ROOT}/checkpoints/body_models/smplh/neutral/model.npz"
SAM3D_CKPT="${ROOT}/checkpoints/hymotion_v2m/sam3d/sam3d_body.ckpt"
SAM3D_MHR="${ROOT}/checkpoints/hymotion_v2m/sam3d/mhr_model.pt"
YOLOX_CKPT="${ROOT}/checkpoints/hymotion_v2m/yolox/yolox_l.pth"

cd "${ROOT}"
mkdir -p "${METHOD_ROOT}/logs"
exec > >(tee -a "${METHOD_ROOT}/logs/bootstrap.log") 2>&1
echo "[$(date -Is)] bootstrap method=hymotion_v2m output_method=${OUTPUT_METHOD} gpus=${GPU_COUNT}"
if [[ -f "${OCCUPY_PID_FILE}" ]]; then
  tracked_pid="$(<"${OCCUPY_PID_FILE}")"
  if [[ "${tracked_pid}" =~ ^[0-9]+$ ]] && kill -0 "${tracked_pid}" 2>/dev/null; then
    kill -- "-${tracked_pid}" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 "${tracked_pid}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${tracked_pid}" 2>/dev/null; then
      kill -9 -- "-${tracked_pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${OCCUPY_PID_FILE}"
fi
setsid "${OCCUPY_PYTHON}" "${ROOT}/../occupy.py" \
  --gpus all \
  --mem-frac-of-free 0.7 \
  --reserve-gib 8 \
  --duration-s 21600 \
  --report-every-s 60 \
  >"${METHOD_ROOT}/logs/setup_occupy.log" 2>&1 &
OCCUPY_PID="$!"
echo "${OCCUPY_PID}" >"${OCCUPY_PID_FILE}"
cleanup_occupy() {
  if kill -0 "${OCCUPY_PID}" 2>/dev/null; then
    kill -- "-${OCCUPY_PID}" 2>/dev/null || true
    wait "${OCCUPY_PID}" 2>/dev/null || true
  fi
  rm -f "${OCCUPY_PID_FILE}"
  if [[ "${MOTIUS_RESTART_POOL_OCCUPY:-1}" == "1" ]]; then
    setsid env -u PYTHONPATH "${OCCUPY_PYTHON}" "${ROOT}/../occupy.py" \
      --gpus all \
      --mem-frac-of-free 0.7 \
      --reserve-gib 1 \
      --duration-s 0 \
      --report-every-s 60 \
      >"${METHOD_ROOT}/logs/pool_occupy_after.log" \
      2>&1 < /dev/null &
    echo "$!" >"${OCCUPY_PID_FILE}"
  fi
}
trap cleanup_occupy EXIT INT TERM
sleep 5
kill -0 "${OCCUPY_PID}"

declare -A ASSET_SIZES=(
  ["${HYMOTION_CKPT}"]=763342207
  ["${HYMOTION_MEAN_STD}"]=57462
  ["${SMPLH_MODEL}"]=87259106
  ["${SAM3D_CKPT}"]=2109129346
  ["${SAM3D_MHR}"]=696110248
  ["${YOLOX_CKPT}"]=434357141
)
for asset in \
  "${HYMOTION_CKPT}" \
  "${HYMOTION_MEAN_STD}" \
  "${SMPLH_MODEL}" \
  "${SAM3D_CKPT}" \
  "${SAM3D_MHR}" \
  "${YOLOX_CKPT}"; do
  for _ in {1..360}; do
    if [[ -f "${asset}" ]] && \
       [[ "$(stat -Lc %s "${asset}")" == "${ASSET_SIZES[${asset}]}" ]]; then
      break
    fi
    sleep 10
  done
  if [[ ! -f "${asset}" ]] || \
     [[ "$(stat -Lc %s "${asset}")" != "${ASSET_SIZES[${asset}]}" ]]; then
    echo "Timed out waiting one hour for complete HYMotion-V2M asset: ${asset}" >&2
    exit 3
  fi
done
stage_asset() {
  local source="$1"
  local destination="$2"
  local expected_size="$3"
  if [[ -f "${destination}" ]] && \
     [[ "$(stat -Lc %s "${destination}")" == "${expected_size}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  local temporary="${destination}.partial.$$"
  rm -f "${temporary}"
  cp --reflink=auto "${source}" "${temporary}"
  if [[ "$(stat -Lc %s "${temporary}")" != "${expected_size}" ]]; then
    echo "Staged HYMotion asset has the wrong size: ${temporary}" >&2
    rm -f "${temporary}"
    exit 6
  fi
  mv -f "${temporary}" "${destination}"
}

RUNTIME_CHECKPOINT_DIR="${ASSET_CACHE}/checkpoint"
RUNTIME_HYMOTION_CKPT="${RUNTIME_CHECKPOINT_DIR}/epoch100.ckpt"
RUNTIME_HYMOTION_MEAN_STD="${RUNTIME_CHECKPOINT_DIR}/mean_std.json"
RUNTIME_SMPLH_MODEL="${ASSET_CACHE}/body_models/smplh/neutral/model.npz"
RUNTIME_SAM3D_CKPT="${ASSET_CACHE}/sam3d/sam3d_body.ckpt"
RUNTIME_SAM3D_MHR="${ASSET_CACHE}/sam3d/mhr_model.pt"
RUNTIME_YOLOX_CKPT="${ASSET_CACHE}/yolox/yolox_l.pth"
stage_asset \
  "${ROOT}/checkpoints/hymotion_v2m/config.yml" \
  "${RUNTIME_CHECKPOINT_DIR}/config.yml" \
  "$(stat -Lc %s "${ROOT}/checkpoints/hymotion_v2m/config.yml")"
stage_asset \
  "${HYMOTION_CKPT}" "${RUNTIME_HYMOTION_CKPT}" \
  "${ASSET_SIZES[${HYMOTION_CKPT}]}"
stage_asset \
  "${HYMOTION_MEAN_STD}" "${RUNTIME_HYMOTION_MEAN_STD}" \
  "${ASSET_SIZES[${HYMOTION_MEAN_STD}]}"
stage_asset \
  "${SMPLH_MODEL}" "${RUNTIME_SMPLH_MODEL}" \
  "${ASSET_SIZES[${SMPLH_MODEL}]}"
stage_asset \
  "${ROOT}/checkpoints/hymotion_v2m/sam3d/model_config.yaml" \
  "${ASSET_CACHE}/model_config.yaml" \
  "$(stat -Lc %s "${ROOT}/checkpoints/hymotion_v2m/sam3d/model_config.yaml")"
stage_asset \
  "${SAM3D_CKPT}" "${RUNTIME_SAM3D_CKPT}" \
  "${ASSET_SIZES[${SAM3D_CKPT}]}"
stage_asset \
  "${SAM3D_MHR}" "${RUNTIME_SAM3D_MHR}" \
  "${ASSET_SIZES[${SAM3D_MHR}]}"
stage_asset \
  "${YOLOX_CKPT}" "${RUNTIME_YOLOX_CKPT}" \
  "${ASSET_SIZES[${YOLOX_CKPT}]}"
echo "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf  ${RUNTIME_SAM3D_CKPT}" |
  sha256sum --check -
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  ${RUNTIME_SAM3D_MHR}" |
  sha256sum --check -

if command -v yum >/dev/null 2>&1; then
  toolchain_ready=0
  for attempt in 1 2 3; do
    if yum install -y \
      python3.11-devel \
      gcc-toolset-11-gcc \
      gcc-toolset-11-gcc-c++ \
      ninja-build; then
      toolchain_ready=1
      break
    fi
    echo "Toolchain installation attempt ${attempt} failed; retrying." >&2
    sleep 5
  done
  if [[ "${toolchain_ready}" != "1" ]]; then
    echo "Failed to install the HYMotion build toolchain after 3 attempts." >&2
    exit 5
  fi
  # shellcheck disable=SC1091
  set +u
  source /opt/rh/gcc-toolset-11/enable
  set -u
  export CC=gcc CXX=g++
  command -v gcc >/dev/null
  "${CC}" --version
  export MAX_JOBS="${MAX_JOBS:-8}"
fi

python3 -m pip install --quiet --upgrade uv
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  uv venv "${ENV_ROOT}" --python 3.11
fi
# shellcheck disable=SC1091
source "${ENV_ROOT}/bin/activate"
uv pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install \
  "numpy<2" mmengine omegaconf scipy scikit-learn einops smplx \
  opencv-python supervision imageio-ffmpeg torchdiffeq pyyaml tqdm \
  pytorch-lightning pyrender yacs scikit-image timm dill pandas rich \
  hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset \
  networkx==3.2.1 roma joblib seaborn wandb appdirs ffmpeg cython \
  jsonlines xtcocotools loguru optree fvcore pycocotools tensorboard \
  huggingface_hub thop
uv pip install pip setuptools
uv pip install chumpy --no-build-isolation
uv pip install \
  "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" \
  --no-build-isolation --no-deps

prepare_checkout() {
  local repository="$1"
  local revision="$2"
  local destination="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    rm -rf "${destination}"
    git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
}
prepare_checkout \
  https://github.com/facebookresearch/sam-3d-body.git \
  "${SAM3D_REVISION}" \
  "${SAM3D_ROOT}"
prepare_checkout \
  https://github.com/Megvii-BaseDetection/YOLOX.git \
  "${YOLOX_REVISION}" \
  "${YOLOX_ROOT}"
prepare_checkout \
  https://github.com/facebookresearch/dinov3.git \
  "${DINOV3_REVISION}" \
  "${HOME}/.cache/torch/hub/facebookresearch_dinov3_main"
test -f \
  "${HOME}/.cache/torch/hub/facebookresearch_dinov3_main/hubconf.py"
test -f \
  "${HOME}/.cache/torch/hub/facebookresearch_dinov3_main/dinov3/eval/detection/models/transformer.py"
uv pip install -e "${YOLOX_ROOT}" --no-build-isolation

tools/stage_3dpw_test_data.sh "${STAGE_ROOT}"
"${ENV_ROOT}/bin/python" tools/materialize_3dpw_test_videos.py \
  --data-root "${STAGE_ROOT}" \
  --output-dir "${STAGE_ROOT}/videos"
HYMOTION_CKPT_SHA256="$(sha256sum "${RUNTIME_HYMOTION_CKPT}" | awk '{print $1}')"
PLAN_ARGS=()
if [[ -n "${MOTIUS_MAX_SEQUENCES:-}" ]]; then
  PLAN_ARGS+=(--max-sequences "${MOTIUS_MAX_SEQUENCES}")
fi
"${ENV_ROOT}/bin/python" tools/build_3dpw_monocular_shard_plan.py \
  --video-manifest "${STAGE_ROOT}/videos/manifest.json" \
  --ground-truth-index "${TARGET_INDEX}" \
  --prediction-dir "${METHOD_ROOT}/predictions" \
  --num-shards "${GPU_COUNT}" \
  --output "${SHARD_PLAN}" \
  "${PLAN_ARGS[@]}"
PENDING_SEQUENCES="$(
  "${ENV_ROOT}/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["pending"])' \
    "${SHARD_PLAN}"
)"
if [[ "${PENDING_SEQUENCES}" == "0" ]]; then
  echo "All requested HYMotion-V2M sequences already have predictions."
  exit 0
fi

kill -- "-${OCCUPY_PID}"
for _ in {1..60}; do
  kill -0 "${OCCUPY_PID}" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "${OCCUPY_PID}" 2>/dev/null; then
  kill -9 -- "-${OCCUPY_PID}"
fi
rm -f "${OCCUPY_PID_FILE}"

pids=()
cleanup_shards() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_shards INT TERM
FFMPEG_BIN="$("${ENV_ROOT}/bin/python" -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
MAX_SEQUENCE_ARGS=()
if [[ -n "${MOTIUS_MAX_SEQUENCES:-}" ]]; then
  MAX_SEQUENCE_ARGS+=(--max-sequences "${MOTIUS_MAX_SEQUENCES}")
fi

for ((rank = 0; rank < GPU_COUNT; rank++)); do
  CUDA_VISIBLE_DEVICES="${rank}" \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  PYTHONPATH="${ROOT}" \
  "${ENV_ROOT}/bin/python" tools/run_3dpw_hymotion_v2m_shard.py \
    --checkpoint-dir "${RUNTIME_CHECKPOINT_DIR}" \
    --mean-std-path "${RUNTIME_HYMOTION_MEAN_STD}" \
    --body-model-path "${RUNTIME_SMPLH_MODEL}" \
    --checkpoint-sha256 "${HYMOTION_CKPT_SHA256}" \
    --video-manifest "${STAGE_ROOT}/videos/manifest.json" \
    --video-dir "${STAGE_ROOT}/videos" \
    --output-root "${OUTPUT_ROOT}" \
    --output-method "${OUTPUT_METHOD}" \
    --assignment-plan "${SHARD_PLAN}" \
    --sam3d-repo "${SAM3D_ROOT}" \
    --sam3d-checkpoint "${RUNTIME_SAM3D_CKPT}" \
    --sam3d-mhr "${RUNTIME_SAM3D_MHR}" \
    --yolox-checkpoint "${RUNTIME_YOLOX_CKPT}" \
    --ffmpeg "${FFMPEG_BIN}" \
    --shard-id "${rank}" \
    --num-shards "${GPU_COUNT}" \
    "${MAX_SEQUENCE_ARGS[@]}" \
    >"${METHOD_ROOT}/logs/shard_${rank}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - INT TERM EXIT

METHOD_ROOT="${METHOD_ROOT}" FAILED="${failed}" GPU_COUNT="${GPU_COUNT}" \
"${ENV_ROOT}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["METHOD_ROOT"])
records = [
    json.loads(path.read_text())
    for path in sorted((root / "status").glob("*.json"))
]
payload = {
    "method": "hymotion_v2m",
    "output_method": root.name,
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "shards_failed": bool(int(os.environ["FAILED"])),
    "complete": sum(item.get("status") == "complete" for item in records),
    "failed": sum(item.get("status") == "failed" for item in records),
    "population": len(records),
}
(root / "finish.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY

if [[ "${MOTIUS_RESTART_POOL_OCCUPY:-1}" == "1" ]]; then
  setsid env -u PYTHONPATH "${OCCUPY_PYTHON}" "${ROOT}/../occupy.py" \
    --gpus all \
    --mem-frac-of-free 0.7 \
    --reserve-gib 1 \
    --duration-s 0 \
    --report-every-s 60 \
    >"${METHOD_ROOT}/logs/pool_occupy_after.log" 2>&1 < /dev/null &
  echo "$!" >"${OCCUPY_PID_FILE}"
fi

exit "${failed}"
