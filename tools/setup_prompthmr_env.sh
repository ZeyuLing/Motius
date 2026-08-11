#!/usr/bin/env bash
# Create the isolated runtime for the pinned official PromptHMR-Video release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="3b566b7dbb28ce506c7ea972c18693f4c705ce8c"
REPOSITORY="https://github.com/yufu-wang/PromptHMR.git"
UPSTREAM_DIR="${UPSTREAM_DIR:-${ROOT}/outputs/tmp/prompthmr/upstream}"
PT_VERSION="${PT_VERSION:-2.4}"
WORLD_VIDEO="${WORLD_VIDEO:-true}"
DOWNLOAD_VIDEO_CHECKPOINT="${DOWNLOAD_VIDEO_CHECKPOINT:-false}"
VIDEO_CHECKPOINT="${VIDEO_CHECKPOINT:-bedlam1+2}"
ENV_NAME="phmr_pt${PT_VERSION}"

if [[ "${PROMPTHMR_ACCEPT_LICENSE:-}" != "1" ]]; then
  echo "PromptHMR is non-commercial research software."
  echo "Review its LICENSE, then rerun with PROMPTHMR_ACCEPT_LICENSE=1."
  exit 2
fi
if [[ "${PT_VERSION}" != "2.4" && "${PT_VERSION}" != "2.6" ]]; then
  echo "PT_VERSION must be 2.4 or 2.6."
  exit 2
fi
if [[ "${WORLD_VIDEO}" != "true" && "${WORLD_VIDEO}" != "false" ]]; then
  echo "WORLD_VIDEO must be true or false."
  exit 2
fi

command -v conda >/dev/null
command -v git >/dev/null

mkdir -p "$(dirname "${UPSTREAM_DIR}")"
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  if [[ -e "${UPSTREAM_DIR}" ]]; then
    echo "${UPSTREAM_DIR} exists but is not a git checkout."
    exit 1
  fi
  git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${REVISION}"
  git -C "${UPSTREAM_DIR}" checkout --detach "${REVISION}"
else
  python3 "${ROOT}/tools/patch_prompthmr_runtime.py" \
    --runtime-root "${UPSTREAM_DIR}" \
    --restore
  if ! git -C "${UPSTREAM_DIR}" diff --quiet ||
    ! git -C "${UPSTREAM_DIR}" diff --cached --quiet; then
    echo "Refusing to change tracked files in ${UPSTREAM_DIR}."
    exit 1
  fi
  git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${REVISION}"
  if [[ "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)" != "${REVISION}" ]]; then
    git -C "${UPSTREAM_DIR}" checkout --detach "${REVISION}"
  fi
fi

if ! conda run -n "${ENV_NAME}" python -c "import torch" >/dev/null 2>&1; then
  (
    cd "${UPSTREAM_DIR}"
    bash scripts/install.sh \
      --pt_version="${PT_VERSION}" \
      --world-video="${WORLD_VIDEO}"
  )
fi

# The upstream installer does not use `set -e`: if Open3D 0.19 is unavailable
# from the image's package index, pip aborts the whole requirements transaction
# and the installer silently continues. Install the pinned requirements
# explicitly, substituting the available Open3D release; the video path does
# not depend on APIs added in 0.19.
FILTERED_REQUIREMENTS="/tmp/prompthmr_requirements_without_open3d.txt"
python - "${UPSTREAM_DIR}/requirements.txt" "${FILTERED_REQUIREMENTS}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.write_text(
    "".join(
        line for line in source.read_text().splitlines(keepends=True)
        if not line.strip().startswith("open3d==")
    )
)
PY
conda run -n "${ENV_NAME}" python -m pip install \
  -r "${FILTERED_REQUIREMENTS}"
conda run -n "${ENV_NAME}" python -m pip install open3d==0.18.0

# These runtime imports are used by official scripts but are not listed in the
# pinned upstream requirements.
conda run -n "${ENV_NAME}" python -m pip install \
  joblib tyro imageio-ffmpeg

python3 "${ROOT}/tools/patch_prompthmr_runtime.py" \
  --runtime-root "${UPSTREAM_DIR}"

if [[ "${DOWNLOAD_VIDEO_CHECKPOINT}" == "true" ]]; then
  case "${VIDEO_CHECKPOINT}" in
    bedlam1|b1)
      FILENAME="phmr_b1.ckpt"
      EXPECTED_SHA256="d06ae5ddc74ef74c252f4ec34e4e3092cd8fc18cba104af5aa978cdd2c669b5a"
      ;;
    bedlam1+2|b1b2)
      FILENAME="phmr_b1b2.ckpt"
      EXPECTED_SHA256="2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5"
      ;;
    bedlam2|b2)
      FILENAME="phmr_b2.ckpt"
      EXPECTED_SHA256="631433bf4dfd548dc5c6e2df037e11a11ce4a83c37367ee0f31b2f1627aa06d9"
      ;;
    *)
      echo "Unknown VIDEO_CHECKPOINT=${VIDEO_CHECKPOINT}."
      exit 2
      ;;
  esac
  CHECKPOINT_DIR="${UPSTREAM_DIR}/data/pretrain/phmr_vid"
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/${FILENAME}"
  mkdir -p "${CHECKPOINT_DIR}"
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    if command -v gdown >/dev/null 2>&1; then
      case "${FILENAME}" in
        phmr_b1.ckpt) DRIVE_ID="1Q1ZhDYhPNMzg0lU4zN9JoV7MCSqfMvED" ;;
        phmr_b1b2.ckpt) DRIVE_ID="1a0IqV5IBA5L0H9BYbLRVLm3KwQDPXFO0" ;;
        phmr_b2.ckpt) DRIVE_ID="13yrY33o_iB27XCiekDkxogExAi2fG2rL" ;;
      esac
      gdown "${DRIVE_ID}" --output "${CHECKPOINT_PATH}"
    else
      curl --fail --location \
        --output "${CHECKPOINT_PATH}" \
        "https://download.is.tue.mpg.de/bedlam2/ml/videos/${FILENAME}"
    fi
  fi
  ACTUAL_SHA256="$(sha256sum "${CHECKPOINT_PATH}" | awk '{print $1}')"
  if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "SHA256 mismatch for ${CHECKPOINT_PATH}."
    exit 1
  fi
fi

echo "PromptHMR source: ${UPSTREAM_DIR}"
echo "Pinned revision: ${REVISION}"
echo "Applied audited device-transfer patch: pipeline/utils_detectron2.py"
echo "Conda environment: ${ENV_NAME}"
echo "The image model, SMPL-X files, and third-party checkpoints are not"
echo "redistributed. Obtain them with the official licensed data scripts."
