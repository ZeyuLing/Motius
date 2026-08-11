#!/usr/bin/env bash
set -euo pipefail

MOTIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-${MOTIUS_ROOT}/outputs/envs/gvhmr}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  mkdir -p "$(dirname "${ENV_PREFIX}")"
  if command -v conda >/dev/null 2>&1; then
    conda create -y -p "${ENV_PREFIX}" python=3.10
  else
    PYTHON_BOOTSTRAP="${PYTHON3_10:-$(command -v python3.10 || true)}"
    if [[ -z "${PYTHON_BOOTSTRAP}" ]]; then
      echo "Python 3.10 or conda is required for the GVHMR environment." >&2
      exit 1
    fi
    "${PYTHON_BOOTSTRAP}" -m venv "${ENV_PREFIX}"
  fi
fi

PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --upgrade pip wheel setuptools==79.0.1
"${PYTHON}" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.0+cu121 torchvision==0.18.0+cu121
"${PYTHON}" -m pip install \
  "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"
"${PYTHON}" -m pip install --no-build-isolation chumpy==0.70
"${PYTHON}" -m pip install \
  av==13.0.0 braceexpand cython-bbox einops ffmpeg-python \
  hydra-colorlog hydra-core hydra-zen imageio==2.34.1 joblib lapx \
  lightning==2.3.0 pytorch-lightning==2.3.0 loguru mmengine numpy==1.23.5 \
  opencv-python==4.10.0.84 pycolmap==0.6.1 rich scikit-image smplx \
  supervision==0.20.0 tensorboardX termcolor thop timm==0.9.12 \
  trimesh ultralytics==8.2.42 wis3d==1.0.1 huggingface_hub
"${PYTHON}" -m pip install --no-deps -e "${MOTIUS_ROOT}"

echo "GVHMR environment: ${ENV_PREFIX}"
echo "Set MOTIUS_GVHMR_PYTHON=${PYTHON}"
echo "No external GVHMR source checkout is required."
