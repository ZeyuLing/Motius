#!/usr/bin/env bash
set -euo pipefail

REVISION="6ec3ca39336c50492c0fae65fba2fb831fc7d866"
REPOSITORY="https://github.com/zju3dv/GVHMR.git"
MOTIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${1:-${MOTIUS_ROOT}/outputs/tmp/gvhmr/upstream}"
ENV_PREFIX="${2:-${MOTIUS_ROOT}/outputs/tmp/gvhmr/conda-env}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required by the official GVHMR installation instructions." >&2
  exit 1
fi

NEW_RUNTIME=0
if [[ ! -d "${RUNTIME_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${RUNTIME_ROOT}")"
  git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${RUNTIME_ROOT}"
  NEW_RUNTIME=1
fi

if [[ "${NEW_RUNTIME}" -eq 0 ]]; then
  python3 "${MOTIUS_ROOT}/tools/patch_gvhmr_runtime.py" \
    --runtime-root "${RUNTIME_ROOT}" \
    --restore
fi
if [[ "${NEW_RUNTIME}" -eq 0 && -n "$(git -C "${RUNTIME_ROOT}" status --short)" ]]; then
  echo "Refusing to change a dirty GVHMR runtime: ${RUNTIME_ROOT}" >&2
  exit 1
fi
git -C "${RUNTIME_ROOT}" fetch --depth 1 origin "${REVISION}"
git -C "${RUNTIME_ROOT}" checkout --detach "${REVISION}"
ACTUAL_REVISION="$(git -C "${RUNTIME_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_REVISION}" != "${REVISION}" ]]; then
  echo "GVHMR revision verification failed." >&2
  exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda run -p "${ENV_PREFIX}" python -m pip install --upgrade pip setuptools wheel
FILTERED_REQUIREMENTS="/tmp/motius_gvhmr_requirements.txt"
RUNTIME_ROOT="${RUNTIME_ROOT}" FILTERED_REQUIREMENTS="${FILTERED_REQUIREMENTS}" \
  python3 - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["RUNTIME_ROOT"]) / "requirements.txt"
destination = Path(os.environ["FILTERED_REQUIREMENTS"])
lines = [
    line
    for line in source.read_text().splitlines()
    if line.strip() and line.strip() != "chumpy"
]
destination.write_text("\n".join(lines) + "\n")
PY
conda run -p "${ENV_PREFIX}" python -m pip install -r "${FILTERED_REQUIREMENTS}"
conda run -p "${ENV_PREFIX}" python -m pip install \
  --no-build-isolation chumpy==0.70
conda run -p "${ENV_PREFIX}" python -m pip install imageio-ffmpeg mmengine
conda run -p "${ENV_PREFIX}" python -m pip install -e "${RUNTIME_ROOT}"
python3 "${MOTIUS_ROOT}/tools/patch_gvhmr_runtime.py" \
  --runtime-root "${RUNTIME_ROOT}"

CHECKPOINT="${RUNTIME_ROOT}/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
echo "Pinned GVHMR runtime: ${ACTUAL_REVISION}"
echo "Set MOTIUS_GVHMR_ROOT=${RUNTIME_ROOT}"
echo "Set MOTIUS_GVHMR_PYTHON=${ENV_PREFIX}/bin/python"
if [[ -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint SHA256: $(sha256sum "${CHECKPOINT}" | cut -d ' ' -f 1)"
else
  echo "Checkpoint not installed: ${CHECKPOINT}"
  echo "Download it from the Google Drive folder linked by GVHMR docs/INSTALL.md."
fi
echo "SMPL-X and SMPL files must also follow the upstream inputs/checkpoints layout."
