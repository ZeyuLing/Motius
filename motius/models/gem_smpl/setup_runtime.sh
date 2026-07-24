#!/usr/bin/env bash
set -euo pipefail

REVISION="16bebf402d8893184249ee206d957b8248cd8310"
HF_REVISION="5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
CHECKPOINT_SHA256="1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_ROOT="${1:-${GEM_SMPL_RUNTIME_ROOT:-${REPO_ROOT}/outputs/tmp/gem_smpl/upstream}}"
TORCH_INDEX_URL="${GEM_SMPL_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/outputs/tmp/uv-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/outputs/tmp/huggingface-cache}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if [[ ! -e "$RUNTIME_ROOT" ]]; then
  mkdir -p "$(dirname "$RUNTIME_ROOT")"
  git clone --filter=blob:none https://github.com/NVlabs/GENMO.git "$RUNTIME_ROOT"
elif [[ ! -d "$RUNTIME_ROOT/.git" ]]; then
  echo "Refusing to replace non-git path: $RUNTIME_ROOT" >&2
  exit 2
fi

if ! git -C "$RUNTIME_ROOT" diff --quiet ||
   ! git -C "$RUNTIME_ROOT" diff --cached --quiet; then
  echo "Refusing to change a dirty GEM-SMPL checkout: $RUNTIME_ROOT" >&2
  exit 2
fi

git -C "$RUNTIME_ROOT" fetch origin "$REVISION"
git -C "$RUNTIME_ROOT" checkout --detach "$REVISION"

if [[ ! -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
  rm -rf "$RUNTIME_ROOT/.venv"
  uv venv "$RUNTIME_ROOT/.venv" --python 3.10
fi
# shellcheck disable=SC1091
source "$RUNTIME_ROOT/.venv/bin/activate"
uv pip install torch torchvision --index-url "$TORCH_INDEX_URL"
(cd "$RUNTIME_ROOT" && bash scripts/install_env.sh)
uv pip install huggingface_hub mmengine imageio-ffmpeg ffmpeg-python

if [[ "${DOWNLOAD_WEIGHTS:-0}" == "1" ]]; then
  hf download nvidia/GEM-X gem_smpl.ckpt \
    --revision "$HF_REVISION" \
    --local-dir "$RUNTIME_ROOT/inputs/pretrained"
  echo "$CHECKPOINT_SHA256  $RUNTIME_ROOT/inputs/pretrained/gem_smpl.ckpt" |
    sha256sum --check -
fi

cat <<EOF
Pinned GEM-SMPL runtime ready at:
  $RUNTIME_ROOT

Still required for video inference:
  inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz (licensed SMPL-X asset)
  inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
  inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth

Set DOWNLOAD_WEIGHTS=1 to download and verify the official gem_smpl.ckpt.
EOF
