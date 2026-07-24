#!/usr/bin/env bash
set -euo pipefail

REVISION="32992550dba114c62243fb55e361311972dce8f9"
SOMA_REVISION="e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"
SAM3D_REVISION="b5c765a0d89d789985e186d396315e7590887b94"
HF_REVISION="5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
CHECKPOINT_SHA256="4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_ROOT="${1:-${GEM_X_RUNTIME_ROOT:-${REPO_ROOT}/outputs/tmp/gem_x/upstream}}"
SOURCE_CACHE="${GEM_X_SOURCE_CACHE:-}"
PYTHON_VERSION="${GEM_X_PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${GEM_X_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/outputs/tmp/uv-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/outputs/tmp/huggingface-cache}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if [[ ! -e "$RUNTIME_ROOT" ]]; then
  mkdir -p "$(dirname "$RUNTIME_ROOT")"
  if [[ -n "$SOURCE_CACHE" && -d "$SOURCE_CACHE/.git" ]]; then
    if [[ "$(git -C "$SOURCE_CACHE" rev-parse HEAD)" != "$REVISION" ]]; then
      echo "Unexpected cached GEM-X revision: $SOURCE_CACHE" >&2
      exit 3
    fi
    git clone "$SOURCE_CACHE" "$RUNTIME_ROOT"
    git -C "$RUNTIME_ROOT" config \
      submodule.third_party/soma.url \
      "$SOURCE_CACHE/third_party/soma"
    git -C "$RUNTIME_ROOT" config \
      submodule.third_party/sam-3d-body.url \
      "$SOURCE_CACHE/third_party/sam-3d-body"
  else
    git clone --filter=blob:none https://github.com/NVlabs/GEM-X.git "$RUNTIME_ROOT"
  fi
elif [[ ! -d "$RUNTIME_ROOT/.git" ]]; then
  echo "Refusing to replace non-git path: $RUNTIME_ROOT" >&2
  exit 2
fi

if ! git -C "$RUNTIME_ROOT" diff --quiet ||
   ! git -C "$RUNTIME_ROOT" diff --cached --quiet; then
  echo "Refusing to change a dirty GEM-X checkout: $RUNTIME_ROOT" >&2
  exit 2
fi

if ! git -C "$RUNTIME_ROOT" cat-file -e "${REVISION}^{commit}"; then
  git -C "$RUNTIME_ROOT" fetch origin "$REVISION"
fi
git -C "$RUNTIME_ROOT" checkout --detach "$REVISION"
git -c protocol.file.allow=always -C "$RUNTIME_ROOT" \
  submodule update --init --recursive \
  third_party/soma third_party/sam-3d-body
actual_soma_revision="$(git -C "$RUNTIME_ROOT/third_party/soma" rev-parse HEAD)"
if [[ "$actual_soma_revision" != "$SOMA_REVISION" ]]; then
  echo "Unexpected SOMA revision: $actual_soma_revision" >&2
  exit 3
fi
actual_sam3d_revision="$(
  git -C "$RUNTIME_ROOT/third_party/sam-3d-body" rev-parse HEAD
)"
if [[ "$actual_sam3d_revision" != "$SAM3D_REVISION" ]]; then
  echo "Unexpected SAM-3D-Body revision: $actual_sam3d_revision" >&2
  exit 3
fi

if [[ ! -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
  rm -rf "$RUNTIME_ROOT/.venv"
  # Upstream declares Python 3.12, but its mandatory open3d dependency has no
  # compatible wheel on TencentOS. Python 3.11 is the newest ABI supported by
  # the pinned open3d release and runs the unmodified inference code.
  uv venv "$RUNTIME_ROOT/.venv" --python "$PYTHON_VERSION"
fi
# shellcheck disable=SC1091
source "$RUNTIME_ROOT/.venv/bin/activate"
# Match the CUDA toolkit inside the execution image so Detectron2 can compile.
# The official model code/checkpoint is CUDA-minor agnostic.
uv pip install torch torchvision --index-url "$TORCH_INDEX_URL"
uv pip install -e "$RUNTIME_ROOT/third_party/soma"
(cd "$RUNTIME_ROOT/third_party/soma" && git lfs pull)
mkdir -p "$RUNTIME_ROOT/inputs"
if [[ ! -e "$RUNTIME_ROOT/inputs/soma_assets" ]]; then
  ln -s "$RUNTIME_ROOT/third_party/soma/assets" \
    "$RUNTIME_ROOT/inputs/soma_assets"
fi
(cd "$RUNTIME_ROOT" && bash scripts/install_env.sh)
uv pip uninstall onnxruntime || true
uv pip install \
  "numpy<2" huggingface_hub mmengine imageio-ffmpeg ffmpeg-python \
  onnxruntime-gpu==1.16.3
# Warp's current PyPI wheel is built with CUDA 12.9 and requires a newer
# driver than the R525 Taiji A100 image. Use NVIDIA's official CUDA 11 wheel,
# which remains API-compatible with the SOMA kernels used by this revision.
uv pip install --force-reinstall --no-deps \
  "https://github.com/NVIDIA/warp/releases/download/v1.7.0/warp_lang-1.7.0+cu11-py3-none-manylinux2014_x86_64.whl"

if [[ "${DOWNLOAD_WEIGHTS:-0}" == "1" ]]; then
  hf download nvidia/GEM-X gem_soma.ckpt \
    --revision "$HF_REVISION" \
    --local-dir "$RUNTIME_ROOT/inputs/pretrained"
  echo "$CHECKPOINT_SHA256  $RUNTIME_ROOT/inputs/pretrained/gem_soma.ckpt" |
    sha256sum --check -
fi

cat <<EOF
Pinned GEM-X runtime ready at:
  $RUNTIME_ROOT

Still required for video inference:
  SOMA assets under inputs/soma_assets/
  SAM-3D-Body assets described by the fixed upstream docs/INSTALL.md

Set DOWNLOAD_WEIGHTS=1 to download and verify the official gem_soma.ckpt.
EOF
