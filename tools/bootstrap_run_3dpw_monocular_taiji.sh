#!/usr/bin/env bash
set -euo pipefail

ROOT="${MOTIUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
METHOD="${1:?Usage: tools/bootstrap_run_3dpw_monocular_taiji.sh METHOD [GPU_COUNT]}"
GPU_COUNT="${2:-8}"
OUTPUT_METHOD="${MOTIUS_OUTPUT_METHOD:-${METHOD}}"
export UV_CACHE_DIR="/tmp/motius_uv_cache"
export HF_HOME="/tmp/motius_hf_cache"
export STAGE_ROOT="/tmp/motius_3dpw_eval"
OCCUPY_PID_FILE="/tmp/motius_setup_occupy.pid"
POOL_OCCUPY_SESSION="${MOTIUS_POOL_OCCUPY_SESSION:-motius_pool_occupy}"
POOL_OCCUPY_MEM_FRAC="${MOTIUS_POOL_OCCUPY_MEM_FRAC:-0.70}"
POOL_OCCUPY_RESERVE_GIB="${MOTIUS_POOL_OCCUPY_RESERVE_GIB:-1}"
POOL_OCCUPY_LAYERS="${MOTIUS_POOL_OCCUPY_LAYERS:-4}"
LOG_ROOT="${ROOT}/outputs/evaluation/monocular_capture/3dpw_test/${OUTPUT_METHOD}/logs"
OCCUPY_LOG="${LOG_ROOT}/setup_occupy.log"
OCCUPY_PYTHON="$(command -v python3)"
SETUP_OCCUPY="${MOTIUS_SETUP_OCCUPY:-1}"

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/bootstrap.log") 2>&1
echo "[$(date -Is)] bootstrap method=${METHOD} output_method=${OUTPUT_METHOD} gpus=${GPU_COUNT}"
stop_tracked_occupy() {
  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "${POOL_OCCUPY_SESSION}" 2>/dev/null || true
  fi
  if [[ ! -f "${OCCUPY_PID_FILE}" ]]; then
    return
  fi
  local pid
  pid="$(<"${OCCUPY_PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 -- "-${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${OCCUPY_PID_FILE}"
}
stop_tracked_occupy
if [[ "${METHOD}" != "gem_x" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y ffmpeg || true
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y ffmpeg || true
  elif command -v yum >/dev/null 2>&1; then
    yum install -y ffmpeg || true
  fi
fi
if [[ "${METHOD}" != "gem_x" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  if python3 - <<'PY'
import sys

raise SystemExit(0 if sys.version_info >= (3, 7) else 1)
PY
  then
    python3 -m pip install --quiet imageio-ffmpeg
    IMAGEIO_FFMPEG_BIN="$(
      python3 - <<'PY'
import imageio_ffmpeg

print(imageio_ffmpeg.get_ffmpeg_exe())
PY
    )"
    if [[ -x "${IMAGEIO_FFMPEG_BIN}" ]]; then
      install -m 0755 "${IMAGEIO_FFMPEG_BIN}" /usr/local/bin/ffmpeg
    fi
  fi
fi
if [[ "${METHOD}" != "gem_x" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg CLI is unavailable; continuing for OpenCV-only methods." >&2
fi
OCCUPY_PID=""
if [[ "${SETUP_OCCUPY}" == "1" ]]; then
  setsid env -u PYTHONPATH "${OCCUPY_PYTHON}" "${ROOT}/../occupy.py" \
    --gpus all \
    --mem-frac-of-free 0.7 \
    --reserve-gib 8 \
    --duration-s 21600 \
    --report-every-s 60 \
    >"${OCCUPY_LOG}" 2>&1 &
  OCCUPY_PID="$!"
  echo "${OCCUPY_PID}" >"${OCCUPY_PID_FILE}"
fi
cleanup_occupy() {
  trap - EXIT INT TERM
  if [[ -n "${OCCUPY_PID}" ]] && kill -0 "${OCCUPY_PID}" 2>/dev/null; then
    kill -- "-${OCCUPY_PID}" 2>/dev/null || true
    wait "${OCCUPY_PID}" 2>/dev/null || true
  fi
  rm -f "${OCCUPY_PID_FILE}"
  if [[ "${MOTIUS_RESTART_POOL_OCCUPY:-0}" == "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
      echo "tmux is required for persistent pool occupancy." >&2
      return
    fi
    tmux kill-session -t "${POOL_OCCUPY_SESSION}" 2>/dev/null || true
    local layer
    local launch_mode
    for ((layer = 0; layer < POOL_OCCUPY_LAYERS; layer++)); do
      if ((layer == 0)); then
        launch_mode=(new-session -d -s "${POOL_OCCUPY_SESSION}")
      else
        launch_mode=(new-window -d -t "${POOL_OCCUPY_SESSION}")
      fi
      tmux "${launch_mode[@]}" -n "occupy_${layer}" \
        "exec env -u PYTHONPATH '${OCCUPY_PYTHON}' '${ROOT}/../occupy.py' \
--gpus all --mem-frac-of-free '${POOL_OCCUPY_MEM_FRAC}' \
--reserve-gib '${POOL_OCCUPY_RESERVE_GIB}' --duration-s 0 \
--report-every-s 60 >'${LOG_ROOT}/pool_occupy_after_${layer}.log' 2>&1"
      sleep 10
    done
    tmux list-panes -t "${POOL_OCCUPY_SESSION}" -F '#{pane_pid}' \
      | head -n 1 >"${OCCUPY_PID_FILE}"
  fi
}
trap cleanup_occupy EXIT INT TERM
if [[ -n "${OCCUPY_PID}" ]]; then
  sleep 5
  if ! kill -0 "${OCCUPY_PID}" 2>/dev/null; then
    echo "occupy.py failed during environment setup; see ${OCCUPY_LOG}" >&2
    exit 4
  fi
fi
if [[ "${METHOD}" == "prompthmr" || "${METHOD}" == "gvhmr" ]]; then
  DRIVER_MAJOR="$(
    nvidia-smi --query-gpu=driver_version --format=csv,noheader \
      | awk -F. 'NR==1{print $1}'
  )"
  if [[ ! "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]] || (( DRIVER_MAJOR < 525 )); then
    echo "${METHOD} requires an NVIDIA driver compatible with CUDA 12; found driver major ${DRIVER_MAJOR:-unknown}." >&2
    exit 5
  fi
fi

if [[ "${METHOD}" != "gem_x" ]]; then
  python3 -m pip install --quiet --upgrade uv huggingface_hub gdown
fi

case "${METHOD}" in
  prompthmr)
    CONDA_ROOT="/tmp/motius_miniforge"
    if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
      bash outputs/tmp/Miniforge3-Linux-x86_64.sh -b -p "${CONDA_ROOT}"
    fi
    export PATH="${CONDA_ROOT}/bin:${PATH}"
    export CONDA_ENVS_PATH="/tmp/motius_conda_envs"
    export CONDA_PKGS_DIRS="/tmp/motius_conda_pkgs"
    export UPSTREAM_DIR="/tmp/motius_prompthmr"
    PROMPTHMR_ACCEPT_LICENSE=1 DOWNLOAD_VIDEO_CHECKPOINT=false \
      bash tools/setup_prompthmr_env.sh
    SAFE_SITE="/tmp/motius_prompthmr_safe_site"
    mkdir -p "${SAFE_SITE}"
    install -m 0644 \
      "${ROOT}/tools/runtime_sites/prompthmr/sitecustomize.py" \
      "${SAFE_SITE}/sitecustomize.py"
    export PYTHONPATH="${SAFE_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
    export MOTIUS_PROMPTHMR_AUDITED_PATCH=1
    mkdir -p \
      "${UPSTREAM_DIR}/data/body_models/smplx" \
      "${UPSTREAM_DIR}/data/body_models/smpl"
    ln -sf \
      "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/body_models/smplx/SMPLX_NEUTRAL_2020.npz" \
      "${UPSTREAM_DIR}/data/body_models/smplx/SMPLX_NEUTRAL.npz"
    ln -sf \
      "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
      "${UPSTREAM_DIR}/data/body_models/smpl/SMPL_NEUTRAL.pkl"
    for body_asset in \
      J_regressor_h36m.npy \
      smpl_mean_params.npz \
      smplx2smpl_joints.npy \
      smplx2smpl.pkl; do
      ln -sf \
        "${ROOT}/checkpoints/prompthmr/body_models/${body_asset}" \
        "${UPSTREAM_DIR}/data/body_models/${body_asset}"
    done
    mkdir -p \
      "${UPSTREAM_DIR}/data/pretrain/phmr" \
      "${UPSTREAM_DIR}/data/pretrain/phmr_vid" \
      "${UPSTREAM_DIR}/data/pretrain/sam2_ckpts"
    rm -f "${UPSTREAM_DIR}/data/pretrain/phmr/checkpoint.ckpt"
    cp "${ROOT}/checkpoints/prompthmr/phmr/checkpoint.ckpt" \
      "${UPSTREAM_DIR}/data/pretrain/phmr/checkpoint.ckpt"
    ln -sf "${ROOT}/checkpoints/prompthmr/phmr/config.yaml" \
      "${UPSTREAM_DIR}/data/pretrain/phmr/config.yaml"
    for name in phmr_b1b2.ckpt prhmr_release_002.yaml; do
      ln -sf "${ROOT}/checkpoints/prompthmr/phmr_vid/${name}" \
        "${UPSTREAM_DIR}/data/pretrain/phmr_vid/${name}"
    done
    for name in keypoint_rcnn_5ad38f.pkl sam2_hiera_tiny.pt; do
      ln -sf "${ROOT}/checkpoints/prompthmr/third_party/${name}" \
        "${UPSTREAM_DIR}/data/pretrain/sam2_ckpts/${name}"
    done
    for name in camcalib_sa_biased_l2.ckpt droidcalib.pth vitpose-h-coco_25.pth; do
      ln -sf "${ROOT}/checkpoints/prompthmr/third_party/${name}" \
        "${UPSTREAM_DIR}/data/pretrain/${name}"
    done
    # Upstream video configs retain the official Docker image's `/code/data`
    # prefix. Map that isolated-container path to this pinned runtime's assets.
    mkdir -p /code
    ln -sTfn "${UPSTREAM_DIR}/data" /code/data
    export PROMPTHMR_ROOT="${UPSTREAM_DIR}"
    export PROMPTHMR_PYTHON="${CONDA_ENVS_PATH}/phmr_pt2.4/bin/python"
    PROMPTHMR_NVIDIA_LIBS="$("${PROMPTHMR_PYTHON}" - <<'PY'
from pathlib import Path
import site

root = Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib"))))
PY
)"
    export LD_LIBRARY_PATH="${PROMPTHMR_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${PROMPTHMR_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PromptHMR requires a CUDA 12-compatible driver.")
print("PromptHMR CUDA:", torch.version.cuda, torch.cuda.get_device_name(0))
PY
    ;;
  gvhmr)
    CONDA_ROOT="/tmp/motius_miniforge"
    if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
      bash outputs/tmp/Miniforge3-Linux-x86_64.sh -b -p "${CONDA_ROOT}"
    fi
    export PATH="${CONDA_ROOT}/bin:${PATH}"
    export CONDA_ENVS_PATH="/tmp/motius_conda_envs"
    export CONDA_PKGS_DIRS="/tmp/motius_conda_pkgs"
    export GVHMR_ROOT="/tmp/motius_gvhmr"
    export GVHMR_ENV="/tmp/motius_conda_envs/gvhmr"
    bash tools/setup_gvhmr_env.sh "${GVHMR_ROOT}" "${GVHMR_ENV}"
    mkdir -p \
      "${GVHMR_ROOT}/inputs/checkpoints/gvhmr" \
      "${GVHMR_ROOT}/inputs/checkpoints/hmr2" \
      "${GVHMR_ROOT}/inputs/checkpoints/vitpose" \
      "${GVHMR_ROOT}/inputs/checkpoints/yolo" \
      "${GVHMR_ROOT}/inputs/checkpoints/body_models/smplx" \
      "${GVHMR_ROOT}/inputs/checkpoints/body_models/smpl"
    stage_local_asset() {
      local source="$1"
      local destination="$2"
      local temporary="${destination}.tmp.$$"
      if [[ -f "${destination}" && ! -L "${destination}" ]] \
        && [[ "$(stat -c %s "${source}")" == "$(stat -c %s "${destination}")" ]]; then
        return
      fi
      rm -f "${destination}" "${temporary}"
      cp --dereference --reflink=auto "${source}" "${temporary}"
      if [[ "$(stat -c %s "${source}")" != "$(stat -c %s "${temporary}")" ]]; then
        echo "Incomplete local GVHMR asset copy: ${source}" >&2
        rm -f "${temporary}"
        exit 6
      fi
      mv -f "${temporary}" "${destination}"
    }
    stage_local_asset \
      "${ROOT}/checkpoints/gvhmr/gvhmr/gvhmr_siga24_release.ckpt" \
      "${GVHMR_ROOT}/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
    stage_local_asset \
      "${ROOT}/checkpoints/gem_smpl/hmr2/epoch=10-step=25000.ckpt" \
      "${GVHMR_ROOT}/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    stage_local_asset \
      "${ROOT}/checkpoints/gem_smpl/vitpose/vitpose-h-multi-coco.pth" \
      "${GVHMR_ROOT}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    stage_local_asset \
      "${ROOT}/checkpoints/gvhmr/yolo/yolov8x.pt" \
      "${GVHMR_ROOT}/inputs/checkpoints/yolo/yolov8x.pt"
    stage_local_asset \
      "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/body_models/smplx/SMPLX_NEUTRAL_2020.npz" \
      "${GVHMR_ROOT}/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    stage_local_asset \
      "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
      "${GVHMR_ROOT}/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
    export GVHMR_PYTHON="${GVHMR_ENV}/bin/python"
    GVHMR_SAFE_SITE="/tmp/motius_gvhmr_safe_site"
    mkdir -p "${GVHMR_SAFE_SITE}"
    cat >"${GVHMR_SAFE_SITE}/sitecustomize.py" <<'PY'
import torch

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
PY
    export PYTHONPATH="${GVHMR_SAFE_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
    export MOTIUS_GVHMR_SKIP_RENDER=1
    "${GVHMR_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("GVHMR requires a CUDA 12-compatible driver.")
print("GVHMR CUDA:", torch.version.cuda, torch.cuda.get_device_name(0))
PY
    ;;
  gem_smpl)
    export GEM_SMPL_ROOT="/tmp/motius_gem_smpl"
    DOWNLOAD_WEIGHTS=0 \
      bash motius/models/gem_smpl/setup_runtime.sh "${GEM_SMPL_ROOT}"
    GEM_SMPL_ROOT="${GEM_SMPL_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["GEM_SMPL_ROOT"])
demo = root / "scripts/demo/demo_smpl.py"
text = demo.read_text()
old = "model = load_model(ckpt_path, load_text_encoder=True)"
new = (
    "# Motius video-only evaluation: all segments have has_text=False, so the "
    "11GB T5-3B encoder is provably unused.\n"
    "    model = load_model(ckpt_path, load_text_encoder=False)"
)
if old not in text and "load_text_encoder=False" not in text:
    raise RuntimeError("Pinned GEM-SMPL video-only patch context changed.")
if old in text:
    demo.write_text(text.replace(old, new, 1))
gem_module = root / "gem/gem.py"
gem_text = gem_module.read_text()
old_bool = "no_text = ~has_text.to(device)"
new_bool = "no_text = ~has_text.to(device=device, dtype=torch.bool)"
if old_bool not in gem_text and new_bool not in gem_text:
    raise RuntimeError("Pinned GEM-SMPL has_text patch context changed.")
if old_bool in gem_text:
    gem_module.write_text(gem_text.replace(old_bool, new_bool, 1))
encoder = root / "gem/network/base_arch/transformer/encoder_rope.py"
encoder_text = encoder.read_text()
old_empty = (
    '            if multi_text_data is not None:\n'
    '                # Note: positional encoding not yet supported for MHA cross-attention\n'
    '                out = []'
)
new_empty = (
    '            if multi_text_data is not None:\n'
    '                # A video-only request legitimately has zero text windows.\n'
    '                if len(multi_text_data["text_embed_feats"]) == 0:\n'
    '                    return torch.zeros_like(x)\n'
    '                # Note: positional encoding not yet supported for MHA cross-attention\n'
    '                out = []'
)
if old_empty not in encoder_text and "zero text windows" not in encoder_text:
    raise RuntimeError("Pinned GEM-SMPL empty-text patch context changed.")
if old_empty in encoder_text:
    encoder.write_text(encoder_text.replace(old_empty, new_empty, 1))
demo_utils = root / "scripts/demo/demo_utils.py"
utils_text = demo_utils.read_text()
start = utils_text.index("def detect_and_track(")
end = utils_text.index("\n# ---- 1b.", start)
tracker_impl = '''def detect_and_track(video_path: str, preprocess_dir: str) -> torch.Tensor:
    """Track one persistent person with Ultralytics ByteTrack."""
    cache_path = os.path.join(preprocess_dir, "bbx.pt")
    if os.path.exists(cache_path):
        print(f"[Stage 1a] Loading cached bounding boxes from {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    if os.environ.get("MOTIUS_GEM_SMPL_REQUIRE_TARGET_CROP") == "1":
        raise RuntimeError(
            f"Required per-target 3DPW crop cache is missing: {cache_path}"
        )

    print("[Stage 1a] Running YOLOv8 + ByteTrack person tracking ...")
    from ultralytics import YOLO
    from gem.utils.video_io_utils import get_video_lwh, read_video_np
    from gem.utils.geo_transform import get_bbx_xys_from_xyxy
    from gem.utils.net_utils import moving_average_smooth

    length, width, height = get_video_lwh(video_path)
    frames = read_video_np(video_path)
    model = YOLO("yolov8x.pt")
    tracks = {}
    fallback = []
    for frame_index, frame in enumerate(tqdm(frames, desc="YOLO ByteTrack", leave=False)):
        result = model.track(
            frame[..., ::-1].copy(),
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if len(boxes) else np.empty((0, 4))
        fallback.append(
            xyxy[np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))]
            if len(xyxy)
            else None
        )
        if boxes.id is None:
            continue
        ids = boxes.id.detach().cpu().numpy().astype(np.int64)
        for track_id, box in zip(ids, xyxy):
            area = float(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))
            record = tracks.setdefault(int(track_id), {"frames": [], "boxes": [], "area": 0.0})
            record["frames"].append(frame_index)
            record["boxes"].append(box.astype(np.float32))
            record["area"] += area

    if tracks:
        selected_id, selected = max(
            tracks.items(),
            key=lambda item: (len(item[1]["frames"]), item[1]["area"]),
        )
        known_frames = np.asarray(selected["frames"], dtype=np.int64)
        known_boxes = np.asarray(selected["boxes"], dtype=np.float32)
        print(
            f"[Stage 1a] Selected ByteTrack ID {selected_id}: "
            f"{len(known_frames)}/{length} detected frames"
        )
    else:
        valid = [(index, box) for index, box in enumerate(fallback) if box is not None]
        if not valid:
            raise RuntimeError("YOLO ByteTrack found no person in the video.")
        known_frames = np.asarray([item[0] for item in valid], dtype=np.int64)
        known_boxes = np.asarray([item[1] for item in valid], dtype=np.float32)

    timeline = np.arange(length, dtype=np.float32)
    dense_boxes = np.stack(
        [
            np.interp(timeline, known_frames, known_boxes[:, coordinate])
            for coordinate in range(4)
        ],
        axis=1,
    )
    bbx_xyxy = torch.from_numpy(dense_boxes).float()
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, width - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, height - 1)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)
    bbx_xys = moving_average_smooth(bbx_xys, window_size=5, dim=0)
    os.makedirs(preprocess_dir, exist_ok=True)
    torch.save(bbx_xys, cache_path)
    print(f"[Stage 1a] ByteTrack boxes saved to {cache_path} ({len(bbx_xys)} frames)")
    return bbx_xys
'''
demo_utils.write_text(utils_text[:start] + tracker_impl + utils_text[end:])
(root / ".motius_video_only_no_text_encoder").write_text("3dpw video-only\n")
(root / ".motius_bytetrack_patch").write_text("ultralytics persistent ID\n")
PY
    export MOTIUS_GEM_SMPL_REQUIRE_TARGET_CROP=1
    mkdir -p \
      "${GEM_SMPL_ROOT}/inputs/pretrained" \
      "${GEM_SMPL_ROOT}/inputs/checkpoints/body_models/smplx" \
      "${GEM_SMPL_ROOT}/inputs/checkpoints/hmr2" \
      "${GEM_SMPL_ROOT}/inputs/checkpoints/vitpose"
    ln -sf \
      "${ROOT}/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" \
      "${GEM_SMPL_ROOT}/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    ln -sf "${ROOT}/checkpoints/gem_smpl/gem_smpl.ckpt" \
      "${GEM_SMPL_ROOT}/inputs/pretrained/gem_smpl.ckpt"
    ln -sf \
      "${ROOT}/checkpoints/gem_smpl/hmr2/epoch=10-step=25000.ckpt" \
      "${GEM_SMPL_ROOT}/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
    ln -sf \
      "${ROOT}/checkpoints/gem_smpl/vitpose/vitpose-h-multi-coco.pth" \
      "${GEM_SMPL_ROOT}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    mkdir -p "${GEM_SMPL_ROOT}/gem/utils/body_model"
    for body_asset in \
      coco_aug_dict.pth \
      smpl_3dpw14_J_regressor_sparse.pt \
      smpl_coco17_J_regressor.pt \
      smpl_neutral_J_regressor.pt \
      smplx2smpl_sparse.pt \
      smplx_verts437.pt; do
      ln -sf \
        "${ROOT}/checkpoints/gem_smpl/body_model/${body_asset}" \
        "${GEM_SMPL_ROOT}/gem/utils/body_model/${body_asset}"
    done
    export GEM_SMPL_PYTHON="${GEM_SMPL_ROOT}/.venv/bin/python"
    ;;
  gem_x)
    export GEM_X_ROOT="/tmp/motius_gem_x"
    export GEM_X_SOURCE_CACHE="${ROOT}/outputs/tmp/gem_x/upstream"
    GEM_X_ASSET_CACHE="/tmp/motius_gem_x_assets"
    stage_gem_x_asset() {
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
        echo "Staged GEM-X asset has the wrong size: ${temporary}" >&2
        rm -f "${temporary}"
        exit 8
      fi
      mv -f "${temporary}" "${destination}"
    }
    declare -A GEM_X_ASSET_SIZES=(
      [gem_soma.ckpt]=541758499
      [vitpose.pth]=3388483384
      [sam3d_body.ckpt]=2109129346
      [model_config.yaml]=1488
      [mhr_model.pt]=696110248
      [scale_mean.pth]=1451
      [scale_comps.pth]=8816
    )
    for asset in "${!GEM_X_ASSET_SIZES[@]}"; do
      stage_gem_x_asset \
        "${ROOT}/checkpoints/gem_x/${asset}" \
        "${GEM_X_ASSET_CACHE}/${asset}" \
        "${GEM_X_ASSET_SIZES[${asset}]}"
    done
    echo "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e  ${GEM_X_ASSET_CACHE}/gem_soma.ckpt" |
      sha256sum --check -
    if [[ ! -f "${GEM_X_ASSET_CACHE}/torch_cache/.motius_complete" ]]; then
      rm -rf "${GEM_X_ASSET_CACHE}/torch_cache"
      rm -rf "${GEM_X_ASSET_CACHE}/torch_cache.partial"
      cp -a \
        "${ROOT}/checkpoints/gem_x/torch_cache" \
        "${GEM_X_ASSET_CACHE}/torch_cache.partial"
      touch "${GEM_X_ASSET_CACHE}/torch_cache.partial/.motius_complete"
      mv \
        "${GEM_X_ASSET_CACHE}/torch_cache.partial" \
        "${GEM_X_ASSET_CACHE}/torch_cache"
    fi
    export TORCH_HOME="${GEM_X_ASSET_CACHE}/torch_cache"
    if command -v yum >/dev/null 2>&1; then
      yum install -y \
        python3.11-devel \
        gcc-toolset-11-gcc \
        gcc-toolset-11-gcc-c++ \
        ninja-build
      # shellcheck disable=SC1091
      set +u
      source /opt/rh/gcc-toolset-11/enable
      set -u
      export CC=gcc CXX=g++
      export MAX_JOBS="${MAX_JOBS:-8}"
    fi
    if ! command -v python3.11 >/dev/null 2>&1; then
      echo "GEM-X bootstrap requires Python 3.11." >&2
      exit 7
    fi
    python3.11 -m ensurepip --upgrade
    python3.11 -m pip install --quiet --upgrade uv huggingface_hub gdown
    if ! command -v git-lfs >/dev/null 2>&1 && \
       ! git lfs version >/dev/null 2>&1; then
      if command -v yum >/dev/null 2>&1; then
        yum install -y git-lfs
      else
        apt-get update && apt-get install -y git-lfs
      fi
    fi
    DOWNLOAD_WEIGHTS=0 \
      bash motius/models/gem_x/setup_runtime.sh "${GEM_X_ROOT}"
    GEM_X_ROOT="${GEM_X_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

demo = Path(os.environ["GEM_X_ROOT"]) / "scripts/demo/demo_soma.py"
text = demo.read_text()
old = (
    "    render_incam(cfg, fps=fps)\n"
    "    render_global_o3d(cfg, fps=fps)\n\n"
    "    if args.retarget:"
)
new = (
    "    # Motius metric runs consume hpe_results directly; Open3D rendering is\n"
    "    # optional and unavailable in the headless Taiji image.\n"
    "    if os.environ.get('MOTIUS_GEM_X_SKIP_RENDER') == '1':\n"
    "        Log.info('[Done] Skipping optional Open3D render for Motius export')\n"
    "        return\n\n"
    "    render_incam(cfg, fps=fps)\n"
    "    render_global_o3d(cfg, fps=fps)\n\n"
    "    if args.retarget:"
)
if old not in text and "MOTIUS_GEM_X_SKIP_RENDER" not in text:
    raise RuntimeError("Pinned GEM-X render-skip patch context changed.")
if old in text:
    text = text.replace(old, new, 1)
old_overlay = (
    "    render_2d_keypoints(\n"
    "        video_path=cfg.video_path,\n"
    "        vitpose_path=cfg.paths.vitpose,\n"
    "        bbx_path=cfg.paths.bbx,\n"
    "        output_path=str(Path(cfg.output_dir) / \"0_kp2d77_overlay.mp4\"),\n"
    "        fps=fps,\n"
    "    )\n"
)
new_overlay = (
    "    if os.environ.get('MOTIUS_GEM_X_SKIP_RENDER') != '1':\n"
    + "".join(f"    {line}\n" for line in old_overlay.splitlines())
    + "    else:\n"
    "        Log.info('[2D KP] Skipping optional overlay for Motius export')\n"
)
if old_overlay not in text and "Skipping optional overlay for Motius export" not in text:
    raise RuntimeError("Pinned GEM-X keypoint-render patch context changed.")
if old_overlay in text:
    text = text.replace(old_overlay, new_overlay, 1)
demo.write_text(text)
PY
    export MOTIUS_GEM_X_SKIP_RENDER=1
    mkdir -p \
      "${GEM_X_ROOT}/inputs/pretrained" \
      "${GEM_X_ROOT}/inputs/checkpoints/vitpose" \
      "${GEM_X_ROOT}/inputs/checkpoints/sam-3d-body-dinov3" \
      "${GEM_X_ROOT}/inputs/mhr_data" \
      "${GEM_X_ROOT}/inputs/soma_data"
    ln -sf "${GEM_X_ASSET_CACHE}/gem_soma.ckpt" \
      "${GEM_X_ROOT}/inputs/pretrained/gem_soma.ckpt"
    ln -sf "${GEM_X_ASSET_CACHE}/vitpose.pth" \
      "${GEM_X_ROOT}/inputs/checkpoints/vitpose/vitpose.pth"
    ln -sf "${GEM_X_ASSET_CACHE}/sam3d_body.ckpt" \
      "${GEM_X_ROOT}/inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt"
    ln -sf "${GEM_X_ASSET_CACHE}/model_config.yaml" \
      "${GEM_X_ROOT}/inputs/checkpoints/sam-3d-body-dinov3/model_config.yaml"
    ln -sf "${GEM_X_ASSET_CACHE}/mhr_model.pt" \
      "${GEM_X_ROOT}/inputs/mhr_data/mhr_model.pt"
    ln -sf "${GEM_X_ASSET_CACHE}/scale_mean.pth" \
      "${GEM_X_ROOT}/inputs/soma_data/scale_mean.pth"
    ln -sf "${GEM_X_ASSET_CACHE}/scale_comps.pth" \
      "${GEM_X_ROOT}/inputs/soma_data/scale_comps.pth"
    export GEM_X_PYTHON="${GEM_X_ROOT}/.venv/bin/python"
    GEM_X_NVIDIA_LIBS="$("${GEM_X_PYTHON}" - <<'PY'
from pathlib import Path
import site

root = Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib"))))
PY
)"
    export LD_LIBRARY_PATH="/usr/lib64:${GEM_X_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${GEM_X_PYTHON}" - <<'PY'
import onnxruntime as ort
import warp as wp

providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise RuntimeError(f"GEM-X requires ONNX CUDAExecutionProvider, got {providers}")
wp.init()
devices = wp.get_cuda_devices()
if not devices:
    raise RuntimeError("GEM-X requires at least one CUDA device in NVIDIA Warp.")
if not hasattr(wp, "where"):
    raise RuntimeError("GEM-X requires NVIDIA Warp with wp.where support.")
PY
    ;;
  *)
    echo "Unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${ROOT}/outputs/evaluation/monocular_capture/3dpw_test" \
  tools/run_3dpw_monocular_taiji.sh "${METHOD}" "${GPU_COUNT}"
