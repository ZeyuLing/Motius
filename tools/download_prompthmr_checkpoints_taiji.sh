#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/checkpoints/prompthmr"
LOG="${OUT}/download.log"
mkdir -p "${OUT}/phmr" "${OUT}/phmr_vid" "${OUT}/third_party"

python3 "${ROOT}/../occupy.py" \
  --gpus all \
  --mem-frac-of-free 0.7 \
  --reserve-gib 8 \
  --duration-s 21600 \
  --report-every-s 60 \
  >"${OUT}/download_occupy.log" 2>&1 &
OCCUPY_PID="$!"
cleanup() {
  if kill -0 "${OCCUPY_PID}" 2>/dev/null; then
    kill "${OCCUPY_PID}" 2>/dev/null || true
    wait "${OCCUPY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python3 -m pip install --quiet --upgrade gdown
download() {
  local id="$1"
  local output="$2"
  gdown "${id}" --continue --output "${output}"
}

download "1uQwMCkkqtyQIwuBeWwE83XcOlISKbTHB" "${OUT}/phmr/checkpoint.ckpt"
download "1P3EEmBDeRRORhBkUhwPZclZxd7U3Z-yo" "${OUT}/phmr/config.yaml"
download "1a0IqV5IBA5L0H9BYbLRVLm3KwQDPXFO0" "${OUT}/phmr_vid/phmr_b1b2.ckpt"
download "1ArARc4hMpxSSZc0r6JIpXPFkzR6c8uEB" "${OUT}/phmr_vid/prhmr_release_002.ckpt"
download "1t282bZ_VmTSmB4GnivJRyVhLIsvhyscx" "${OUT}/phmr_vid/prhmr_release_002.yaml"
download "1KyOwvTE51wel2t-eKnBKdUGju_jHaMpN" "${OUT}/third_party/keypoint_rcnn_5ad38f.pkl"
download "1WndLgBhxrB3JIo9Zp2hZqqEUEedJ8VSe" "${OUT}/third_party/sam2_hiera_tiny.pt"
download "1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO" "${OUT}/third_party/camcalib_sa_biased_l2.ckpt"
download "14hgb59Jk2Pvfiqy4nntE7dUrcKgFmKSj" "${OUT}/third_party/droidcalib.pth"
download "1ZprPoNXe_f9a9flr0RhS3XCJBfqhFSeE" "${OUT}/third_party/vitpose-h-coco_25.pth"

echo \
  "2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5  ${OUT}/phmr_vid/phmr_b1b2.ckpt" |
  sha256sum --check -

OUT="${OUT}" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUT"])
records = []
for path in sorted(root.glob("*/*")):
    if not path.is_file() or path.name.endswith(".log"):
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
(root / "manifest.json").write_text(
    json.dumps(
        {
            "source": "official PromptHMR Google Drive release",
            "files": records,
        },
        indent=2,
    )
    + "\n"
)
PY
