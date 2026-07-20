#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "Usage: $0 ARTIFACT ANNOTATION ANNOTATION_ROOT OUTPUT_DIR [NUM_GPUS] [BATCH_SIZE] [CAPTION_MAP]" >&2
  exit 2
fi

artifact=$1
annotation=$2
annotation_root=$3
output_dir=$4
num_gpus=${5:-8}
batch_size=${6:-64}
caption_map=${7:-}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
log_dir="$output_dir/_logs"
mkdir -p "$log_dir"

for ((rank = 0; rank < num_gpus; rank++)); do
  session="maskcontrol_t2m_${rank}"
  tmux kill-session -t "$session" 2>/dev/null || true
  caption_arg=""
  if [[ -n "$caption_map" ]]; then
    caption_arg=$(printf ' --caption-map %q' "$caption_map")
  fi
  command=$(printf \
    'cd %q && CUDA_VISIBLE_DEVICES=%q python3 tools/eval_maskcontrol_humanml3d.py --artifact %q --annotation %q --annotation-root %q --out-dir %q --batch-size %q --num-shards %q --shard-index %q --seed 42 --skip-existing%s > %q 2>&1' \
    "$repo_root" "$rank" "$artifact" "$annotation" "$annotation_root" \
    "$output_dir" "$batch_size" "$num_gpus" "$rank" \
    "$caption_arg" "$log_dir/shard_$(printf '%02d' "$rank").log")
  tmux new-session -d -s "$session" "$command"
done

echo "Started $num_gpus MaskControl shards; logs: $log_dir"
