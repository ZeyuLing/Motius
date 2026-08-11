#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 MANIFEST MODEL OUTPUT_DIR NUM_SHARDS SHARD_START SHARD_END" >&2
  exit 2
fi

manifest="$1"
model="$2"
output_dir="$3"
num_shards="$4"
shard_start="$5"
shard_end="$6"
gpus_csv="${GPUS:-0,1,2,3,4,5,6,7}"
guidance_scale="${GUIDANCE_SCALE:-5}"
num_inference_steps="${NUM_INFERENCE_STEPS:-50}"
pad_to_frames="${PAD_TO_FRAMES:-360}"
ar_condition_frames="${AR_CONDITION_FRAMES:-5}"
seed="${SEED:-42}"

IFS=',' read -r -a gpus <<< "${gpus_csv}"
if (( ${#gpus[@]} == 0 )); then
  echo "GPUS must contain at least one CUDA device" >&2
  exit 2
fi
if (( shard_start < 0 || shard_end < shard_start || shard_end >= num_shards )); then
  echo "require 0 <= SHARD_START <= SHARD_END < NUM_SHARDS" >&2
  exit 2
fi

mkdir -p "${output_dir}/_logs"
pids=()
shards=()
for shard in $(seq "${shard_start}" "${shard_end}"); do
  gpu="${gpus[$(((shard - shard_start) % ${#gpus[@]}))]}"
  log="${output_dir}/_logs/shard_$(printf '%03d' "${shard}").log"
  status="${output_dir}/_logs/shard_$(printf '%03d' "${shard}").status"
  rm -f "${status}"
  (
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PYTHONPATH:-$PWD}" \
      python3 tools/generate_babel_sequential_prism.py \
      --manifest "${manifest}" \
      --model "${model}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --num-inference-steps "${num_inference_steps}" \
      --guidance-scale "${guidance_scale}" \
      --pad-to-frames "${pad_to_frames}" \
      --ar-condition-frames "${ar_condition_frames}" \
      --kafs-mode none \
      --seed "${seed}" \
      --num-shards "${num_shards}" \
      --shard-index "${shard}" \
      > "${log}" 2>&1
    code=$?
    echo "${code}" > "${status}"
    exit "${code}"
  ) &
  pids+=("$!")
  shards+=("${shard}")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
  shard="${shards[$index]}"
  status="$(cat "${output_dir}/_logs/shard_$(printf '%03d' "${shard}").status" 2>/dev/null || echo missing)"
  printf 'shard=%s status=%s\n' "${shard}" "${status}"
  if [[ "${status}" != "0" ]]; then
    failed=1
  fi
done
exit "${failed}"
