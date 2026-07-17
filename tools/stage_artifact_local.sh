#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SOURCE_DIR DESTINATION_DIR" >&2
  exit 2
fi

source_dir="$(realpath "$1")"
destination_dir="$(realpath -m "$2")"
stage_dir="${destination_dir}.stage.$$"

if [[ ! -d "${source_dir}" ]]; then
  echo "source directory does not exist: ${source_dir}" >&2
  exit 1
fi
if [[ "${destination_dir}" == "/" || "${destination_dir}" == "${source_dir}" ]]; then
  echo "unsafe destination directory: ${destination_dir}" >&2
  exit 1
fi

cleanup() {
  rm -rf "${stage_dir}"
}
trap cleanup EXIT

mkdir -p "${stage_dir}"
export source_dir stage_dir
find "${source_dir}" -type f -print0 | xargs -0 -P 8 -n 1 bash -c '
  source_file="$1"
  relative_path="${source_file#"${source_dir}/"}"
  output_file="${stage_dir}/${relative_path}"
  mkdir -p "$(dirname "${output_file}")"
  cp "${source_file}" "${output_file}"
' _

source_count="$(find "${source_dir}" -type f | wc -l)"
staged_count="$(find "${stage_dir}" -type f | wc -l)"
source_bytes="$(find "${source_dir}" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
staged_bytes="$(find "${stage_dir}" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
if [[ "${source_count}" != "${staged_count}" || "${source_bytes}" != "${staged_bytes}" ]]; then
  echo "staging verification failed: files ${source_count}/${staged_count}, bytes ${source_bytes}/${staged_bytes}" >&2
  exit 1
fi

rm -rf "${destination_dir}"
mv "${stage_dir}" "${destination_dir}"
trap - EXIT
printf 'staged %s files (%s bytes) at %s\n' "${staged_count}" "${staged_bytes}" "${destination_dir}"
