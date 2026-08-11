#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${ROOT}/data/3DPW"
DESTINATION="${1:?Usage: tools/stage_3dpw_test_data.sh /local/scratch/3DPW}"

test -f "${SOURCE_ROOT}/test_imageFiles.tar"
test -f "${SOURCE_ROOT}/sequenceFiles.zip"
mkdir -p "${DESTINATION}"

tar -xf "${SOURCE_ROOT}/test_imageFiles.tar" -C "${DESTINATION}"
unzip -q -o "${SOURCE_ROOT}/sequenceFiles.zip" \
  "sequenceFiles/test/*.pkl" -d "${DESTINATION}"

STAGED_3DPW="${DESTINATION}" python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["STAGED_3DPW"])
annotations = tuple((root / "sequenceFiles/test").glob("*.pkl"))
images = tuple((root / "imageFiles").glob("*/*.jpg"))
sequences = {path.parent.name for path in images}
if len(annotations) != 24:
    raise RuntimeError(f"Expected 24 3DPW test annotations, found {len(annotations)}.")
if len(images) != 26240:
    raise RuntimeError(f"Expected 26,240 3DPW test images, found {len(images)}.")
if sequences != {path.stem for path in annotations}:
    raise RuntimeError("Staged image and annotation sequence sets do not match.")
print(
    f"Staged 3DPW test: {len(sequences)} sequences, "
    f"{len(images)} images under {root}"
)
PY
