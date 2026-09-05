#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE226-EXDARK-TRAIN-R1"
SOURCE_ROOT="/root/autodl-tmp/vcas/sources/exdark-stage225-dataverse"
OFFICIAL_REPO="/root/autodl-tmp/vcas/sources/exdark-stage224/official-repository"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE226-EXDARK-TRAIN-ONLY-R1"
LOG="/root/autodl-tmp/vcas/logs/stage226-exdark-train-only-r1.log"
SCRIPT="/root/autodl-tmp/vcas/scripts/prepare_stage226_exdark_train_only.py"
PYTHON="/root/miniconda3/bin/python3"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse existing output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi
for required in \
  "${SOURCE_ROOT}/car.tar.gz" \
  "${SOURCE_ROOT}/bus.tar.gz" \
  "${SOURCE_ROOT}/SHA256SUMS" \
  "${OFFICIAL_REPO}/Groundtruth/imageclasslist.txt" \
  "${SCRIPT}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
mkdir -p "$(dirname "${OUTPUT_ROOT}")" "$(dirname "${LOG}")"

INNER="set -euo pipefail; \
  '${PYTHON}' '${SCRIPT}' \
    --source-root '${SOURCE_ROOT}' \
    --official-repo '${OFFICIAL_REPO}' \
    --output-root '${OUTPUT_ROOT}'; \
  echo STAGE226_COMPLETE"

tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER}\" >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
