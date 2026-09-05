#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE228-EXDARK-TEACHER-R1"
INPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE227-EXDARK-COLOR-PROPOSALS-R1"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE228-EXDARK-COLOR-MULTITEACHER-R1"
TRAINING_CODE="/root/autodl-tmp/vcas/code/training"
LABELS="/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json"
TEACHER256="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-256-FRESH-R1/best.pt"
TEACHER288="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-288-LETTERBOX-FRESH-R1/best.pt"
LOG="/root/autodl-tmp/vcas/logs/stage228-exdark-color-multiteacher-r1.log"
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
  "${INPUT_ROOT}/stage227-exdark-color-proposals.csv" \
  "${INPUT_ROOT}/stage227-exdark-vehicle-color-proposals.json" \
  "${TRAINING_CODE}/scripts/audit_color_multiteacher_consensus.py" \
  "${TRAINING_CODE}/scripts/build_hard_crop_contact_sheets.py" \
  "${LABELS}" "${TEACHER256}" "${TEACHER288}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG}")"

INNER="set -euo pipefail; \
  set +e; \
  '${PYTHON}' '${TRAINING_CODE}/scripts/audit_color_multiteacher_consensus.py' \
    --manifest '${INPUT_ROOT}/stage227-exdark-color-proposals.csv' \
    --dataset-root / \
    --labels '${LABELS}' \
    --checkpoint stage158_256='${TEACHER256}' \
    --checkpoint stage158_288='${TEACHER288}' \
    --output-manifest '${OUTPUT_ROOT}/stage228-exdark-color-reviewed.csv' \
    --output-report '${OUTPUT_ROOT}/stage228-exdark-color-multiteacher.json' \
    --confidence 0.50 --minimum-total 20 --minimum-classes 4 \
    --batch-size 48 --workers 4 --device cuda; \
  rc=\$?; \
  set -e; \
  test \$rc -eq 0 -o \$rc -eq 2; \
  test -s '${OUTPUT_ROOT}/stage228-exdark-color-reviewed.csv'; \
  test -s '${OUTPUT_ROOT}/stage228-exdark-color-multiteacher.json'; \
  '${PYTHON}' '${TRAINING_CODE}/scripts/build_hard_crop_contact_sheets.py' \
    --manifest '${OUTPUT_ROOT}/stage228-exdark-color-reviewed.csv' \
    --dataset-root / --output-dir '${OUTPUT_ROOT}/accepted-contact-sheets' \
    --per-group 30 --columns 5 --field color --accepted-only; \
  sha256sum '${OUTPUT_ROOT}/stage228-exdark-color-reviewed.csv' \
    '${OUTPUT_ROOT}/stage228-exdark-color-multiteacher.json' \
    > '${OUTPUT_ROOT}/SHA256SUMS'; \
  find '${OUTPUT_ROOT}/accepted-contact-sheets' -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | xargs -0 -r sha256sum >> '${OUTPUT_ROOT}/SHA256SUMS'; \
  echo STAGE228_TERMINAL_RC=\$rc; \
  exit \$rc"

tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER}\" >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
