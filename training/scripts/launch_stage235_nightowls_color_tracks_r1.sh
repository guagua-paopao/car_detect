#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE235-NIGHTOWLS-COLOR-R1"
INPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE234-NIGHTOWLS-VEHICLE-TRACKS-R1"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE235-NIGHTOWLS-COLOR-TRACKS-R1"
TRAINING_CODE="/root/autodl-tmp/vcas/code/training"
LABELS="/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json"
TEACHER256="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-256-FRESH-R1/best.pt"
TEACHER288="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-288-LETTERBOX-FRESH-R1/best.pt"
LOG="/root/autodl-tmp/vcas/logs/stage235-nightowls-color-tracks-r1.log"
PYTHON="/root/miniconda3/bin/python3"

run_worker() {
  set +e
  "${PYTHON}" "${TRAINING_CODE}/scripts/audit_stage235_nightowls_color_tracks.py" \
    --manifest "${INPUT_ROOT}/stage234-nightowls-vehicle-tracks.csv" \
    --dataset-root / \
    --labels "${LABELS}" \
    --checkpoint stage158_256="${TEACHER256}" \
    --checkpoint stage158_288="${TEACHER288}" \
    --output-manifest "${OUTPUT_ROOT}/stage235-nightowls-color-reviewed.csv" \
    --output-report "${OUTPUT_ROOT}/stage235-nightowls-color-track-consensus.json" \
    --base-confidence 0.72 \
    --minimum-track-frames 3 \
    --minimum-unique-track-frames 2 \
    --minimum-track-support-ratio 0.80 \
    --maximum-supervised-frames-per-track 3 \
    --minimum-accepted-tracks 100 \
    --minimum-colors 4 \
    --batch-size 64 --workers 4 --device cuda
  rc=$?
  set -e
  test "${rc}" -eq 0 -o "${rc}" -eq 2
  test -s "${OUTPUT_ROOT}/stage235-nightowls-color-reviewed.csv"
  test -s "${OUTPUT_ROOT}/stage235-nightowls-color-track-consensus.json"
  "${PYTHON}" "${TRAINING_CODE}/scripts/build_hard_crop_contact_sheets.py" \
    --manifest "${OUTPUT_ROOT}/stage235-nightowls-color-reviewed.csv" \
    --dataset-root / --output-dir "${OUTPUT_ROOT}/accepted-contact-sheets" \
    --per-group 40 --columns 5 --field color --accepted-only
  sha256sum "${OUTPUT_ROOT}/stage235-nightowls-color-reviewed.csv" \
    "${OUTPUT_ROOT}/stage235-nightowls-color-track-consensus.json" \
    > "${OUTPUT_ROOT}/SHA256SUMS"
  find "${OUTPUT_ROOT}/accepted-contact-sheets" -maxdepth 1 -type f -name '*.jpg' -print0 \
    | sort -z | xargs -0 -r sha256sum >> "${OUTPUT_ROOT}/SHA256SUMS"
  echo "STAGE235_TERMINAL_RC=${rc}"
  exit "${rc}"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse existing output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi
for required in \
  "${INPUT_ROOT}/stage234-nightowls-vehicle-tracks.csv" \
  "${TRAINING_CODE}/scripts/audit_stage235_nightowls_color_tracks.py" \
  "${TRAINING_CODE}/scripts/build_hard_crop_contact_sheets.py" \
  "${LABELS}" "${TEACHER256}" "${TEACHER288}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG}")"

tmux new-session -d -s "${SESSION}" \
  "bash '$0' --worker >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
