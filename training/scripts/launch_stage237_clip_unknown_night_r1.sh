#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE237-CLIP-NIGHT-R1"
TRAINING_CODE="/root/autodl-tmp/vcas/code/training"
PYTHON="/root/autodl-tmp/vcas/venvs/stage237-clip/bin/python"
INPUT_MANIFEST="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE179-FULL-COLOR-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv"
CALIBRATION="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE237-CLIP-UNKNOWN-NIGHT-R1/inputs/stage223-color-night-agent-visual-overlay.csv"
NIGHTOWLS="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE234-NIGHTOWLS-VEHICLE-TRACKS-R1/stage234-nightowls-vehicle-tracks.csv"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE237-CLIP-UNKNOWN-NIGHT-R1"
CACHE="/root/autodl-tmp/vcas/weights/open_clip"
WEIGHT="${CACHE}/ViT-B-32.pt"
LOG="/root/autodl-tmp/vcas/logs/stage237-clip-unknown-night-r1.log"

run_worker() {
  set +e
  "${PYTHON}" "${TRAINING_CODE}/scripts/audit_stage237_clip_unknown_night.py" \
    --active-manifest "${INPUT_MANIFEST}" \
    --calibration-overlay "${CALIBRATION}" \
    --nightowls-manifest "${NIGHTOWLS}" \
    --model ViT-B-32-quickgelu --pretrained "${WEIGHT}" --cache-dir "${CACHE}" \
    --output-candidates "${OUTPUT_ROOT}/stage237-clip-night-candidates.csv" \
    --output-calibration "${OUTPUT_ROOT}/stage237-clip-calibration.csv" \
    --output-report "${OUTPUT_ROOT}/stage237-clip-unknown-night-audit.json" \
    --minimum-calibration-precision 0.90 \
    --minimum-calibration-recall 0.40 \
    --minimum-nightowls-recall 0.50 \
    --maximum-known-daylight-fpr 0.05 \
    --nightowls-diagnostic-sample 1200 \
    --daylight-diagnostic-sample 1200 \
    --batch-size 256 --workers 8 --device cuda
  rc=$?
  set -e
  test "${rc}" -eq 0 -o "${rc}" -eq 2
  test -s "${OUTPUT_ROOT}/stage237-clip-night-candidates.csv"
  test -s "${OUTPUT_ROOT}/stage237-clip-calibration.csv"
  test -s "${OUTPUT_ROOT}/stage237-clip-unknown-night-audit.json"
  sha256sum \
    "${OUTPUT_ROOT}/stage237-clip-night-candidates.csv" \
    "${OUTPUT_ROOT}/stage237-clip-calibration.csv" \
    "${OUTPUT_ROOT}/stage237-clip-unknown-night-audit.json" \
    > "${OUTPUT_ROOT}/SHA256SUMS"
  echo "STAGE237_TERMINAL_RC=${rc}"
  exit "${rc}"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi
for required in "${PYTHON}" "${TRAINING_CODE}/scripts/audit_stage237_clip_unknown_night.py" \
  "${INPUT_MANIFEST}" "${CALIBRATION}" "${NIGHTOWLS}" "${WEIGHT}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
for output in \
  "${OUTPUT_ROOT}/stage237-clip-night-candidates.csv" \
  "${OUTPUT_ROOT}/stage237-clip-calibration.csv" \
  "${OUTPUT_ROOT}/stage237-clip-unknown-night-audit.json"; do
  [[ ! -e "${output}" ]] || { echo "refusing to overwrite: ${output}" >&2; exit 3; }
done
mkdir -p "${OUTPUT_ROOT}/inputs" "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash '$0' --worker >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
