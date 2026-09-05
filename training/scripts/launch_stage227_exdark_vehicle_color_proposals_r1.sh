#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-STAGE227-EXDARK-COLOR-R1"
INPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE226-EXDARK-TRAIN-ONLY-R1"
OUTPUT_ROOT="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE227-EXDARK-COLOR-PROPOSALS-R1"
SCRIPT="/root/autodl-tmp/vcas/scripts/build_stage227_exdark_vehicle_color_proposals.py"
FOREGROUND_HELPER="/root/autodl-tmp/vcas/code/training/scripts/build_bmd45_track_color_pseudolabels.py"
DETECTOR="/root/autodl-tmp/vcas/weights/yolo11x.pt"
STAGE177="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv"
STAGE188="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1/stage188-myvid-v2-component-manifest.csv"
LOG="/root/autodl-tmp/vcas/logs/stage227-exdark-vehicle-color-proposals-r1.log"
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
  "${INPUT_ROOT}/report.json" \
  "${INPUT_ROOT}/train_manifest.csv" \
  "${SCRIPT}" "${FOREGROUND_HELPER}" "${DETECTOR}" "${STAGE177}" "${STAGE188}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 4; }
done
"${PYTHON}" - "${INPUT_ROOT}/report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "complete_train_pixels_only"
assert report["selected_train_images"] == 500
assert report["validation_pixels_opened"] == 0
assert report["test_pixels_opened"] == 0
assert report["frozen_video_used"] is False
PY

INNER="set -euo pipefail; \
  '${PYTHON}' '${SCRIPT}' \
    --input-manifest '${INPUT_ROOT}/train_manifest.csv' \
    --input-root '${INPUT_ROOT}' \
    --detector '${DETECTOR}' \
    --foreground-helper '${FOREGROUND_HELPER}' \
    --cross-source-manifest '${STAGE177}' \
    --cross-source-manifest '${STAGE188}' \
    --output-root '${OUTPUT_ROOT}' \
    --batch-size 8 --image-size 960 --device 0 --near-distance 2; \
  echo STAGE227_COMPLETE"

tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER}\" >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; output_root=${OUTPUT_ROOT}"
