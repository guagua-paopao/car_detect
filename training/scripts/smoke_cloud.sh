#!/usr/bin/env bash
set -euo pipefail

VCAS_ROOT="${VCAS_ROOT:-/root/autodl-tmp/vcas}"
CODE_ROOT="${CODE_ROOT:-${VCAS_ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${VCAS_ROOT}/datasets/dataset-pilot-v1}"
RUNS_ROOT="${RUNS_ROOT:-${VCAS_ROOT}/runs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${VCAS_ROOT}/artifacts}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${VCAS_ROOT}/manifests}"
RUN_ATTRIBUTE="${RUN_ATTRIBUTE:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

mkdir -p "${RUNS_ROOT}" "${ARTIFACT_ROOT}" "${MANIFEST_ROOT}"
cd "${CODE_ROOT}"

nvidia-smi | tee "${MANIFEST_ROOT}/nvidia-smi.txt"
"${PYTHON_BIN}" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" \
  | tee "${MANIFEST_ROOT}/torch-environment.txt"
"${PYTHON_BIN}" -m pip freeze > "${MANIFEST_ROOT}/requirements-lock.txt"

"${PYTHON_BIN}" training/scripts/validate_training_inputs.py \
  --detection-root "${DATASET_ROOT}/detection" \
  --min-detection-images 500 \
  --summary "${MANIFEST_ROOT}/detection-validation.json"

"${PYTHON_BIN}" training/scripts/train_detector.py \
  --data "${DATASET_ROOT}/detection/vehicle_det_pilot.yaml" \
  --model yolo11s.pt \
  --imgsz 640 \
  --epochs 3 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --project "${RUNS_ROOT}" \
  --name DET-SMOKE-640

if [[ "${RUN_ATTRIBUTE}" == "1" ]]; then
  "${PYTHON_BIN}" training/scripts/validate_training_inputs.py \
    --attribute-csv "${DATASET_ROOT}/attributes/attribute_manifest.csv" \
    --labels config/vehicle_labels.v1.json \
    --min-attribute-crops 300 \
    --summary "${MANIFEST_ROOT}/attribute-validation.json"

  "${PYTHON_BIN}" training/scripts/train_attribute.py \
    --manifest "${DATASET_ROOT}/attributes/attribute_manifest.csv" \
    --labels config/vehicle_labels.v1.json \
    --input-size 224 \
    --epochs 3 \
    --batch-size 32 \
    --workers 4 \
    --device cuda \
    --output-dir "${RUNS_ROOT}/ATTR-SMOKE-224"

  "${PYTHON_BIN}" training/scripts/export_attribute_onnx.py \
    --checkpoint "${RUNS_ROOT}/ATTR-SMOKE-224/best.pt" \
    --output "${ARTIFACT_ROOT}/vehicle-attr-smoke-224.onnx"
fi

echo "PASS: VCAS cloud smoke finished; copy runs/artifacts/manifests before shutdown"
