#!/usr/bin/env bash
set -euo pipefail

VCAS_ROOT="${VCAS_ROOT:-/root/autodl-tmp/vcas}"
CODE_ROOT="${CODE_ROOT:-${VCAS_ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${VCAS_ROOT}/datasets/dataset-v1}"
ATTRIBUTE_MANIFEST="${ATTRIBUTE_MANIFEST:-${DATASET_ROOT}/attributes/attribute_manifest.csv}"
RUNS_ROOT="${RUNS_ROOT:-${VCAS_ROOT}/runs}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${VCAS_ROOT}/artifacts}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-${VCAS_ROOT}/manifests}"
DEVICE="${DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

cd "${CODE_ROOT}"

if [[ ! -f "${MANIFESTS_ROOT}/SOURCE_REVISION" ]]; then
  echo "ERROR: run training/scripts/bootstrap_autodl.sh first" >&2
  exit 2
fi

CODE_REVISION="$(tr -d '[:space:]' < "${MANIFESTS_ROOT}/SOURCE_REVISION")"
if [[ -z "${CODE_REVISION}" ]]; then
  echo "ERROR: source revision is empty" >&2
  exit 2
fi

"${PYTHON_BIN}" training/scripts/verify_cloud_environment.py \
  --workspace-root "${VCAS_ROOT}" \
  --output "${MANIFESTS_ROOT}/cloud-environment.preflight.json" \
  --min-vram-gb 20 \
  --min-free-disk-gb 80

"${PYTHON_BIN}" training/scripts/validate_training_inputs.py \
  --detection-root "${DATASET_ROOT}/detection" \
  --attribute-csv "${ATTRIBUTE_MANIFEST}" \
  --labels config/vehicle_labels.v1.json \
  --min-detection-images 3000 \
  --min-attribute-crops 8000 \
  --require-approved \
  --summary "${MANIFESTS_ROOT}/training-input-validation.json"

for config in \
  training/configs/det_a_640.json \
  training/configs/det_b_960.json \
  training/configs/attr_a_224.json \
  training/configs/attr_b_256.json
do
  "${PYTHON_BIN}" training/scripts/run_experiment.py \
    --config "${config}" \
    --dataset-root "${DATASET_ROOT}" \
    --attribute-manifest "${ATTRIBUTE_MANIFEST}" \
    --runs-root "${RUNS_ROOT}" \
    --artifacts-root "${ARTIFACTS_ROOT}" \
    --manifests-root "${MANIFESTS_ROOT}" \
    --labels config/vehicle_labels.v1.json \
    --dataset-version dataset-v1 \
    --run-kind formal \
    --device "${DEVICE}" \
    --code-revision "${CODE_REVISION}" \
    --auto-resume
done

"${PYTHON_BIN}" training/scripts/collect_training_artifacts.py \
  --runs-root "${RUNS_ROOT}" \
  --artifacts-root "${ARTIFACTS_ROOT}" \
  --manifests-root "${MANIFESTS_ROOT}" \
  --output "${VCAS_ROOT}/vcas-formal-training-results.tar.gz"

echo "PASS: all formal VCAS experiments completed"
echo "Download ${VCAS_ROOT}/vcas-formal-training-results.tar.gz before shutting down"
