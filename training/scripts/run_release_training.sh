#!/usr/bin/env bash
set -euo pipefail

VCAS_ROOT="${VCAS_ROOT:-/root/autodl-tmp/vcas}"
CODE_ROOT="${CODE_ROOT:-${VCAS_ROOT}/code}"
DATASET_ROOT="${DATASET_ROOT:-${VCAS_ROOT}/datasets/dataset-release-v1}"
ATTRIBUTE_MANIFEST="${ATTRIBUTE_MANIFEST:-${DATASET_ROOT}/attributes/attribute_manifest.csv}"
RELEASE_ROOT="${RELEASE_ROOT:-${VCAS_ROOT}/release}"
RUNS_ROOT="${RUNS_ROOT:-${RELEASE_ROOT}/runs}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${RELEASE_ROOT}/artifacts}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-${RELEASE_ROOT}/manifests}"
DEVICE="${DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SOURCE_REVISION_FILE="${SOURCE_REVISION_FILE:-${VCAS_ROOT}/manifests/SOURCE_REVISION}"
DATASET_VERSION="${DATASET_VERSION:-dataset-final-v1}"
MIN_ATTRIBUTE_CROPS="${MIN_ATTRIBUTE_CROPS:-3000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

cd "${CODE_ROOT}"

if [[ ! -f "${SOURCE_REVISION_FILE}" ]]; then
  echo "ERROR: source revision is missing: ${SOURCE_REVISION_FILE}" >&2
  exit 2
fi
CODE_REVISION="$(tr -d '[:space:]' < "${SOURCE_REVISION_FILE}")"

"${PYTHON_BIN}" training/scripts/verify_cloud_environment.py \
  --workspace-root "${VCAS_ROOT}" \
  --output "${MANIFESTS_ROOT}/cloud-environment.release-preflight.json" \
  --min-vram-gb 20 \
  --min-free-disk-gb 80

"${PYTHON_BIN}" training/scripts/validate_training_inputs.py \
  --detection-root "${DATASET_ROOT}/detection" \
  --attribute-csv "${ATTRIBUTE_MANIFEST}" \
  --labels config/vehicle_labels.v1.json \
  --min-detection-images 5000 \
  --min-attribute-crops "${MIN_ATTRIBUTE_CROPS}" \
  --require-approved \
  --summary "${MANIFESTS_ROOT}/training-input-validation.release.json"

for config in \
  training/configs/attr_release_224.json \
  training/configs/det_release_960.json
do
  "${PYTHON_BIN}" training/scripts/run_experiment.py \
    --config "${config}" \
    --dataset-root "${DATASET_ROOT}" \
    --attribute-manifest "${ATTRIBUTE_MANIFEST}" \
    --runs-root "${RUNS_ROOT}" \
    --artifacts-root "${ARTIFACTS_ROOT}" \
    --manifests-root "${MANIFESTS_ROOT}" \
    --labels config/vehicle_labels.v1.json \
    --dataset-version "${DATASET_VERSION}" \
    --run-kind formal \
    --device "${DEVICE}" \
    --code-revision "${CODE_REVISION}" \
    --auto-resume
done

cp "${ARTIFACTS_ROOT}/vehicle-det-det-release-960.onnx" \
  "${ARTIFACTS_ROOT}/vehicle-det-v1.onnx"
cp "${ARTIFACTS_ROOT}/vehicle-attr-attr-release-224.onnx" \
  "${ARTIFACTS_ROOT}/vehicle-attr-v1.onnx"

"${PYTHON_BIN}" training/scripts/validate_release_onnx.py \
  --detection-onnx "${ARTIFACTS_ROOT}/vehicle-det-v1.onnx" \
  --attribute-onnx "${ARTIFACTS_ROOT}/vehicle-attr-v1.onnx" \
  --detection-yaml "${DATASET_ROOT}/detection/vehicle_det_v1.yaml" \
  --labels config/vehicle_labels.v1.json \
  --registry models/manifests/model_registry.v1.json \
  --output "${MANIFESTS_ROOT}/release-onnx-contract.json"

"${PYTHON_BIN}" training/scripts/collect_training_artifacts.py \
  --runs-root "${RUNS_ROOT}" \
  --artifacts-root "${ARTIFACTS_ROOT}" \
  --manifests-root "${MANIFESTS_ROOT}" \
  --output "${RELEASE_ROOT}/vcas-release-training-results.tar.gz"

echo "PASS: single VCAS release training pipeline completed"
echo "Download ${RELEASE_ROOT}/vcas-release-training-results.tar.gz before shutdown"
