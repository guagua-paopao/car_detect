#!/usr/bin/env bash
set -euo pipefail

VCAS_ROOT="${VCAS_ROOT:-/root/autodl-tmp/vcas}"
CODE_ROOT="${CODE_ROOT:-${VCAS_ROOT}/code}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-${VCAS_ROOT}/manifests}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

mkdir -p \
  "${VCAS_ROOT}/datasets" \
  "${VCAS_ROOT}/runs" \
  "${VCAS_ROOT}/artifacts" \
  "${MANIFESTS_ROOT}"

cd "${CODE_ROOT}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r training/requirements-cloud.txt
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m pip freeze > "${MANIFESTS_ROOT}/requirements-lock.txt"

"${PYTHON_BIN}" training/scripts/verify_cloud_environment.py \
  --workspace-root "${VCAS_ROOT}" \
  --output "${MANIFESTS_ROOT}/cloud-environment.json" \
  --min-vram-gb 20 \
  --min-free-disk-gb 80

find training config/vehicle_labels.v1.json \
  -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${MANIFESTS_ROOT}/SOURCE_FILES.sha256"

sha256sum "${MANIFESTS_ROOT}/SOURCE_FILES.sha256" \
  | awk '{print $1}' \
  > "${MANIFESTS_ROOT}/SOURCE_REVISION"

echo "PASS: AutoDL environment is ready"
echo "source_revision=$(cat "${MANIFESTS_ROOT}/SOURCE_REVISION")"
