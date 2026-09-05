#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MANIFEST_BUILDER="${MANIFEST_BUILDER:-/root/autodl-tmp/vcas/work/domain-retrain-v1/build_hf_tree_manifest.py}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DESTINATION="${DESTINATION:-/root/autodl-tmp/vcas/sources/bmd-45}"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/vcas/manifests/domain-retrain-v1}"
FILE_MANIFEST="${FILE_MANIFEST:-${LOG_ROOT}/bmd45-files.tsv}"
DOWNLOAD_LOG="${DOWNLOAD_LOG:-${LOG_ROOT}/bmd45-files-download.log}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
PARALLEL_FILES="${PARALLEL_FILES:-2}"

mkdir -p "${DESTINATION}" "${LOG_ROOT}"

if [[ ! -s "${FILE_MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${MANIFEST_BUILDER}" \
    --endpoint "${HF_ENDPOINT}" \
    --repo iisc-aim/BMD-45 \
    --output "${FILE_MANIFEST}"
fi

download_one() {
  local path="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local encoded_path="$4"
  local output="${DESTINATION}/${path}"
  local url="${HF_ENDPOINT}/datasets/iisc-aim/BMD-45/resolve/main/${encoded_path}"
  local attempt actual_size actual_sha256

  mkdir -p "$(dirname "${output}")"
  if [[ -f "${output}" ]]; then
    actual_size="$(stat -c %s "${output}")"
    if [[ "${actual_size}" == "${expected_size}" ]]; then
      if [[ "${expected_sha256}" == "-" ]]; then
        return 0
      fi
      actual_sha256="$(sha256sum "${output}" | cut -d " " -f 1)"
      if [[ "${actual_sha256}" == "${expected_sha256}" ]]; then
        return 0
      fi
    fi
  fi

  for ((attempt = 1; attempt <= MAX_ATTEMPTS; ++attempt)); do
    if wget -4 -c -nv --timeout=60 --read-timeout=180 --tries=1 \
      "${url}" -O "${output}" >>"${DOWNLOAD_LOG}" 2>&1; then
      actual_size="$(stat -c %s "${output}")"
      if [[ "${actual_size}" != "${expected_size}" ]]; then
        sleep "$((attempt * 5))"
        continue
      fi
      if [[ "${expected_sha256}" != "-" ]]; then
        actual_sha256="$(sha256sum "${output}" | cut -d " " -f 1)"
        if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
          sleep "$((attempt * 5))"
          continue
        fi
      fi
      return 0
    fi
    sleep "$((attempt * 5))"
  done

  printf '%s failed path=%s\n' "$(date -Iseconds)" "${path}" >>"${DOWNLOAD_LOG}"
  return 1
}

export HF_ENDPOINT DESTINATION DOWNLOAD_LOG MAX_ATTEMPTS
export -f download_one

xargs -P "${PARALLEL_FILES}" -n 4 bash -c \
  'download_one "$1" "$2" "$3" "$4"' _ <"${FILE_MANIFEST}"

printf '%s all BMD-45 files downloaded and verified\n' "$(date -Iseconds)"
