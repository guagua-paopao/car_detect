#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-DL-EXDARK224"
SOURCE_ROOT="/root/autodl-tmp/vcas/sources/exdark-stage224"
REPO_ROOT="${SOURCE_ROOT}/official-repository"
IMAGE_PART="${SOURCE_ROOT}/ExDark.zip.part"
IMAGE_ZIP="${SOURCE_ROOT}/ExDark.zip"
ANNOTATION_PART="${SOURCE_ROOT}/ExDark_Anno.zip.part"
ANNOTATION_ZIP="${SOURCE_ROOT}/ExDark_Anno.zip"
LOG="/root/autodl-tmp/vcas/logs/stage224-exdark-official-download-r1.log"
PYTHON="/root/miniconda3/bin/python3"
IMAGE_ID="1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"
ANNOTATION_ID="1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${SOURCE_ROOT}" ]]; then
  echo "refusing to reuse existing source root: ${SOURCE_ROOT}" >&2
  exit 3
fi
mkdir -p "${SOURCE_ROOT}" "$(dirname "${LOG}")"

AVAILABLE_KB="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
if [[ -z "${AVAILABLE_KB}" || "${AVAILABLE_KB}" -lt 8388608 ]]; then
  echo "need at least 8 GiB free before ExDark download; available_kb=${AVAILABLE_KB:-unknown}" >&2
  exit 4
fi

INNER="set -euo pipefail; \
  git clone --depth 1 https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.git '${REPO_ROOT}'; \
  git -C '${REPO_ROOT}' rev-parse HEAD > '${SOURCE_ROOT}/official-repository.commit'; \
  if ! '${PYTHON}' -c 'import gdown' >/dev/null 2>&1; then '${PYTHON}' -m pip install --no-cache-dir 'gdown==5.2.0'; fi; \
  '${PYTHON}' -m gdown --id '${IMAGE_ID}' --output '${IMAGE_PART}'; \
  mv '${IMAGE_PART}' '${IMAGE_ZIP}'; \
  '${PYTHON}' -m gdown --id '${ANNOTATION_ID}' --output '${ANNOTATION_PART}'; \
  mv '${ANNOTATION_PART}' '${ANNOTATION_ZIP}'; \
  unzip -tq '${IMAGE_ZIP}'; \
  unzip -tq '${ANNOTATION_ZIP}'; \
  sha256sum '${IMAGE_ZIP}' '${ANNOTATION_ZIP}' '${REPO_ROOT}/README.md' '${REPO_ROOT}/LICENSE' > '${SOURCE_ROOT}/SHA256SUMS'; \
  df -h /root/autodl-tmp; \
  echo STAGE224_DOWNLOAD_COMPLETE"

tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER}\" >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; source_root=${SOURCE_ROOT}"
