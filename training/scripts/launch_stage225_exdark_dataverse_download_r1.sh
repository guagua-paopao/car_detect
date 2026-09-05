#!/usr/bin/env bash
set -euo pipefail

SESSION="VCAS-DL-EXDARK225"
SOURCE_ROOT="/root/autodl-tmp/vcas/sources/exdark-stage225-dataverse"
LOG="/root/autodl-tmp/vcas/logs/stage225-exdark-dataverse-download-r1.log"
DATASET_API="https://researchdata.um.edu.my/api/datasets/:persistentId/?persistentId=doi:10.22452/RD/JUSQEK"
CAR_URL="https://researchdata.um.edu.my/api/access/datafile/256"
BUS_URL="https://researchdata.um.edu.my/api/access/datafile/259"
CAR_MD5="c4c8e6ced85ac27fff16dc5560461f21"
BUS_MD5="a8a7363be3282873e3c31418834c9560"

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
if [[ -z "${AVAILABLE_KB}" || "${AVAILABLE_KB}" -lt 4194304 ]]; then
  echo "need at least 4 GiB free; available_kb=${AVAILABLE_KB:-unknown}" >&2
  exit 4
fi

INNER="set -euo pipefail; \
  curl -fL --retry 12 --retry-all-errors --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 -o '${SOURCE_ROOT}/dataset-metadata.json.part' '${DATASET_API}'; \
  mv '${SOURCE_ROOT}/dataset-metadata.json.part' '${SOURCE_ROOT}/dataset-metadata.json'; \
  curl -fL --retry 20 --retry-all-errors --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 -C - -o '${SOURCE_ROOT}/car.tar.gz.part' '${CAR_URL}'; \
  mv '${SOURCE_ROOT}/car.tar.gz.part' '${SOURCE_ROOT}/car.tar.gz'; \
  curl -fL --retry 20 --retry-all-errors --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 -C - -o '${SOURCE_ROOT}/bus.tar.gz.part' '${BUS_URL}'; \
  mv '${SOURCE_ROOT}/bus.tar.gz.part' '${SOURCE_ROOT}/bus.tar.gz'; \
  printf '%s  %s\n%s  %s\n' '${CAR_MD5}' '${SOURCE_ROOT}/car.tar.gz' '${BUS_MD5}' '${SOURCE_ROOT}/bus.tar.gz' > '${SOURCE_ROOT}/MD5SUMS.expected'; \
  md5sum -c '${SOURCE_ROOT}/MD5SUMS.expected'; \
  tar -tzf '${SOURCE_ROOT}/car.tar.gz' >/dev/null; \
  tar -tzf '${SOURCE_ROOT}/bus.tar.gz' >/dev/null; \
  sha256sum '${SOURCE_ROOT}/car.tar.gz' '${SOURCE_ROOT}/bus.tar.gz' '${SOURCE_ROOT}/dataset-metadata.json' > '${SOURCE_ROOT}/SHA256SUMS'; \
  du -sh '${SOURCE_ROOT}'; \
  df -h /root/autodl-tmp; \
  echo STAGE225_DOWNLOAD_COMPLETE"

tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER}\" >> '${LOG}' 2>&1"
echo "started ${SESSION}; log=${LOG}; source_root=${SOURCE_ROOT}"
