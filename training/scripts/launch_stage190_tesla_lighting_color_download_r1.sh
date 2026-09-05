#!/usr/bin/env bash
set -euo pipefail

source_root=/root/autodl-tmp/vcas/sources/stage189-tesla-lighting-color
archive="${source_root}/Tesla-dataset.zip"
record="${source_root}/zenodo-record-19814157.json"
log="${source_root}/stage190-download.log"
expected_bytes=8841072987
minimum_post_download_buffer=3000000000
url=https://zenodo.org/api/records/19814157/files/Tesla-dataset.zip/content

mkdir -p "${source_root}"
exec > >(tee -a "${log}") 2>&1
test -f "${source_root}/tesla_dataset_labels.csv"
test -f "${source_root}/labels_description.txt"
test -f "${source_root}/label-studio-code.xml"
available_bytes="$(df --output=avail -B1 "${source_root}" | tail -n 1 | tr -d ' ')"
required_bytes=$((expected_bytes + minimum_post_download_buffer))
test -n "${available_bytes}"
test "${available_bytes}" -ge "${required_bytes}"
if test -f "${archive}"; then
  current_bytes="$(stat -c %s "${archive}")"
  test "${current_bytes}" -le "${expected_bytes}"
fi

curl --fail --location --silent --show-error \
  https://zenodo.org/api/records/19814157 \
  --output "${record}"
aria2c \
  --continue=true \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=8M \
  --file-allocation=none \
  --max-tries=0 \
  --retry-wait=5 \
  --timeout=30 \
  --dir="${source_root}" \
  --out="$(basename "${archive}")" \
  "${url}"

test "$(stat -c %s "${archive}")" = "${expected_bytes}"
test "$(md5sum "${archive}" | cut -d' ' -f1)" = 9ce3079d42273f07374bd25344809d2f
unzip -tq "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
