#!/usr/bin/env bash
set -euo pipefail

output_root=/root/autodl-tmp/vcas/sources/stage185-myvid-v2
archive="${output_root}/MY-VID_v2.0_2025-12-09.zip"
mkdir -p "${output_root}"

curl --fail --location --silent --show-error \
  https://zenodo.org/api/records/17861464 \
  --output "${output_root}/zenodo-record.json"
curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
  https://zenodo.org/api/records/17861464/files/MY-VID_v2.0_2025-12-09.zip/content \
  --output "${archive}"

actual_size="$(stat -c %s "${archive}")"
actual_md5="$(md5sum "${archive}" | cut -d' ' -f1)"
test "${actual_size}" = 2637612140
test "${actual_md5}" = cd506cf6e6ba8ef41d030b08bcc9d006
unzip -tq "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
