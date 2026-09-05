#!/usr/bin/env bash
set -euo pipefail

output_root=/root/autodl-tmp/vcas/sources/stage183-myvid-v1
archive="${output_root}/MY-VID_v1.0_2025-08-14.zip"
mkdir -p "${output_root}"

curl --fail --location --silent --show-error \
  https://zenodo.org/api/records/16866509 \
  --output "${output_root}/zenodo-record.json"
curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
  https://zenodo.org/api/records/16866509/files/MY-VID_v1.0_2025-08-14.zip/content \
  --output "${archive}"

actual_md5="$(md5sum "${archive}" | cut -d' ' -f1)"
test "${actual_md5}" = 8ec1f03001fae99739fbe592916a9e86
unzip -tq "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
