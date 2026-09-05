#!/usr/bin/env bash
set -euo pipefail

output_root=/root/autodl-tmp/vcas/sources/stage200-hvsd-night-r1
archive="${output_root}/NIGHT_Experiment.zip"
url=https://data.mendeley.com/public-files/datasets/zdgmdg7p25/files/13d55483-efbf-4727-80af-5e190bdc9e48/file_downloaded
expected_bytes=4235324
expected_sha=3cc2b812220d765998e5976b9abeb9c05ab5101337a625f0b74f81994f8773ba

test ! -e "${output_root}"
mkdir -p "${output_root}"
curl --fail --location --retry 5 --retry-delay 2 --connect-timeout 15 \
  --output "${archive}.part" "${url}"
test "$(stat -c %s -- "${archive}.part")" = "${expected_bytes}"
test "$(sha256sum -- "${archive}.part" | awk '{print $1}')" = "${expected_sha}"
mv -- "${archive}.part" "${archive}"
unzip -tq "${archive}" >/dev/null
printf '%s  %s\n' "${expected_sha}" "$(basename "${archive}")" > "${archive}.sha256"
unzip -Z1 "${archive}" > "${output_root}/archive-members.txt"
sha256sum "${output_root}/archive-members.txt" > "${output_root}/archive-members.txt.sha256"
echo "STAGE200_DOWNLOAD_COMPLETE bytes=${expected_bytes} sha256=${expected_sha}"
