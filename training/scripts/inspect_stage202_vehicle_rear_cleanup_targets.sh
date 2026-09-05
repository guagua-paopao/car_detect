#!/usr/bin/env bash
set -euo pipefail

archive=/root/autodl-tmp/vcas/sources/vehicle-rear/data.tgz
parts=/root/autodl-tmp/vcas/sources/vehicle-rear/.data.tgz.parts
selected=/root/autodl-tmp/vcas/sources/vehicle-rear-stage149/selected-r3

[[ "$(readlink -f -- "${archive}")" == "${archive}" ]]
[[ -f "${archive}" && ! -L "${archive}" ]]
[[ "$(stat -c %s -- "${archive}")" == 6451005771 ]]
[[ "$(sha256sum -- "${archive}" | awk '{print $1}')" == 511d92ed433d35e9861ced29b6b3b3a44f317122a42cb8a37fce654f5887858c ]]
[[ "$(readlink -f -- "${parts}")" == "${parts}" ]]
[[ -d "${parts}" && ! -L "${parts}" ]]
[[ -d "${selected}" && ! -L "${selected}" ]]
if find "${parts}" -type l -print -quit | grep -q .; then
  echo "symlink found under parts; refusing" >&2
  exit 8
fi
non_regular=$(find "${parts}" -mindepth 1 ! -type f ! -type d -print | wc -l)
file_count=$(find "${parts}" -type f -printf . | wc -c)
directory_count=$(find "${parts}" -mindepth 1 -type d -printf . | wc -c)
parts_bytes=$(find "${parts}" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
selected_files=$(find "${selected}" -type f -printf . | wc -c)
selected_bytes=$(find "${selected}" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
printf 'archive=%s bytes=%s sha256=%s\n' "${archive}" "$(stat -c %s -- "${archive}")" "$(sha256sum -- "${archive}" | awk '{print $1}')"
printf 'parts=%s files=%s directories=%s bytes=%s non_regular=%s\n' "${parts}" "${file_count}" "${directory_count}" "${parts_bytes}" "${non_regular}"
printf 'retained_selected=%s files=%s bytes=%s\n' "${selected}" "${selected_files}" "${selected_bytes}"
find "${parts}" -maxdepth 1 -type f -printf '%f %s\n' | sort | sed -n '1,5p;$p'
