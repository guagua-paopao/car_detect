#!/usr/bin/env bash
set -euo pipefail

archive=/root/autodl-tmp/vcas/sources/vehicle-rear/data.tgz
parts=/root/autodl-tmp/vcas/sources/vehicle-rear/.data.tgz.parts
selected=/root/autodl-tmp/vcas/sources/vehicle-rear-stage149/selected-r3
expected_total=6451005771
expected_hash=511d92ed433d35e9861ced29b6b3b3a44f317122a42cb8a37fce654f5887858c

[[ "$(readlink -f -- "${archive}")" == "${archive}" ]]
[[ -f "${archive}" && ! -L "${archive}" ]]
[[ "$(stat -c %s -- "${archive}")" -eq "${expected_total}" ]]
[[ "$(sha256sum -- "${archive}" | awk '{print $1}')" == "${expected_hash}" ]]

[[ "$(readlink -f -- "${parts}")" == "${parts}" ]]
[[ -d "${parts}" && ! -L "${parts}" ]]
[[ "$(readlink -f -- "${selected}")" == "${selected}" ]]
[[ -d "${selected}" && ! -L "${selected}" ]]

mapfile -d '' part_files < <(find "${parts}" -mindepth 1 -maxdepth 1 -type f -print0 | sort -z)
[[ "${#part_files[@]}" -eq 97 ]]
[[ "$(find "${parts}" -mindepth 1 -maxdepth 1 -type d -printf . | wc -c)" -eq 0 ]]
[[ "$(find "${parts}" -mindepth 1 -maxdepth 1 -type l -printf . | wc -c)" -eq 0 ]]
[[ "$(find "${parts}" -mindepth 1 -maxdepth 1 ! -type f -printf . | wc -c)" -eq 0 ]]

parts_total=0
for index in $(seq 0 96); do
  start=$((index * 67108864))
  if [[ "${index}" -lt 96 ]]; then
    end=$((start + 67108864 - 1))
    expected_size=67108864
  else
    end=$((expected_total - 1))
    expected_size=$((expected_total - start))
  fi
  expected_name=$(printf 'part-%05d-%d-%d.bin' "${index}" "${start}" "${end}")
  expected_path="${parts}/${expected_name}"
  [[ -f "${expected_path}" && ! -L "${expected_path}" ]]
  actual_size=$(stat -c %s -- "${expected_path}")
  [[ "${actual_size}" -eq "${expected_size}" ]]
  parts_total=$((parts_total + actual_size))
done
[[ "${parts_total}" -eq "${expected_total}" ]]

if command -v fuser >/dev/null 2>&1; then
  set +e
  fuser -- "${archive}" "${part_files[@]}" >/tmp/stage202-fuser.out 2>/tmp/stage202-fuser.err
  fuser_rc=$?
  set -e
  if [[ "${fuser_rc}" -eq 0 ]]; then
    echo 'A cleanup target is in use; refusing deletion.' >&2
    cat /tmp/stage202-fuser.out /tmp/stage202-fuser.err >&2
    exit 20
  fi
  [[ "${fuser_rc}" -eq 1 ]]
fi

selected_files=$(find "${selected}" -type f -printf . | wc -c)
selected_bytes=$(find "${selected}" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum + 0}')
[[ "${selected_files}" -eq 7014 ]]
[[ "${selected_bytes}" -eq 1728937910 ]]

printf 'validated archive=%s bytes=%d\n' "${archive}" "${expected_total}"
printf 'validated parts=%s files=%d bytes=%d\n' "${parts}" "${#part_files[@]}" "${parts_total}"
printf 'preserving selected=%s files=%s bytes=%s\n' "${selected}" "${selected_files}" "${selected_bytes}"

for part_file in "${part_files[@]}"; do
  rm -- "${part_file}"
done
rmdir -- "${parts}"
rm -- "${archive}"

[[ ! -e "${archive}" ]]
[[ ! -e "${parts}" ]]
[[ -d "${selected}" && ! -L "${selected}" ]]

printf 'removed_bytes=%d\n' "$((expected_total * 2))"
df -B1 --output=size,used,avail,pcent,target /root/autodl-tmp | tail -n 1
