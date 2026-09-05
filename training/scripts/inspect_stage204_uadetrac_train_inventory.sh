#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/vcas/sources/ua-detrac-hf-v1
[[ "$(readlink -f -- "${root}")" == "${root}" ]]
[[ -d "${root}" && ! -L "${root}" ]]

printf '%s\n' 'TOP_LEVEL'
find "${root}" -mindepth 1 -maxdepth 2 -printf '%y %p\n' | sort

printf '%s\n' 'DIRECTORY_SIZES'
du -sh -- "${root}"/* 2>/dev/null | sort -h

printf '%s\n' 'TRAIN_NAMED_DIRECTORIES_ONLY'
mapfile -d '' train_dirs < <(find "${root}" -type d \( -iname '*train*' -o -iname 'Insight-MVT_Annotation_Train' \) -print0)
for train_dir in "${train_dirs[@]}"; do
  case "${train_dir,,}" in
    *test*)
      echo "refusing test-like path: ${train_dir}" >&2
      continue
      ;;
  esac
  regular_files=$(find "${train_dir}" -type f -printf . | wc -c)
  image_files=$(find "${train_dir}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -printf . | wc -c)
  xml_files=$(find "${train_dir}" -type f -iname '*.xml' -printf . | wc -c)
  bytes=$(find "${train_dir}" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum + 0}')
  printf 'path=%s regular_files=%s image_files=%s xml_files=%s bytes=%s\n' \
    "${train_dir}" "${regular_files}" "${image_files}" "${xml_files}" "${bytes}"
done

printf '%s\n' 'TRAIN_SEQUENCE_DIRECTORY_COUNTS'
for train_dir in "${train_dirs[@]}"; do
  case "${train_dir,,}" in *test*) continue ;; esac
  find "${train_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | wc -l | \
    awk -v path="${train_dir}" '{printf "path=%s child_directories=%s\n", path, $1}'
done

printf '%s\n' 'NO_TEST_PIXELS_OR_ANNOTATIONS_OPENED'
printf '%s\n' 'FIRST_TRAIN_XML_HEADER'
first_xml=$(find "${root}/original-xml/training" -maxdepth 1 -type f -iname '*.xml' -print | sort | head -n 1)
[[ -n "${first_xml}" ]]
printf 'path=%s\n' "${first_xml}"
sed -n '1,12p' "${first_xml}"
