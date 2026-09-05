#!/usr/bin/env bash
set -euo pipefail
URL='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
OUT='/root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
TMP='/root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.parallel'
SIZE=102540417
PARTS=8
mkdir -p "$TMP"
for index in $(seq 0 $((PARTS - 1))); do
  start=$((index * SIZE / PARTS))
  end=$((((index + 1) * SIZE / PARTS) - 1))
  if [ "$index" -eq $((PARTS - 1)) ]; then end=$((SIZE - 1)); fi
  part="$TMP/part-${index}"
  if [ -s "$part" ] && [ "$(stat -c '%s' "$part")" -eq $((end - start + 1)) ]; then continue; fi
  rm -f "$part"
  curl -L --fail --retry 5 --retry-delay 1 --range "${start}-${end}" "${URL}?x=1" -o "$part"
done
for index in $(seq 0 $((PARTS - 1))); do
  part="$TMP/part-${index}"
  expected_start=$((index * SIZE / PARTS))
  expected_end=$((((index + 1) * SIZE / PARTS) - 1))
  if [ "$index" -eq $((PARTS - 1)) ]; then expected_end=$((SIZE - 1)); fi
  test "$(stat -c '%s' "$part")" -eq $((expected_end - expected_start + 1))
done
cat "$TMP"/part-* > "$OUT"
test "$(stat -c '%s' "$OUT")" -eq "$SIZE"
sha256sum "$OUT"
