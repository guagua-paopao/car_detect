#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/vcas/sources/stage193-abtd-v3
url=https://data.mendeley.com/public-files/datasets/2f9jp8bj45/files/c5510f1a-8040-4697-a099-54915684aac5/file_downloaded
expected_bytes=1885610631
expected_sha=40c57c19488e05ac6cc50b724b493f7a0827a45b85c20dbdd79c02cee120ac28

[[ -d "$root" ]] || { echo "missing ABTD source root" >&2; exit 2; }
[[ ! -e "$root/images.zip" ]] || { echo "refusing to overwrite images.zip" >&2; exit 3; }
[[ ! -e "$root/images.zip.aria2" ]] || { echo "unexpected stale aria2 control file" >&2; exit 4; }

aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=1M \
  --file-allocation=none --auto-file-renaming=false --allow-overwrite=false \
  --retry-wait=2 --max-tries=20 --timeout=30 --connect-timeout=20 \
  --dir="$root" --out=images.zip "$url"

[[ "$(stat -c %s "$root/images.zip")" == "$expected_bytes" ]] || { echo "ABTD image size mismatch" >&2; exit 5; }
actual_sha=$(sha256sum "$root/images.zip" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || { echo "ABTD image SHA256 mismatch" >&2; exit 6; }
unzip -t "$root/images.zip" >/dev/null
printf '%s  %s\n' "$actual_sha" images.zip >"$root/images.zip.sha256"
echo "STAGE194_COMPLETE bytes=$expected_bytes sha256=$actual_sha"
