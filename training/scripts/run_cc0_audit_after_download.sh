#!/usr/bin/env bash
set -euo pipefail

download_session="vcas_cc0_download"
download_state="/root/autodl-tmp/vcas/sources/kaggle-vehicle-color-cc0-v1/download-state.json"
archive_path="/root/autodl-tmp/vcas/sources/kaggle-vehicle-color-cc0-v1/archive.zip"
output_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/kaggle-color-cc0-v1"
expected_md5="edd46f42024531019e5b0633f6e5b52a"

while tmux has-session -t "${download_session}" 2>/dev/null; do
  sleep 20
done

download_status=$(
  /root/miniconda3/bin/python - "${download_state}" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state = json.loads(state_path.read_text(encoding="utf-8"))
print(state.get("status", "missing"))
PY
)

if [[ "${download_status}" != "complete_checksum_verified" ]]; then
  echo "fail_closed: download status is ${download_status}" >&2
  exit 2
fi

if [[ -e "${output_root}" ]]; then
  echo "fail_closed: output root already exists: ${output_root}" >&2
  exit 3
fi

exec /root/miniconda3/bin/python \
  /root/autodl-tmp/vcas/code/training/scripts/build_kaggle_vehicle_color_manifest.py \
  --archive "${archive_path}" \
  --output-root "${output_root}" \
  --expected-md5 "${expected_md5}" \
  --dataset-version kaggle-color-cc0-v1 \
  --min-side 48 \
  --near-duplicate-hamming 2
