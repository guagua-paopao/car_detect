#!/usr/bin/env bash
set -euo pipefail

audit_session="vcas_stage52_color_audit"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
source_root="${dataset_root}/bmd45-stage52-independent-color-v1"
manifest="${source_root}/attribute_supplement.csv"
report="${source_root}/audit-report.json"
output_dir="${source_root}/contact-sheets"

while tmux has-session -t "=${audit_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${report}" "${manifest}" <<'PY'
import csv
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
manifest = Path(sys.argv[2])
if not report.is_file() or not manifest.is_file():
    raise SystemExit("fail_closed: Stage52 audit evidence is missing")
data = json.loads(report.read_text(encoding="utf-8"))
if data.get("status") != "pass":
    raise SystemExit("fail_closed: Stage52 independent color audit did not pass")
if data.get("policy", {}).get("validation_or_test_used") is not False:
    raise SystemExit("fail_closed: validation/test isolation evidence is invalid")
if data.get("policy", {}).get("frozen_video_used") is not False:
    raise SystemExit("fail_closed: frozen-video isolation evidence is invalid")
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != int(data.get("accepted_rows", -1)):
    raise SystemExit("fail_closed: Stage52 report/manifest row count mismatch")
if not rows:
    raise SystemExit("fail_closed: no accepted Stage52 rows")
PY

test ! -e "${output_dir}"
exec "${python_bin}" "${project_root}/training/scripts/build_attribute_audit_mosaics.py" \
  --manifest "${manifest}" \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --field color \
  --per-group 40 \
  --columns 5 \
  --seed 20260824
