#!/usr/bin/env bash
set -euo pipefail

audit_session="vcas_stage53_oi_teacher_audit"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/dataset-v1/attributes"
stage_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage53-color-v1"
manifest="${stage_root}/teacher-approved.csv"
report="${stage_root}/teacher-audit-report.json"
output_dir="${stage_root}/contact-sheets"

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
    raise SystemExit("fail_closed: Stage53 teacher audit evidence is missing")
data = json.loads(report.read_text(encoding="utf-8"))
if data.get("status") != "pass":
    raise SystemExit("fail_closed: Stage53 teacher audit did not pass")
policy = data.get("policy", {})
if policy.get("test_split_not_used") is not True:
    raise SystemExit("fail_closed: Stage53 test isolation evidence is invalid")
if policy.get("frozen_video_not_used") is not True:
    raise SystemExit("fail_closed: Stage53 frozen-video isolation evidence is invalid")
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
approved = [row for row in rows if row.get("color_teacher_consensus") == "accepted"]
if len(approved) != int(data.get("accepted_rows", -1)) or not approved:
    raise SystemExit("fail_closed: Stage53 accepted row count mismatch")
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
