#!/usr/bin/env bash
set -euo pipefail

domain_session="vcas_stage54_oi_domain_audit"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
stage_root="${dataset_root}/openimages-stage54-hard-color-v1"

while tmux has-session -t "=${domain_session}" 2>/dev/null; do sleep 30; done

"${python_bin}" - "${stage_root}/domain-audit-report.json" "${stage_root}/domain-filtered-supplement.csv" <<'PY'
import csv, json, sys
from pathlib import Path
report, manifest = map(Path, sys.argv[1:])
if not report.is_file() or not manifest.is_file(): raise SystemExit("fail_closed: Stage54 domain evidence missing")
x = json.loads(report.read_text(encoding="utf-8"))
if x.get("status") not in {"pass", "label_quality_pass_complex_scene_quota_fail_auxiliary_only"}:
    raise SystemExit("fail_closed: Stage54 domain audit invalid")
q = x.get("policy", {})
if q.get("validation_or_test_used") is not False or q.get("frozen_video_used") is not False:
    raise SystemExit("fail_closed: Stage54 isolation invalid")
with manifest.open(encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
if len(rows) != int(x.get("retained_rows", -1)) or not rows:
    raise SystemExit("fail_closed: Stage54 retained row count mismatch")
PY

test ! -e "${stage_root}/contact-sheets"
exec "${python_bin}" "${project_root}/training/scripts/build_attribute_audit_mosaics.py" \
  --manifest "${stage_root}/domain-filtered-supplement.csv" \
  --dataset-root "${dataset_root}" \
  --output-dir "${stage_root}/contact-sheets" \
  --field color --per-group 40 --columns 5 --seed 20260825
