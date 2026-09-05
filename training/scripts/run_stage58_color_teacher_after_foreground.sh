#!/usr/bin/env bash
set -euo pipefail

foreground_session="vcas_stage58_color_foreground"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
stage_root="${dataset_root}/openimages-stage58-existing-hard-color-v1"

while tmux has-session -t "=${foreground_session}" 2>/dev/null; do sleep 30; done

"${python_bin}" - "${stage_root}/foreground-proposal-report.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit("fail_closed: Stage58 proposal report missing")
x = json.loads(p.read_text(encoding="utf-8"))
if x.get("status") != "pass": raise SystemExit("fail_closed: Stage58 proposal failed")
q = x.get("policy", {})
if q.get("train_images_opened_only") is not True:
    raise SystemExit("fail_closed: Stage58 train isolation invalid")
if q.get("validation_or_test_images_opened") is not False:
    raise SystemExit("fail_closed: Stage58 validation/test isolation invalid")
if q.get("frozen_video_used") is not False:
    raise SystemExit("fail_closed: Stage58 frozen-video isolation invalid")
PY

exec "${python_bin}" "${project_root}/training/scripts/audit_color_multiteacher_consensus.py" \
  --manifest "${stage_root}/foreground-proposals.csv" \
  --dataset-root "${dataset_root}" \
  --labels "${project_root}/config/vehicle_labels.v1.json" \
  --checkpoint "production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --checkpoint "research224=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1/best.pt" \
  --checkpoint "permissive224=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1/best.pt" \
  --output-manifest "${stage_root}/teacher-approved.csv" \
  --output-report "${stage_root}/teacher-audit-report.json" \
  --confidence 0.80 \
  --minimum-total 300 \
  --minimum-classes 6 \
  --batch-size 128 \
  --workers 6 \
  --device cuda
