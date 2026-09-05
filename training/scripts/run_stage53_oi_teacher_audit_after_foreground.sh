#!/usr/bin/env bash
set -euo pipefail

proposal_session="vcas_stage53_oi_foreground"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
dataset_root="/root/autodl-tmp/vcas/datasets/dataset-v1/attributes"
stage_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage53-color-v1"
proposal_manifest="${stage_root}/foreground-proposals.csv"
proposal_report="${stage_root}/foreground-proposal-report.json"
approved_manifest="${stage_root}/teacher-approved.csv"
audit_report="${stage_root}/teacher-audit-report.json"

while tmux has-session -t "=${proposal_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${proposal_report}" "${proposal_manifest}" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
manifest = Path(sys.argv[2])
if not report.is_file() or not manifest.is_file():
    raise SystemExit("fail_closed: Stage53 foreground proposal evidence is missing")
data = json.loads(report.read_text(encoding="utf-8"))
if data.get("status") != "pass":
    raise SystemExit("fail_closed: Stage53 foreground proposal did not pass")
policy = data.get("policy", {})
if policy.get("train_images_opened_only") is not True:
    raise SystemExit("fail_closed: Stage53 proposal train-only evidence is invalid")
if policy.get("validation_or_test_images_opened") is not False:
    raise SystemExit("fail_closed: Stage53 proposal validation/test isolation is invalid")
if policy.get("frozen_video_used") is not False:
    raise SystemExit("fail_closed: Stage53 proposal frozen-video isolation is invalid")
PY

exec "${python_bin}" "${project_root}/training/scripts/audit_color_multiteacher_consensus.py" \
  --manifest "${proposal_manifest}" \
  --dataset-root "${dataset_root}" \
  --labels "${project_root}/config/vehicle_labels.v1.json" \
  --checkpoint "production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --checkpoint "research224=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1/best.pt" \
  --checkpoint "permissive224=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1/best.pt" \
  --output-manifest "${approved_manifest}" \
  --output-report "${audit_report}" \
  --confidence 0.80 \
  --minimum-total 100 \
  --minimum-classes 4 \
  --batch-size 128 \
  --workers 6 \
  --device cuda
