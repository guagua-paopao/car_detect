#!/usr/bin/env bash
set -euo pipefail

training_session="vcas_stage50_type"
project_root="/root/autodl-tmp/vcas/code"
scripts_root="${project_root}/training/scripts"
labels_path="${project_root}/config/vehicle_labels.v1.json"
run_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE50-VTID2-OOF-TYPE-256-R1"
checkpoint_path="${run_root}/best.pt"
hard_manifest="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv"
ua_manifest="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv"
vfg_manifest="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv"
python_bin="/root/miniconda3/bin/python"

while tmux has-session -t "${training_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${run_root}/metrics.json" "${run_root}/test_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, test_path = map(Path, sys.argv[1:])
if not metrics_path.is_file() or not test_path.is_file():
    raise SystemExit("fail_closed: completed training evidence is missing")
test_metrics = json.loads(test_path.read_text(encoding="utf-8"))
if test_metrics.get("status") != "not_run":
    raise SystemExit("fail_closed: iterative candidate unexpectedly evaluated the test split")
PY

test -f "${checkpoint_path}"
test ! -e "${run_root}/hard-validation-sweep.json"
test ! -e "${run_root}/ua-validation-fusion-sweep.json"
test ! -e "${run_root}/vfg7-validation-comparison"

thresholds=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99 0.995)

"${python_bin}" "${scripts_root}/evaluate_attribute_baseline.py" \
  --manifest "${hard_manifest}" \
  --checkpoint "${checkpoint_path}" \
  --labels "${labels_path}" \
  --output "${run_root}/hard-validation-sweep.json" \
  --split validation \
  --batch-size 128 \
  --workers 4 \
  --device cuda \
  --type-threshold 0.97 \
  --color-threshold 0.40 \
  --threshold-sweep "${thresholds[@]}"

"${python_bin}" "${scripts_root}/sweep_attribute_track_fusion.py" \
  --manifest "${ua_manifest}" \
  --checkpoint "${checkpoint_path}" \
  --output "${run_root}/ua-validation-fusion-sweep.json" \
  --split validation \
  --minimum-window-frames 3 \
  --thresholds "${thresholds[@]}" \
  --windows 3 5 \
  --minimum-shares 0.50 0.60 0.70 \
  --minimum-margins 0.05 0.10 0.15 \
  --batch-size 128 \
  --workers 4 \
  --device cuda

"${python_bin}" "${scripts_root}/run_vfg7_validation_comparison.py" \
  --manifest "${vfg_manifest}" \
  --labels "${labels_path}" \
  --scripts-root "${scripts_root}" \
  --output-dir "${run_root}/vfg7-validation-comparison" \
  --python "${python_bin}" \
  --model "production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --model "stage47=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-STAGE47-HARDTYPE-UVH-256-R1/best.pt" \
  --model "stage50=${checkpoint_path}" \
  --batch-size 128 \
  --workers 4
