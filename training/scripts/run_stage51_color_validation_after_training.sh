#!/usr/bin/env bash
set -euo pipefail

training_session="vcas_stage51_color"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
scripts_root="${project_root}/training/scripts"
labels_path="${project_root}/config/vehicle_labels.v1.json"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
runs_root="/root/autodl-tmp/vcas/runs/attributes"
hard_manifest="${dataset_root}/attribute_manifest.bmd-raw-v34-hard-v2.csv"
vfg_manifest="${dataset_root}/vfg7-eval-v1/attribute_manifest.csv"
validation_root="${runs_root}/ATTR-DOMAIN-STAGE51-COLOR-VALIDATION-V1"
research224="${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1"
research256="${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-256-R1"
permissive224="${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1"

# Prefix matching would make this validation session match the completed
# `vcas_stage51_color` session name and wait on itself forever.  The leading
# `=` requests an exact tmux session-name match.
while tmux has-session -t "=${training_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${research224}" "${research256}" "${permissive224}" <<'PY'
import json
import sys
from pathlib import Path

for raw_root in sys.argv[1:]:
    root = Path(raw_root)
    if not (root / "best.pt").is_file():
        raise SystemExit(f"fail_closed: candidate checkpoint missing: {root}")
    test_path = root / "test_metrics.json"
    if not test_path.is_file():
        raise SystemExit(f"fail_closed: test-policy evidence missing: {root}")
    if json.loads(test_path.read_text(encoding="utf-8")).get("status") != "not_run":
        raise SystemExit(f"fail_closed: iterative candidate used the test split: {root}")
PY

test ! -e "${validation_root}"
mkdir -p "${validation_root}"
thresholds=(0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.72 0.74 0.75 0.76 0.78 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99)

"${python_bin}" "${scripts_root}/run_vfg7_validation_comparison.py" \
  --manifest "${vfg_manifest}" --labels "${labels_path}" --scripts-root "${scripts_root}" \
  --output-dir "${validation_root}/vfg7" --python "${python_bin}" \
  --model "production=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt" \
  --model "stage44=/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-BMD45-RAW-V34-256-R1/best.pt" \
  --model "research224=${research224}/best.pt" \
  --model "research256=${research256}/best.pt" \
  --model "permissive224=${permissive224}/best.pt" \
  --batch-size 128 --workers 4

for spec in \
  "research224=${research224}/best.pt" \
  "research256=${research256}/best.pt" \
  "permissive224=${permissive224}/best.pt"
do
  candidate_name="${spec%%=*}"
  checkpoint_path="${spec#*=}"
  "${python_bin}" "${scripts_root}/evaluate_attribute_baseline.py" \
    --manifest "${hard_manifest}" --checkpoint "${checkpoint_path}" --labels "${labels_path}" \
    --output "${validation_root}/${candidate_name}.hard-validation.json" --split validation \
    --batch-size 128 --workers 4 --device cuda --type-threshold 0.97 --color-threshold 0.70 \
    --threshold-sweep "${thresholds[@]}"
done

echo "stage51 color validation complete"
