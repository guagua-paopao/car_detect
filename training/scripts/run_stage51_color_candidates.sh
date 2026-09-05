#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
train_script="${project_root}/training/scripts/train_attribute.py"
labels_path="${project_root}/config/vehicle_labels.v1.json"
dataset_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2"
runs_root="/root/autodl-tmp/vcas/runs/attributes"
research_manifest="${dataset_root}/attribute_manifest.stage51-cc0-color-research-v2.csv"
permissive_manifest="${dataset_root}/attribute_manifest.stage51-cc0-color-permissive-v2.csv"
research_report="${dataset_root}/kaggle-color-cc0-v1/stage51-research-v2-manifest-report.json"
permissive_report="${dataset_root}/kaggle-color-cc0-v1/stage51-permissive-v2-manifest-report.json"
production_checkpoint="/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
stage44_checkpoint="/root/autodl-tmp/vcas/runs/attributes/ATTR-DOMAIN-BMD45-RAW-V34-256-R1/best.pt"

"${python_bin}" - "${research_report}" "${permissive_report}" <<'PY'
import json
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise SystemExit(f"fail_closed: manifest report did not pass: {path}")
    if report.get("baseline_eval_exact_digest") != report.get("output_eval_exact_digest"):
        raise SystemExit(f"fail_closed: validation/test rows changed: {path}")
PY

test -f "${research_manifest}"
test -f "${permissive_manifest}"
test -f "${production_checkpoint}"
test -f "${stage44_checkpoint}"

research224_output="${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1"
research256_output="${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-256-R1"
permissive224_output="${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1"
test ! -e "${research224_output}"
test ! -e "${research256_output}"
test ! -e "${permissive224_output}"

echo "START ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1"
"${python_bin}" "${train_script}" \
  --manifest "${research_manifest}" --labels "${labels_path}" \
  --input-size 224 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 96 --workers 6 --learning-rate 0.00005 --weight-decay 0.0001 \
  --body-loss-weight 0.0 --color-loss-weight 1.0 --focal-gamma 0.0 --color-focal-gamma 1.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --freeze-backbone-epochs 1 --patience 4 \
  --type-threshold 0.97 --color-threshold 0.70 \
  --gate-type-precision 0.93 --gate-type-coverage 0.45 --gate-color-precision 0.93 --gate-color-coverage 0.25 \
  --init-checkpoint "${production_checkpoint}" --augmentation-profile color_scene --selection-head color \
  --hard-sample-weight 0.2 --small-sample-weight 0.2 --color-sample-weight 0.5 \
  --run-kind formal --dataset-version stage51-cc0-color-research-v2 \
  --code-revision stage51-color-specialist-v2 --skip-test --output-dir "${research224_output}" \
  > "${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1.log" 2>&1
echo "DONE ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-224-R1"

echo "START ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-256-R1"
"${python_bin}" "${train_script}" \
  --manifest "${research_manifest}" --labels "${labels_path}" \
  --input-size 256 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 72 --workers 6 --learning-rate 0.00004 --weight-decay 0.0001 \
  --body-loss-weight 0.0 --color-loss-weight 1.0 --focal-gamma 0.0 --color-focal-gamma 1.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --freeze-backbone-epochs 1 --patience 4 \
  --type-threshold 0.97 --color-threshold 0.70 \
  --gate-type-precision 0.93 --gate-type-coverage 0.45 --gate-color-precision 0.93 --gate-color-coverage 0.25 \
  --init-checkpoint "${stage44_checkpoint}" --augmentation-profile color_scene --selection-head color \
  --hard-sample-weight 0.2 --small-sample-weight 0.2 --color-sample-weight 0.5 \
  --run-kind formal --dataset-version stage51-cc0-color-research-v2 \
  --code-revision stage51-color-specialist-v2 --skip-test --output-dir "${research256_output}" \
  > "${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-256-R1.log" 2>&1
echo "DONE ATTR-DOMAIN-STAGE51-CC0-COLOR-RESEARCH-256-R1"

echo "START ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1"
"${python_bin}" "${train_script}" \
  --manifest "${permissive_manifest}" --labels "${labels_path}" \
  --input-size 224 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 64 --workers 4 --learning-rate 0.0001 --weight-decay 0.0001 \
  --body-loss-weight 0.0 --color-loss-weight 1.0 --focal-gamma 0.0 --color-focal-gamma 1.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --freeze-backbone-epochs 8 --patience 4 \
  --type-threshold 0.97 --color-threshold 0.70 \
  --gate-type-precision 0.93 --gate-type-coverage 0.45 --gate-color-precision 0.93 --gate-color-coverage 0.25 \
  --init-checkpoint "${production_checkpoint}" --augmentation-profile color_scene --selection-head color \
  --hard-sample-weight 0.2 --small-sample-weight 0.2 --color-sample-weight 0.5 \
  --run-kind formal --dataset-version stage51-cc0-color-permissive-v2 \
  --code-revision stage51-color-specialist-v2 --skip-test --output-dir "${permissive224_output}" \
  > "${runs_root}/ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1.log" 2>&1
echo "DONE ATTR-DOMAIN-STAGE51-CC0-COLOR-PERMISSIVE-224-R1"
