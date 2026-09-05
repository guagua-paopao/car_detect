#!/usr/bin/env bash
set -euo pipefail

stage191_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE191-TESLA-ZIP-IMAGE-AUDIT-R1
stage192_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE192-TESLA-TEACHER-QUOTA-R1
stage192_code=/root/autodl-tmp/vcas/code/stage192-tesla-teacher-quota-r1
training_code=/root/autodl-tmp/vcas/code/training
labels=/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json
python_bin=/root/miniconda3/bin/python

while [[ ! -f "$stage191_root/stage191-tesla-zip-image-audit.json" ]]; do
  sleep 20
done

"$python_bin" - "$stage191_root/stage191-tesla-zip-image-audit.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads(p.read_text(encoding="utf-8"))
if r.get("status") != "pass_component_pending_teacher_visibility_audit_no_training":
    raise SystemExit("Stage191 did not pass component audit")
if r.get("gates", {}).get("training_authorized") is not False:
    raise SystemExit("Stage191 unexpectedly authorized training")
PY

if [[ -e "$stage192_root" ]]; then
  echo "refusing to overwrite Stage192 output: $stage192_root" >&2
  exit 3
fi
mkdir -p "$stage192_root"

"$python_bin" "$stage192_code/prepare_stage192_tesla_teacher_inputs.py" \
  --manifest "$stage191_root/stage191-tesla-lighting-color-manifest.csv" \
  --body-output "$stage192_root/body-teacher-input.csv" \
  --color-output "$stage192_root/color-teacher-input.csv"

set +e
"$python_bin" "$training_code/scripts/audit_body_multiteacher_consensus.py" \
  --manifest "$stage192_root/body-teacher-input.csv" --dataset-root / --labels "$labels" \
  --checkpoint stage163_balanced=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE163-BODY-HARDCLASS-R1/ATTR-STAGE163-BODY-CONVNEXT-256-BALANCED-R1/best.pt \
  --checkpoint stage167_stretch=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1/best.pt \
  --output-manifest "$stage192_root/body-teacher-reviewed.csv" \
  --output-report "$stage192_root/body-teacher-report.json" \
  --confidence 0.55 --batch-size 48 --workers 4 --device cuda \
  >"$stage192_root/body-teacher.log" 2>&1
body_rc=$?
set -e
if [[ $body_rc -ne 0 && $body_rc -ne 2 ]] || [[ ! -s "$stage192_root/body-teacher-report.json" ]]; then
  echo "body teacher audit failed without complete evidence: rc=$body_rc" >&2
  exit 4
fi

set +e
"$python_bin" "$training_code/scripts/audit_color_multiteacher_consensus.py" \
  --manifest "$stage192_root/color-teacher-input.csv" --dataset-root / --labels "$labels" \
  --checkpoint stage158_256=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-256-FRESH-R1/best.pt \
  --checkpoint stage158_288=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-288-LETTERBOX-FRESH-R1/best.pt \
  --output-manifest "$stage192_root/color-teacher-reviewed.csv" \
  --output-report "$stage192_root/color-teacher-report.json" \
  --confidence 0.50 --minimum-total 100 --minimum-classes 4 \
  --batch-size 48 --workers 4 --device cuda \
  >"$stage192_root/color-teacher.log" 2>&1
color_rc=$?
set -e
if [[ $color_rc -ne 0 && $color_rc -ne 2 ]] || [[ ! -s "$stage192_root/color-teacher-report.json" ]]; then
  echo "color teacher audit failed without complete evidence: rc=$color_rc" >&2
  exit 5
fi

set +e
"$python_bin" "$stage192_code/merge_stage192_tesla_teacher_quota.py" \
  --stage191-manifest "$stage191_root/stage191-tesla-lighting-color-manifest.csv" \
  --stage191-report "$stage191_root/stage191-tesla-zip-image-audit.json" \
  --body-manifest "$stage192_root/body-teacher-reviewed.csv" \
  --body-report "$stage192_root/body-teacher-report.json" \
  --color-manifest "$stage192_root/color-teacher-reviewed.csv" \
  --color-report "$stage192_root/color-teacher-report.json" \
  --stage178-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE178-HEAD-SCENE-QUOTA-R1/stage178-head-scene-quota-audit.json \
  --stage179-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE179-FULL-COLOR-SCENE-AUDIT-R2/stage177-full-train-scene-audit.json \
  --stage188-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1/stage188-myvid-v2-semantics-decontamination.json \
  --output-manifest "$stage192_root/stage192-tesla-reviewed-manifest.csv" \
  --output-report "$stage192_root/stage192-tesla-teacher-quota-report.json" \
  >"$stage192_root/merge.log" 2>&1
merge_rc=$?
set -e

sha256sum \
  "$stage192_root/body-teacher-input.csv" "$stage192_root/color-teacher-input.csv" \
  "$stage192_root/body-teacher-reviewed.csv" "$stage192_root/body-teacher-report.json" \
  "$stage192_root/color-teacher-reviewed.csv" "$stage192_root/color-teacher-report.json" \
  "$stage192_root/stage192-tesla-reviewed-manifest.csv" \
  "$stage192_root/stage192-tesla-teacher-quota-report.json" \
  >"$stage192_root/SHA256SUMS"

echo "STAGE192_TERMINAL merge_rc=$merge_rc body_rc=$body_rc color_rc=$color_rc"
exit "$merge_rc"
