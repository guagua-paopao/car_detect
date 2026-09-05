#!/usr/bin/env bash
set -euo pipefail

stage191_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE191-TESLA-ZIP-IMAGE-AUDIT-R1
stage192_r1=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE192-TESLA-TEACHER-QUOTA-R1
stage192_r2=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE192-TESLA-TEACHER-QUOTA-R2
code=/root/autodl-tmp/vcas/code/stage192-tesla-teacher-quota-r2
python_bin=/root/miniconda3/bin/python

for required in \
  "$stage191_root/stage191-tesla-lighting-color-manifest.csv" \
  "$stage191_root/stage191-tesla-zip-image-audit.json" \
  "$stage192_r1/body-teacher-reviewed.csv" "$stage192_r1/body-teacher-report.json" \
  "$stage192_r1/color-teacher-reviewed.csv" "$stage192_r1/color-teacher-report.json"; do
  [[ -s "$required" ]] || { echo "missing required evidence: $required" >&2; exit 3; }
done
[[ ! -e "$stage192_r2" ]] || { echo "refusing to overwrite $stage192_r2" >&2; exit 4; }
mkdir -p "$stage192_r2"

set +e
"$python_bin" "$code/merge_stage192_tesla_teacher_quota.py" \
  --stage191-manifest "$stage191_root/stage191-tesla-lighting-color-manifest.csv" \
  --stage191-report "$stage191_root/stage191-tesla-zip-image-audit.json" \
  --body-manifest "$stage192_r1/body-teacher-reviewed.csv" \
  --body-report "$stage192_r1/body-teacher-report.json" \
  --color-manifest "$stage192_r1/color-teacher-reviewed.csv" \
  --color-report "$stage192_r1/color-teacher-report.json" \
  --stage178-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE178-HEAD-SCENE-QUOTA-R1/stage178-head-scene-quota-audit.json \
  --stage179-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE179-FULL-COLOR-SCENE-AUDIT-R2/stage177-full-train-scene-audit.json \
  --stage188-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE188-MYVID-SEMANTICS-DECONTAMINATION-R1/stage188-myvid-v2-semantics-decontamination.json \
  --output-manifest "$stage192_r2/stage192-tesla-reviewed-manifest.csv" \
  --output-report "$stage192_r2/stage192-tesla-teacher-quota-report.json" \
  >"$stage192_r2/merge.log" 2>&1
merge_rc=$?
set -e

for output in "$stage192_r2/stage192-tesla-reviewed-manifest.csv" "$stage192_r2/stage192-tesla-teacher-quota-report.json"; do
  [[ -s "$output" ]] || { echo "merge failed without sealed output: rc=$merge_rc" >&2; exit 5; }
done
sha256sum "$stage192_r2/stage192-tesla-reviewed-manifest.csv" "$stage192_r2/stage192-tesla-teacher-quota-report.json" >"$stage192_r2/SHA256SUMS"
echo "STAGE192_R2_TERMINAL merge_rc=$merge_rc"
exit "$merge_rc"
