#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE_ROOT=/root/autodl-tmp/vcas/code
SCRIPT="$CODE_ROOT/training/scripts/evaluate_stage73_color_pseudolabel_rule.py"
MANIFEST=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-color.csv
DATASET_ROOT=/root/autodl-tmp/vcas/datasets
LABELS="$CODE_ROOT/config/vehicle_labels.v1.json"
PRODUCTION=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
STAGE71_COLOR=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/best.pt
OUTPUT_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE73-COLOR-RULE-VALIDATION-V1
REPORT="$OUTPUT_ROOT/report.json"
STATE="$OUTPUT_ROOT/state.json"
LOG="$OUTPUT_ROOT/run.log"

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print toupper($1)}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  fi
}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite validation evidence: $OUTPUT_ROOT" >&2
  exit 65
fi

assert_sha256 "$SCRIPT" 507A376F29484C361F3F7BB24F4237AE9CB890354C3C468DBD55B66BD22A074F
assert_sha256 "$MANIFEST" C20AFDE14A89494E26B7CB20C98E9C1ABAD3DB7E776C1E3751D8A81CD46A2C9E
assert_sha256 "$LABELS" C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F
assert_sha256 "$PRODUCTION" 6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383
assert_sha256 "$STAGE71_COLOR" 14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121

mkdir -p "$OUTPUT_ROOT"
set +e
"$PYTHON" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --labels "$LABELS" \
  --expected-labels-sha256 C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F \
  --checkpoint "production=$PRODUCTION" \
  --checkpoint "stage71_color=$STAGE71_COLOR" \
  --output "$REPORT" \
  --batch-size 128 \
  --workers 8 \
  --device cuda >"$LOG" 2>&1
rc=$?
set -e

report_sha=""
if [[ -f "$REPORT" ]]; then
  report_sha="$(sha256sum "$REPORT" | awk '{print toupper($1)}')"
  printf '%s  %s\n' "$report_sha" "$(basename "$REPORT")" >"$REPORT.sha256"
fi

"$PYTHON" -c 'import json,sys; from pathlib import Path; rc=int(sys.argv[2]); status="pass_validation_rule_available" if rc == 0 else ("fail_closed_no_rule_passed" if rc == 2 else "failed_runtime"); Path(sys.argv[1]).write_text(json.dumps({"status":status,"exit_code":rc,"report":sys.argv[3],"report_sha256":sys.argv[4],"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"deployment_performed":False},indent=2)+"\n",encoding="utf-8")' "$STATE" "$rc" "$REPORT" "$report_sha"
exit "$rc"
